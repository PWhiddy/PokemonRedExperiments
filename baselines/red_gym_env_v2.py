import uuid
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

event_flags_start = 0xD747
event_flags_end = 0xD7F6 # 0xD761 # 0xD886 temporarily lower event flag range for obs input
museum_ticket = (0xD754, 0)

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
            }
        )

        head = "headless" if config["headless"] else "SDL2"

        log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            debugging=False,
            disable_input=False,
            window_type=head,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = 384
        self.explore_map = np.zeros((self.explore_map_dim,self.explore_map_dim), dtype=np.uint8)

        self.recent_screens = np.zeros( self.output_shape, dtype=np.uint8)
        
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray()[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):
        
        screen = self.render()

        self.update_recent_screens(screen)
        
        # normalize to approx 0-1
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions
        }

        return observation

    def step(self, action):

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        self.party_size = self.read_m(0xD163)

        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        if step_limit_reached:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                # release button
                self.pyboy.send_input(self.release_actions[action])
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": self.get_map_location(map_n),
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad*4, self.coords_pad*4), 
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:,:,0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:,:,0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        self.seen_coords[coord_string] = self.step_count

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        c = (np.array([x_pos,-y_pos])
        + self.get_map_location(map_n)["coordinates"]
        + self.coords_pad*2)
        return self.explore_map.shape[0]-c[1], c[0]

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0]-self.coords_pad:c[0]+self.coords_pad,
                c[1]-self.coords_pad:c[1]+self.coords_pad
            ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:,:, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:,:, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

        """
        if done:
            self.all_runs.append(self.progress_reward)
            with open(
                self.s_path / Path(f"all_runs_{self.instance_id}.json"), "w"
            ) as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                compression="gzip",
                mode="a",
            )
        """

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end) 
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 6,
            "level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew * 4,
            "op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            "dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 5,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.005,
        }

        return state_scores

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount
            else:
                self.died_count += 1

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
    
    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))
    
    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))
    
    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_map_location(self, map_idx):
        map_locations = {
            0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
            1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
            2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
            3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
            62: {"name": "Invaded house (Cerulean City)", "coordinates": np.array([290, 227])},
            63: {"name": "trade house (Cerulean City)", "coordinates": np.array([290, 212])},
            64: {"name": "Pokémon Center (Cerulean City)", "coordinates": np.array([290, 197])},
            65: {"name": "Pokémon Gym (Cerulean City)", "coordinates": np.array([290, 182])},
            66: {"name": "Bike Shop (Cerulean City)", "coordinates": np.array([290, 167])},
            67: {"name": "Poké Mart (Cerulean City)", "coordinates": np.array([290, 152])},
            35: {"name": "Route 24", "coordinates": np.array([250, 235])},
            36: {"name": "Route 25", "coordinates": np.array([270, 267])},
            12: {"name": "Route 1", "coordinates": np.array([70, 43])},
            13: {"name": "Route 2", "coordinates": np.array([70, 151])},
            14: {"name": "Route 3", "coordinates": np.array([100, 179])},
            15: {"name": "Route 4", "coordinates": np.array([150, 197])},
            33: {"name": "Route 22", "coordinates": np.array([20, 71])},
            37: {"name": "Red house first", "coordinates": np.array([61, 9])},
            38: {"name": "Red house second", "coordinates": np.array([61, 0])},
            39: {"name": "Blues house", "coordinates": np.array([91, 9])},
            40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
            41: {"name": "Pokémon Center (Viridian City)", "coordinates": np.array([100, 54])},
            42: {"name": "Poké Mart (Viridian City)", "coordinates": np.array([100, 62])},
            43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
            44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
            47: {"name": "Gate (Viridian City/Pewter City) (Route 2)", "coordinates": np.array([91,143])},
            49: {"name": "Gate (Route 2)", "coordinates": np.array([91,115])},
            50: {"name": "Gate (Route 2/Viridian Forest) (Route 2)", "coordinates": np.array([91,115])},
            51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
            52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
            53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
            54: {"name": "Pokémon Gym (Pewter City)", "coordinates": np.array([49, 176])},
            55: {"name": "House with disobedient Nidoran♂ (Pewter City)", "coordinates": np.array([51, 184])},
            56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
            57: {"name": "House with two Trainers (Pewter City)", "coordinates": np.array([51, 184])},
            58: {"name": "Pokémon Center (Pewter City)", "coordinates": np.array([45, 161])},
            59: {"name": "Mt. Moon (Route 3 entrance)", "coordinates": np.array([153, 234])},
            60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
            61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
            68: {"name": "Pokémon Center (Route 3)", "coordinates": np.array([135, 197])},
            193: {"name": "Badges check gate (Route 22)", "coordinates": np.array([0, 87])}, # TODO this coord is guessed, needs to be updated
            230: {"name": "Badge Man House (Cerulean City)", "coordinates": np.array([290, 137])}
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return {"name": "Unknown", "coordinates": np.array([80, 0])} # TODO once all maps are added this case won't be needed
