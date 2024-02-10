import uuid
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
#from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

event_flags_start = 0xD747
event_flags_end = 0xD7F6 # 0xD761 # 0xD886 temporarily lower event flag range for obs input
museum_ticket = (0xD754, 0)

class PokeRedEnv(Env):
    def __init__(
            self, gb_path, init_state,
            max_steps=2048*8, headless=True,
            action_frequency=24, downscale_factor=2):
        
        self.headless = headless
        self.init_state = init_state
        self.act_freq = action_frequency
        self.max_steps = max_steps
        self.downscale_factor = downscale_factor

        self.pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=False,
            window_type="headless" if self.headless else "SDL2",
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        self.reset_count = 0

        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        self.coords_pad = 12

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 5000)

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

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=self._get_obs()["screen"].shape, dtype=np.uint8)
            }
        )

        if not self.headless:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=0, options={}):
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.explore_map_dim = 384
        self.explore_map = np.zeros((self.explore_map_dim,self.explore_map_dim), dtype=np.uint8)

        self.max_opponent_level = 0
        self.last_health = 1
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}
        self.agent_stats = []

        self.max_map_progress = 0
        self.reset_count += 1

        self.max_levels = 0
        self.max_coords = 0

        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray()[:,:,0]  # (144, 160)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(
                    game_pixels_render, (self.downscale_factor,self.downscale_factor))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):
        
        screen = self.render()

        observation = {
            "screen": screen,
        }

        return observation

    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)


        self.update_seen_coords()

        self.update_explore_map()

        self.party_size = self.read_m(0xD163)

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        #if step_limit_reached:
        if self.step_count % 100 == 0:
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

        level_sum = self.get_levels_sum()
        coord_count = len(self.seen_coords)
        new_rew = 0
        if level_sum > self.max_levels:
            new_rew += level_sum - self.max_levels
            self.max_levels = level_sum

        if coord_count > self.max_coords:
            new_rew += 0.04 * (coord_count - self.max_coords)
            self.max_coords = coord_count

        return obs, new_rew, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                # release button
                self.pyboy.send_input(self.release_actions[action])
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()

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
                "event": self.get_all_events_reward(),
                #"healr": self.total_healing_rew,
            }
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
    
    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

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

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                # healed
                pass
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
