
import json
import sys
import uuid
from math import floor
from pathlib import Path

import hnswlib
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd
from einops import rearrange
from gymnasium import Env, spaces
from numpy.typing import NDArray
from pyboy import PyBoy
from pyboy.logger import log_level
from pyboy.utils import WindowEvent
from skimage.transform import resize
from typing import Any, Optional, TypedDict


class RedGymEnvConfig(TypedDict):
    debug: bool
    session_path: Path
    save_final_state: bool
    print_rewards: bool
    headless: bool
    init_state: str
    action_freq: int
    max_steps: int
    early_stop: bool
    save_video: bool
    fast_video: bool
    explore_weight: Optional[float]
    use_screen_explore: Optional[bool]
    sim_frame_dist: float
    reward_scale: Optional[float]
    extra_buttons: Optional[bool]
    instance_id: Optional[str]


class _AgentStats(TypedDict):
    step: int
    x: int
    y: int
    map: int
    map_location: str
    last_action: int
    pcount: int
    levels: list[int]
    levels_sum: int
    ptypes: int
    hp: float
    frames: Optional[int]
    coord_count: Optional[int]
    deaths: int
    badge: int
    event: float
    healr: float


class RedGymEnv(Env):

    def __init__(self, config: Optional[RedGymEnvConfig] = None):

        self._debug: bool = config['debug'] # unused
        self._s_path: Path = config['session_path']
        self._save_final_state: bool = config['save_final_state']
        self._print_rewards: bool = config['print_rewards']
        self._vec_dim: int = 4320 #1000
        self._headless: bool = config['headless']
        self._num_elements: int = 20000 # max
        self._init_state: str = config['init_state']
        self._act_freq: int = config['action_freq']
        self._max_steps: int = config['max_steps']
        self._early_stopping: bool = config['early_stop']
        self._save_video: bool = config['save_video']
        self._fast_video: bool = config['fast_video']
        self._frame_stacks: int = 3
        self._explore_weight: float = 1 if 'explore_weight' not in config else config['explore_weight']
        self._use_screen_explore: bool = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self._similar_frame_dist: float = config['sim_frame_dist']
        self._reward_scale: float = 1 if 'reward_scale' not in config else config['reward_scale']
        self._extra_buttons: bool = False if 'extra_buttons' not in config else config['extra_buttons']
        self._instance_id: str = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self._s_path.mkdir(exist_ok=True)
        self._reset_count: int = 0
        self._all_runs: list[dict[str, float]] = []

        # Set this in SOME subclasses
        self.metadata: dict[str, Any] = {"render.modes": []}
        self.reward_range: tuple[int, int] = (0, 15000)

        self.valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        
        if self._extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self._release_arrow: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self._release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self._output_shape: tuple[int, int, int] = (36, 40, 3)
        self._mem_padding: int = 2
        self._memory_height: int = 8
        self._col_steps: int = 16
        self._output_full: tuple[int, int, int] = (
            self._output_shape[0] * self._frame_stacks + 2 * (self._mem_padding + self._memory_height),
            self._output_shape[1],
            self._output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self._output_full, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        log_level("ERROR")
        self._pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self._screen = self._pyboy.botsupport_manager().screen()

        if not config['headless']:
            self._pyboy.set_emulation_speed(6)

        # Fields set in reset()
        self._recent_memory: NDArray = np.zeros((1,), dtype=np.uint8)
        self._recent_frames: NDArray = np.zeros((1,), dtype=np.uint8)
        self._agent_stats: list[_AgentStats] = []
        self._full_frame_writer: Optional[media.VideoWriter] = None
        self._model_frame_writer: Optional[media.VideoWriter] = None
        self._levels_satisfied: bool = False
        self._base_explore: int = 0
        self._max_opponent_level: int = 0
        self._max_event_rew: int = 0
        self._max_level_rew: float = 0.
        self._last_health: float = 0.
        self._total_healing_rew: float = 0.
        self._died_count: int = 0
        self._party_size: int = 0
        self._step_count: int = 0
        self._progress_reward: dict[str, float] = {}
        self._total_reward: float = 0.
        self._seen_coords: dict[str, int] = {}
        self._knn_index: Optional[hnswlib.Index] = None
            
        self.reset()

    def reset(self, *,
              seed: int | None = None,
              options: dict[str, Any] | None = None, ) -> tuple[NDArray, dict[str, Any]]:
        # restart game, skipping credits
        with open(self._init_state, "rb") as f:
            self._pyboy.load_state(f)
        
        if self._use_screen_explore:
            self._init_knn()
        else:
            self._init_map_mem()

        self._recent_memory = np.zeros((self._output_shape[1] * self._memory_height, 3), dtype=np.uint8)
        
        self._recent_frames = np.zeros(
            (self._frame_stacks, self._output_shape[0],
             self._output_shape[1], self._output_shape[2]),
            dtype=np.uint8)

        self._agent_stats = []
        
        if self._save_video:
            base_dir = self._s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self._reset_count}_id{self._instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self._reset_count}_id{self._instance_id}').with_suffix('.mp4')
            self._full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self._full_frame_writer.__enter__()
            self._model_frame_writer = media.VideoWriter(base_dir / model_name, self._output_full[:2], fps=60)
            self._model_frame_writer.__enter__()
       
        self._levels_satisfied = False
        self._base_explore = 0
        self._max_opponent_level = 0
        self._max_event_rew = 0
        self._max_level_rew = 0
        self._last_health = 1
        self._total_healing_rew = 0
        self._died_count = 0
        self._party_size = 0
        self._step_count = 0
        self._progress_reward: dict[str, float] = self._get_game_state_reward()
        self._total_reward = sum([val for _, val in self._progress_reward.items()])
        self._reset_count += 1
        return self.render(), {}
    
    def _init_knn(self) -> None:
        # Declaring index
        self._knn_index = hnswlib.Index(space='l2', dim=self._vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self._knn_index.init_index(
            max_elements=self._num_elements, ef_construction=100, M=16)
        
    def _init_map_mem(self):
        self._seen_coords = {}

    def render(self, reduce_res: bool = True, add_memory: bool = True, update_mem: bool = True) -> NDArray:
        game_pixels_render = self._screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self._output_shape)).astype(np.uint8)
            if update_mem:
                self._recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self._mem_padding, self._output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self._create_exploration_memory(),
                        pad,
                        self._create_recent_memory(),
                        pad,
                        rearrange(self._recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    
    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict[str, Any]]:

        self._run_action_on_emulator(action)
        self._append_agent_stats(action)

        self._recent_frames = np.roll(self._recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self._memory_height + self._mem_padding)
        obs_flat = obs_memory[
            frame_start:frame_start+self._output_shape[0], ...].flatten().astype(np.float32)

        if self._use_screen_explore:
            self._update_frame_knn_index(obs_flat)
        else:
            self._update_seen_coords()
            
        self._update_heal_reward()
        self._party_size = self._read_m(0xD163)

        new_reward, new_prog = self._update_reward()
        
        self._last_health = self._read_hp_fraction()

        # shift over short term reward memory
        self._recent_memory = np.roll(self._recent_memory, 3)
        self._recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self._recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self._recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self._save_and_print_info(step_limit_reached, obs_memory)

        self._step_count += 1

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}

    def _run_action_on_emulator(self, action: int) -> None:
        # press button then release after some steps
        self._pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self._save_video and self._headless:
            self._pyboy._rendering(False)
        for i in range(self._act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self._pyboy.send_input(self._release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self._pyboy.send_input(self._release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self._pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self._save_video and not self._fast_video:
                self._add_video_frame()
            if i == self._act_freq-1:
                self._pyboy._rendering(True)
            self._pyboy.tick()
        if self._save_video and self._fast_video:
            self._add_video_frame()
    
    def _add_video_frame(self) -> None:
        self._full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self._model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))
    
    def _append_agent_stats(self, action: int) -> None:
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)
        levels = [self._read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        if self._use_screen_explore:
            expl = ('frames', self._knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self._seen_coords))
        self._agent_stats.append({
            'step': self._step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'map_location': self._get_map_location(map_n),
            'last_action': action,
            'pcount': self._read_m(0xD163),
            'levels': levels, 
            'levels_sum': sum(levels),
            'ptypes': self._read_party(),
            'hp': self._read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self._died_count, 'badge': self._get_badges(),
            'event': self._progress_reward['event'], 'healr': self._total_healing_rew
        })

    def _update_frame_knn_index(self, frame_vec: NDArray) -> None:
        
        if self._get_levels_sum() >= 22 and not self._levels_satisfied:
            self._levels_satisfied = True
            self._base_explore = self._knn_index.get_current_count()
            self._init_knn()

        if self._knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self._knn_index.add_items(
                frame_vec, np.array([self._knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self._knn_index.knn_query(frame_vec, k=1, filter=None)
            if distances[0][0] > self._similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self._knn_index.add_items(
                    frame_vec, np.array([self._knn_index.get_current_count()])
                )
    
    def _update_seen_coords(self) -> None:
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self._get_levels_sum() >= 22 and not self._levels_satisfied:
            self._levels_satisfied = True
            self._base_explore = len(self._seen_coords)
            self._seen_coords = {}
        
        self._seen_coords[coord_string] = self._step_count

    def _update_reward(self) -> tuple[float, tuple[float, float, float]]:
        # compute reward
        old_prog = self._group_rewards()
        self._progress_reward = self._get_game_state_reward()
        new_prog = self._group_rewards()
        new_total = sum([val for _, val in self._progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self._total_reward
        if new_step < 0 and self._read_hp_fraction() > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self._save_screenshot('neg_reward')
    
        self._total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )
    
    def _group_rewards(self) -> tuple[float, float, float]:
        prog = self._progress_reward
        # these values are only used by memory
        return (prog['level'] * 100 / self._reward_scale,
                self._read_hp_fraction() * 2000,
                prog['explore'] * 150 / (self._explore_weight * self._reward_scale))
               #(prog['events'], 
               # prog['levels'] + prog['party_xp'], 
               # prog['explore'])

    def _create_exploration_memory(self) -> NDArray:
        """
        Prepares the part of image with colored bars for level, hp and exploration rewards.
        :return: a [h, w, 3] NDArray(uint8) interpretable as an image
        """
        w = self._output_shape[1]
        h = self._memory_height
        
        def make_reward_channel(r_val: float) -> NDArray:
            col_steps = self._col_steps
            max_r_val = (w-1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self._group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self._get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def _create_recent_memory(self) -> NDArray:
        return rearrange(
            self._recent_memory,
            '(w h) c -> h w c', 
            h=self._memory_height)

    def check_if_done(self) -> bool:
        if self._early_stopping:
            done = False
            if self._step_count > 128 and self._recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self._step_count >= self._max_steps
        #done = self.read_hp_fraction() == 0
        return done

    def _save_and_print_info(self, done: bool, obs_memory: NDArray) -> None:
        if self._print_rewards:
            prog_string = f'step: {self._step_count:6d}'
            for key, val in self._progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self._total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self._step_count % 50 == 0:
            plt.imsave(
                self._s_path / Path(f'curframe_{self._instance_id}.jpeg'),
                self.render(reduce_res=False))

        if self._print_rewards and done:
            print('', flush=True)
            if self._save_final_state:
                fs_path = self._s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self._total_reward:.4f}_{self._reset_count}_small.jpeg'),
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self._total_reward:.4f}_{self._reset_count}_full.jpeg'),
                    self.render(reduce_res=False))

        if self._save_video and done:
            self._full_frame_writer.close()
            self._model_frame_writer.close()

        if done:
            self._all_runs.append(self._progress_reward)
            with open(self._s_path / Path(f'all_runs_{self._instance_id}.json'), 'w') as f:
                json.dump(self._all_runs, f)
            pd.DataFrame(self._agent_stats).to_csv(
                self._s_path / Path(f'agent_stats_{self._instance_id}.csv.gz'), compression='gzip', mode='a')
    
    def _read_m(self, addr: int) -> int:
        return self._pyboy.get_memory_value(addr)

    def _read_bit(self, addr: int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self._read_m(addr))[-bit - 1] == '1'
    
    def _get_levels_sum(self) -> int:
        poke_levels = [max(self._read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def _get_levels_reward(self) -> float:
        explore_thresh = 22
        scale_factor = 4
        level_sum = self._get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self._max_level_rew = max(self._max_level_rew, scaled)
        return self._max_level_rew
    
    def _get_knn_reward(self) -> float:
        
        pre_rew = self._explore_weight * 0.005
        post_rew = self._explore_weight * 0.01
        cur_size = self._knn_index.get_current_count() if self._use_screen_explore else len(self._seen_coords)
        base = (self._base_explore if self._levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self._levels_satisfied else 0) * post_rew
        return base + post
    
    def _get_badges(self) -> int:
        return self._bit_count(self._read_m(0xD356))

    def _read_party(self) -> list[int]:
        return [self._read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
    
    def _update_heal_reward(self) -> None:
        cur_health = self._read_hp_fraction()
        # if health increased and party size did not change
        if (cur_health > self._last_health and
                self._read_m(0xD163) == self._party_size):
            if self._last_health > 0:
                heal_amount = cur_health - self._last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self._save_screenshot('healing')
                self._total_healing_rew += heal_amount * 4
            else:
                self._died_count += 1

    def _get_all_events_reward(self) -> int:
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self._bit_count(self._read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self._read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def _get_game_state_reward(self) -> dict[str, float]:
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = self.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        
        state_scores: dict[str, float] = {
            'event': self._reward_scale * self._update_max_event_rew(),
            #'party_xp': self.reward_scale*0.1*sum(poke_xps),
            'level': self._reward_scale * self._get_levels_reward(),
            'heal': self._reward_scale * self._total_healing_rew,
            'op_lvl': self._reward_scale * self._update_max_op_level(),
            'dead': self._reward_scale * -0.1 * self._died_count,
            'badge': self._reward_scale * self._get_badges() * 5,
            #'op_poke': self.reward_scale*self.max_opponent_poke * 800,
            #'money': self.reward_scale* money * 3,
            #'seen_poke': self.reward_scale * seen_poke_count * 400,
            'explore': self._reward_scale * self._get_knn_reward()
        }
        
        return state_scores
    
    def _save_screenshot(self, name: str) -> None:
        ss_dir = self._s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self._instance_id}_r{self._total_reward:.4f}_{self._reset_count}_{name}.jpeg'),
            self.render(reduce_res=False))
    
    def _update_max_op_level(self) -> float:
        #opponent_level = self.read_m(0xCFE8) - 5 # base level
        opponent_level = max([self._read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self._max_opponent_level = max(self._max_opponent_level, opponent_level)
        return self._max_opponent_level * 0.2
    
    def _update_max_event_rew(self) -> int:
        cur_rew = self._get_all_events_reward()
        self._max_event_rew = max(cur_rew, self._max_event_rew)
        return self._max_event_rew

    def _read_hp_fraction(self) -> float:
        hp_sum = sum([self._read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self._read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def _read_hp(self, start: int) -> int:
        return 256 * self._read_m(start) + self._read_m(start + 1)

    # built-in since python 3.10
    @staticmethod
    def _bit_count(bits: int) -> int:
        return bin(bits).count('1')

    def _read_triple(self, start_add: int) -> int:
        return 256*256*self._read_m(start_add) + 256*self._read_m(start_add + 1) + self._read_m(start_add + 2)
    
    @staticmethod
    def _from_bcd(num: int) -> int:
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def _read_money(self) -> int:
        return (100 * 100 * self._from_bcd(self._read_m(0xD347)) +
                100 * self._from_bcd(self._read_m(0xD348)) +
                self._from_bcd(self._read_m(0xD349)))

    @staticmethod
    def _get_map_location(map_idx: int) -> str:
        map_locations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            37: "Red house first",
            38: "Red house second",
            39: "Blues house",
            40: "oaks lab",
            41: "Pokémon Center (Viridian City)",
            42: "Poké Mart (Viridian City)",
            43: "School (Viridian City)",
            44: "House 1 (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            52: "Pewter Museum (floor 1)",
            53: "Pewter Museum (floor 2)",
            54: "Pokémon Gym (Pewter City)",
            55: "House with disobedient Nidoran♂ (Pewter City)",
            56: "Poké Mart (Pewter City)",
            57: "House with two Trainers (Pewter City)",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)"
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return "Unknown Location"
    
