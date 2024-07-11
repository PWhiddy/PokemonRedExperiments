import numpy as np
import matplotlib.pyplot as plt
from math import floor
import mediapy as media
from pathlib import Path
from einops import rearrange
from skimage.transform import resize
from reader_pyboy import ReaderPyBoy


class Renderer:

    def __init__(self, s_path, pyboy, reward_service, instance_id):
        self.reward_service = reward_service
        self.instance_id = instance_id
        self.s_path = s_path
        self.s_path.mkdir(exist_ok=True)
        self.output_shape = (36, 40, 3)
        self.frame_stacks = 3
        self.mem_padding = 2
        self.memory_height = 8
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )
        self.col_steps = 16
        self.screen = pyboy.botsupport_manager().screen()
        self.reader = ReaderPyBoy(pyboy)

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3),
                    dtype=np.uint8)
                level, hp, explore = self.reward_service.group_rewards_lvl_hp_explore(self.reward_service.get_game_state_rewards())
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(level, hp, explore),
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render

    def save_and_print_info(self):
        plt.imsave(
            self.s_path / Path(f'curframe_{self.instance_id}.jpeg'),
            self.render(reduce_res=False))


    def save_screenshot(self, name, total_reward, reset_count):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(
                f'frame{self.instance_id}_r{total_reward:.4f}_{reset_count}_{name}.jpeg'),
            self.render(reduce_res=False)
        )

    def save_video(self, reset_count):
        base_dir = self.s_path / Path('rollouts')
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f'full_reset_{reset_count}_id{self.instance_id}').with_suffix('.mp4')
        model_name = Path(f'model_reset_{reset_count}_id{self.instance_id}').with_suffix('.mp4')
        self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
        self.model_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def get_obs_flat(self, obs_memory):
        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        return obs_memory[frame_start:frame_start + self.output_shape[0], ...].flatten().astype(np.float32)

    def create_recent_memory(self):
        return rearrange(self.recent_memory,'(w h) c -> h w c', h=self.memory_height)

    def create_exploration_memory(self, level, hp, explore):
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w - 1) * h * col_steps
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

        full_memory = np.stack(
            (make_reward_channel(level), make_reward_channel(hp), make_reward_channel(explore)),
            axis=-1)

        if self.reader.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def save_final_state(self, obs_memory, reset_count, total_reward):
        fs_path = self.s_path / Path('final_states')
        fs_path.mkdir(exist_ok=True)
        plt.imsave(fs_path / Path(f'frame_r{total_reward:.4f}_{reset_count}_small.jpeg'), obs_memory)
        plt.imsave(fs_path / Path(f'frame_r{total_reward:.4f}_{reset_count}_full.jpeg'), self.render(reduce_res=False))

    def reset(self):
        self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0],self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)
