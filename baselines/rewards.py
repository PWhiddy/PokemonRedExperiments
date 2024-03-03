import hnswlib
import numpy as np


class Reward:

    def __init__(self, config, reader):
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.reward_range = (0, 15000)
        self.reader = reader

        # Pokedex
        self.seen_pokemons = 0

        # Level
        self.max_level = 0
        self.levels_satisfied = False
        self.max_opponent_level = 0

        # Event
        self.max_event = 0

        # Health
        self.total_healing = 0
        self.last_party_size = 0
        self.last_health = 1
        self.died_count = 0

        # Explore
        self.base_explore = 0
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.vec_dim = 4320  # 1000
        self.num_elements = 20000  # max
        self.knn_index = None
        self.init_knn()
        self.seen_coords = {}
        self.init_map_mem()
        self.cur_size = 0

        self.last_game_state_rewards = self.get_game_state_rewards()
        self.total_reward = 0

    def reset(self):
        self.max_event = 0
        self.max_level = 0
        self.levels_satisfied = False
        self.total_healing = 0
        self.last_party_size = 0
        self.last_health = 1
        self.max_opponent_level = 0
        self.died_count = 0
        self.seen_pokemons = 0
        self.base_explore = 0
        self.seen_coords = {}
        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem()
        self.total_reward = 0
        self.last_game_state_rewards = self.get_game_state_rewards()

    def get_game_state_rewards(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        return {
            'event': self.reward_scale * self.max_event * 1,
            'level': self.reward_scale * self.compute_level_reward(),
            'heal': self.reward_scale * self.total_healing * 2,
            'op_lvl': self.reward_scale * self.max_opponent_level * 1,
            'dead': self.reward_scale * self.died_count * -0.1,
            'badge': self.reward_scale * self.reader.get_badges() * 5,
            'seen_poke': self.reward_scale * self.seen_pokemons * 1,
            'explore': self.reward_scale * self.compute_explore_reward()
        }

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)

    def init_map_mem(self):
        self.seen_coords = {}

    def update_exploration_reward(self):
        self.cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)

    def get_all_events_flags(self):
        # adds up all event flags, exclude museum ticket
        base_event_flags = 13
        return max(0, sum(self.reader.read_events()) - base_event_flags - int(self.reader.read_museum_tickets()))

    def compute_level_reward(self):
        # Levels count only quarter after 22 threshold
        return int(min(22, self.max_level) + (max(0, (self.max_level - 22)) / 4)) * 1

    def compute_explore_reward(self):
        pre_rew = 0.005
        post_rew = 0.01
        if not self.levels_satisfied:
            return (self.cur_size * 0.005) * self.explore_weight
        else:
            return ((self.base_explore * pre_rew) + (self.cur_size * post_rew)) * self.explore_weight

    def group_rewards_lvl_hp_explore(self, rewards):
        return (rewards['level'] * 100 / self.reward_scale,
                self.reader.read_hp_fraction() * 2000,
                rewards['explore'] * 150 / (self.explore_weight * self.reward_scale))

    def update_rewards(self, obs_flat, step_count):
        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords(step_count)
        self.update_exploration_reward()
        self.update_max_event()
        self.update_total_heal_and_death()
        self.update_max_op_level()
        self.update_seen_pokemons()
        self.update_max_level()

        return self.update_state_reward()

    def update_state_reward(self):
        # compute reward
        last_total = sum([val for _, val in self.last_game_state_rewards.items()])
        new_total = sum([val for _, val in self.get_game_state_rewards().items()])
        self.total_reward = new_total
        reward_delta = new_total - last_total

        self.last_game_state_rewards = self.get_game_state_rewards()

        # used by memory
        old_prog = self.group_rewards_lvl_hp_explore(self.last_game_state_rewards)
        new_prog = self.group_rewards_lvl_hp_explore(self.get_game_state_rewards())
        return reward_delta, (new_prog[0] - old_prog[0], new_prog[1] - old_prog[1], new_prog[2] - old_prog[2])

    def update_max_op_level(self):
        opponent_level = self.reader.get_opponent_level()
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)

    def update_seen_pokemons(self):
        initial_seen_pokemon = 3
        self.seen_pokemons = sum(self.reader.read_seen_pokemons()) - initial_seen_pokemon

    def update_max_level(self):
        # lvl can't decrease
        self.max_level = max(self.max_level, self.reader.get_levels_sum())

    def update_frame_knn_index(self, frame_vec):

        if self.reader.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current
            _, distances = self.knn_index.knn_query(frame_vec, k=1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_seen_coords(self, step_count):
        x_pos = self.reader.read_x_pos()
        y_pos = self.reader.read_y_pos()
        map_n = self.reader.read_map_n()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.reader.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}

        self.seen_coords[coord_string] = step_count

    def update_total_heal_and_death(self):
        cur_health = self.reader.read_hp_fraction()
        # if health increased and party size did not change
        if (cur_health > self.last_health and
                self.reader.read_party_size_address() == self.last_party_size):
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                self.total_healing += heal_amount
            else:
                self.died_count += 1
        self.last_party_size = self.reader.read_party_size_address()
        self.last_health = self.reader.read_hp_fraction()

    def update_max_event(self):
        cur_rew = self.get_all_events_flags()
        self.max_event = max(cur_rew, self.max_event)

    def print_rewards(self, step_count):
        prog_string = f'step: {step_count:6d}'
        rewards_state = self.get_game_state_rewards()
        for key, val in rewards_state.items():
            prog_string += f' {key}: {val:5.2f}'
        prog_string += f' sum: {self.total_reward:5.2f}'
        prog_string += f' {self.reader.get_map_location()}'
        print(f'\r{prog_string}', end='', flush=True)
