from memory_addresses import EVENT_FLAGS_START_ADDRESS, EVENT_FLAGS_END_ADDRESS, MUSEUM_TICKET_ADDRESS
import hnswlib
import numpy as np


class Reward:

    def __init__(self, config, reader, save_screenshot):
        self.save_screenshot = save_screenshot
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
        self.explore_reward = 0

        self.last_game_state_rewards = self.get_game_state_rewards()
        self.total_reward = 0

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)

    def init_map_mem(self):
        self.seen_coords = {}

    def update_exploration_reward(self):
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        if not self.levels_satisfied:
            self.explore_reward = cur_size * pre_rew
        else:
            self.explore_reward = (self.base_explore * pre_rew) + (cur_size * post_rew)

    def get_all_events_flags(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = EVENT_FLAGS_START_ADDRESS
        event_flags_end = EVENT_FLAGS_END_ADDRESS
        museum_ticket = (MUSEUM_TICKET_ADDRESS, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.reader.bit_count(self.reader.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.reader.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
            )

    def get_game_state_rewards(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        return {
            'event': self.reward_scale * self.max_event * 1,
            'level': self.reward_scale * self.max_level * 1,
            'heal': self.reward_scale * self.total_healing * 4,
            'op_lvl': self.reward_scale * self.max_opponent_level * 1,
            'dead': self.reward_scale * self.died_count * -0.1,
            'badge': self.reward_scale * self.reader.get_badges() * 5,
            'explore': self.reward_scale * self.explore_reward,
            # 'party_xp': self.reward_scale*0.1*sum(poke_xps),
            # 'op_poke': self.reward_scale*self.max_opponent_poke * 800,
            # 'money': self.reward_scale* money * 3,
            'seen_poke': self.reward_scale * self.seen_pokemons
        }

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
        if reward_delta < 0 and self.reader.read_hp_fraction() > 0:
            self.save_screenshot('neg_reward')

        self.last_game_state_rewards = self.get_game_state_rewards()

        # used by memory
        old_prog = self.group_rewards_lvl_hp_explore(self.last_game_state_rewards)
        new_prog = self.group_rewards_lvl_hp_explore(self.get_game_state_rewards())
        return reward_delta, (new_prog[0] - old_prog[0], new_prog[1] - old_prog[1], new_prog[2] - old_prog[2])

    def update_max_op_level(self):
        opponent_level = self.reader.get_opponent_level()
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2

    def update_seen_pokemons(self):
        initial_seen_pokemon = 3
        self.seen_pokemons = sum(self.reader.read_seen_pokemons()) - initial_seen_pokemon

    def update_max_level(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.reader.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        # always keeping the max, lvl can't decrease
        self.max_level = max(self.max_level, scaled)
        return self.max_level

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
                    self.save_screenshot('healing')
                self.total_healing += heal_amount
            else:
                self.died_count += 1
        self.last_party_size = self.reader.read_party_size_address()
        self.last_health = self.reader.read_hp_fraction()

    def update_max_event(self):
        cur_rew = self.get_all_events_flags()
        self.max_event = max(cur_rew, self.max_event)

    def reset(self):
        self.max_event = 0
        self.max_level = 0
        self.total_healing = 0
        self.max_opponent_level = 0
        self.died_count = 0
        self.seen_pokemons = 0
        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem()
        self.last_game_state_rewards = self.get_game_state_rewards()
        self.total_reward = 0
