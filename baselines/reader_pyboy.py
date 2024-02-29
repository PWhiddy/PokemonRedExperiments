from memory_addresses import *


class ReaderPyBoy:

    def __init__(self, pyboy):
        self.pyboy = pyboy

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(MONEY_ADDRESS_1)) +
                100 * self.read_bcd(self.read_m(MONEY_ADDRESS_2)) +
                self.read_bcd(self.read_m(MONEY_ADDRESS_3)))

    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in HP_ADDRESSES])
        max_hp_sum = sum([self.read_hp(add) for add in MAX_HP_ADDRESSES])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)

    def get_badges(self):
        return self.bit_count(self.read_m(BADGE_COUNT_ADDRESS))

    def get_opponent_level(self):
        return max([self.read_m(a) for a in OPPONENT_LEVELS_ADDRESSES]) - 5

    def read_party(self):
        return [self.read_m(addr) for addr in PARTY_ADDRESSES]

    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in LEVELS_ADDRESSES]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level

    def read_party_size_address(self):
        return self.read_m(PARTY_SIZE_ADDRESS)

    def read_x_pos(self):
        return self.read_m(X_POS_ADDRESS)

    def read_y_pos(self):
        return self.read_m(Y_POS_ADDRESS)

    def read_map_n(self):
        return self.read_m(MAP_N_ADDRESS)

    def read_events(self):
        return [
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START_ADDRESS, EVENT_FLAGS_END_ADDRESS)
        ]

    def read_museum_tickets(self):
        museum_ticket = (MUSEUM_TICKET_ADDRESS, 0)
        return self.read_bit(museum_ticket[0], museum_ticket[1])

    def read_levels(self):
        return [self.read_m(a) for a in LEVELS_ADDRESSES]

    def read_seen_pokemons(self):
        return [self.bit_count(self.read_m(a)) for a in SEEN_POKEMONS_ADDRESSES]

    def get_map_location(self):
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
        if self.read_map_n() in map_locations.keys():
            return map_locations[self.read_map_n()]
        else:
            return "Unknown Location"
