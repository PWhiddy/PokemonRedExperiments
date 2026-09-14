"""
ram_map.py -- single source of truth for reading Pokemon Red game state from RAM.

All RAM addresses are gathered here so the rest of the curriculum system never
hard-codes a magic number. Addresses are for **Pokemon Red/Blue (GB)** and were
cross-checked against:

  - this repo's own ``baselines/memory_addresses.py`` and ``red_gym_env_v2.py``
  - https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
  - the pret/pokered decompilation (constants/event_constants.asm)

``GameState`` wraps a single PyBoy instance and exposes clean, named accessors.
It is intentionally read-only and side-effect free so it can be called many times
per step cheaply.

Event-flag note
---------------
The repo's ``events.json`` maps keys like ``"0xD74B-2"`` -> ``"Got Starter"``. The
bit index in that file is **MSB-first** (it indexes into ``f"{byte:08b}"``), so
index 0 is bit 7. ``GameState.event_flag(addr, idx)`` uses that same MSB-first
convention so milestone detectors can refer to events.json keys directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Core position / map                                                         #
# --------------------------------------------------------------------------- #
ADDR_MAP_N = 0xD35E          # current map id (matches map_data.json ids)
ADDR_X = 0xD362              # player X (local tile coord)
ADDR_Y = 0xD361              # player Y (local tile coord)
ADDR_PREV_MAP = 0xD35F       # previous map id (last map left)

# --------------------------------------------------------------------------- #
# Party                                                                        #
# --------------------------------------------------------------------------- #
ADDR_PARTY_COUNT = 0xD163
PARTY_STRUCT_SIZE = 0x2C      # 44 bytes per party-mon struct
PARTY_SPECIES_ADDRS = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
PARTY_HP_ADDRS = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]       # 2 bytes each
PARTY_MAXHP_ADDRS = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]    # 2 bytes each
PARTY_STATUS_ADDRS = [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]   # 1 byte each
# Each mon's 4 move ids start here (mon i at PARTY_MOVES_BASE + i*PARTY_STRUCT_SIZE).
PARTY_MOVES_BASE = 0xD173

# Field-move ids used to detect whether a party member can use an HM in the overworld.
MOVE_CUT = 0x0F
MOVE_SURF = 0x39
MOVE_STRENGTH = 0x46
MOVE_FLY = 0x13
MOVE_FLASH = 0x94

# --------------------------------------------------------------------------- #
# Badges / events                                                              #
# --------------------------------------------------------------------------- #
ADDR_BADGES = 0xD356         # bitfield, one bit per Kanto badge
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E     # exclusive-ish upper bound used across this repo
MUSEUM_TICKET = (0xD754, 0)  # (addr, lsb_bit) -- excluded from "progress" events

# --------------------------------------------------------------------------- #
# Battle                                                                        #
# --------------------------------------------------------------------------- #
ADDR_IN_BATTLE = 0xD057      # 0 = not in battle, !=0 = in a battle
ADDR_BATTLE_TYPE = 0xD05A    # 0 wild, 1 trainer, etc.
ADDR_ENEMY_SPECIES = 0xCFE5  # current enemy mon species (battle only)
ADDR_ENEMY_LEVEL = 0xCFF3    # current enemy mon level (battle only)
ADDR_ENEMY_HP = 0xCFE6       # current enemy mon HP, 2 bytes (battle only)
ADDR_ENEMY_MAXHP = 0xCFF4    # current enemy mon max HP, 2 bytes (battle only)
ADDR_ENEMY_PARTY_COUNT = 0xD89C

# --------------------------------------------------------------------------- #
# Resources                                                                    #
# --------------------------------------------------------------------------- #
MONEY_ADDRS = (0xD347, 0xD348, 0xD349)  # 3-byte big-endian BCD
ADDR_BAG_COUNT = 0xD31D
ADDR_BAG_ITEMS = 0xD31E      # (item_id, qty) pairs, 0xFF terminated, up to 20 items
BAG_CAPACITY = 20

# Pokedex bit arrays (each is a run of bytes, 1 bit per species)
POKEDEX_OWNED_START = 0xD2F7
POKEDEX_OWNED_END = 0xD309
POKEDEX_SEEN_START = 0xD30A
POKEDEX_SEEN_END = 0xD31C

# Item ids of interest (Gen 1 internal item ids)
ITEM_POKEBALL_IDS = {0x01, 0x02, 0x03, 0x04}  # MASTER, ULTRA, GREAT, POKE BALL
ITEM_OAKS_PARCEL = 0x46
ITEM_BICYCLE = 0x06
ITEM_SS_TICKET = 0x3F
ITEM_TOWN_MAP = 0x05
ITEM_SILPH_SCOPE = 0x48
ITEM_POKE_FLUTE = 0x49
# HM item ids (HM01..HM05 -> CUT, FLY, SURF, STRENGTH, FLASH)
HM_ITEM_IDS = {0xC4: "HM01_CUT", 0xC5: "HM02_FLY", 0xC6: "HM03_SURF",
               0xC7: "HM04_STRENGTH", 0xC8: "HM05_FLASH"}
# TM item ids span 0xC9..0xF9 (TM01..TM50)
TM_ITEM_ID_RANGE = (0xC9, 0xF9)

# Number of badge bits / a convenience badge-name ordering (Red badge bit order)
BADGE_NAMES = [
    "Boulder", "Cascade", "Thunder", "Rainbow",
    "Soul", "Marsh", "Volcano", "Earth",
]


@dataclass
class PartyMon:
    """A light view of one party slot."""
    species: int
    level: int
    hp: int
    max_hp: int
    status: int

    @property
    def hp_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp > 0 else 0.0

    @property
    def fainted(self) -> bool:
        return self.max_hp > 0 and self.hp == 0


class GameState:
    """Read-only structured view over a PyBoy instance's RAM.

    Parameters
    ----------
    pyboy:
        A live ``pyboy.PyBoy`` instance. Only ``pyboy.memory[addr]`` is used.
    """

    def __init__(self, pyboy):
        self.pyboy = pyboy

    # ----- low level ------------------------------------------------------- #
    def read(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    def read_u16(self, addr: int) -> int:
        """Big-endian 2-byte read (Game Boy stores these hi-byte first here)."""
        return (self.read(addr) << 8) + self.read(addr + 1)

    def read_bit_lsb(self, addr: int, bit: int) -> bool:
        """Read ``bit`` (0 = LSB) of the byte at ``addr``."""
        return bool((self.read(addr) >> bit) & 1)

    def event_flag(self, addr: int, msb_index: int) -> bool:
        """Read an event flag using the events.json MSB-first index convention.

        ``msb_index`` 0 selects bit 7, index 7 selects bit 0. This matches the
        keys in ``events.json`` (e.g. ``"0xD74B-2"``).
        """
        return self.read_bit_lsb(addr, 7 - msb_index)

    # ----- position / map -------------------------------------------------- #
    def map_id(self) -> int:
        return self.read(ADDR_MAP_N)

    def prev_map_id(self) -> int:
        return self.read(ADDR_PREV_MAP)

    def position(self) -> Tuple[int, int, int]:
        """Return ``(x, y, map_id)`` matching ``RedGymEnv.get_game_coords``."""
        return self.read(ADDR_X), self.read(ADDR_Y), self.read(ADDR_MAP_N)

    # ----- party ----------------------------------------------------------- #
    def party_count(self) -> int:
        return self.read(ADDR_PARTY_COUNT)

    def party_species(self) -> List[int]:
        n = self.party_count()
        return [self.read(PARTY_SPECIES_ADDRS[i]) for i in range(min(n, 6))]

    def party_levels(self) -> List[int]:
        n = self.party_count()
        return [self.read(PARTY_LEVEL_ADDRS[i]) for i in range(min(n, 6))]

    def party_mons(self) -> List[PartyMon]:
        n = min(self.party_count(), 6)
        mons = []
        for i in range(n):
            mons.append(PartyMon(
                species=self.read(PARTY_SPECIES_ADDRS[i]),
                level=self.read(PARTY_LEVEL_ADDRS[i]),
                hp=self.read_u16(PARTY_HP_ADDRS[i]),
                max_hp=self.read_u16(PARTY_MAXHP_ADDRS[i]),
                status=self.read(PARTY_STATUS_ADDRS[i]),
            ))
        return mons

    def party_levels_sum(self) -> int:
        return sum(self.party_levels())

    def mon_moves(self, slot: int) -> List[int]:
        """The 4 move ids of party member ``slot`` (0-based)."""
        base = PARTY_MOVES_BASE + slot * PARTY_STRUCT_SIZE
        return [self.read(base + j) for j in range(4)]

    def mon_index_with_move(self, move_id: int) -> int:
        """Return the first party slot whose mon knows ``move_id``, or -1."""
        for i in range(min(self.party_count(), 6)):
            if move_id in self.mon_moves(i):
                return i
        return -1

    def party_knows_move(self, move_id: int) -> bool:
        return self.mon_index_with_move(move_id) >= 0

    def party_hp_fractions(self) -> List[float]:
        return [m.hp_fraction for m in self.party_mons()]

    def total_hp_fraction(self) -> float:
        """Whole-party HP fraction (matches RedGymEnv.read_hp_fraction)."""
        hp = 0
        mx = 0
        for i in range(6):
            hp += self.read_u16(PARTY_HP_ADDRS[i])
            mx += self.read_u16(PARTY_MAXHP_ADDRS[i])
        return hp / max(mx, 1)

    def all_fainted(self) -> bool:
        """True when the party exists but every mon is fainted (blackout)."""
        mons = self.party_mons()
        return len(mons) > 0 and all(m.fainted for m in mons)

    # ----- badges / events ------------------------------------------------- #
    def badges_byte(self) -> int:
        return self.read(ADDR_BADGES)

    def badge_count(self) -> int:
        return bin(self.badges_byte()).count("1")

    def badge_bits(self) -> List[int]:
        return [int(b) for b in f"{self.badges_byte():08b}"]

    def has_badge(self, index: int) -> bool:
        """Badge ``index`` 0..7 in canonical badge order (Boulder..Earth)."""
        return self.read_bit_lsb(ADDR_BADGES, index)

    def event_flags_sum(self) -> int:
        """Total number of set event-flag bits in the tracked range."""
        return sum(bin(self.read(a)).count("1")
                   for a in range(EVENT_FLAGS_START, EVENT_FLAGS_END))

    # ----- battle ---------------------------------------------------------- #
    def in_battle(self) -> bool:
        return self.read(ADDR_IN_BATTLE) != 0

    def battle_type(self) -> int:
        return self.read(ADDR_BATTLE_TYPE)

    def enemy_species(self) -> int:
        return self.read(ADDR_ENEMY_SPECIES) if self.in_battle() else 0

    def enemy_level(self) -> int:
        return self.read(ADDR_ENEMY_LEVEL) if self.in_battle() else 0

    def enemy_hp_fraction(self) -> float:
        if not self.in_battle():
            return 0.0
        mx = self.read_u16(ADDR_ENEMY_MAXHP)
        return self.read_u16(ADDR_ENEMY_HP) / mx if mx > 0 else 0.0

    # ----- resources ------------------------------------------------------- #
    def money(self) -> int:
        """3-byte BCD money value."""
        b1, b2, b3 = (self.read(a) for a in MONEY_ADDRS)

        def bcd(b):
            return (b >> 4) * 10 + (b & 0x0F)

        return bcd(b1) * 10000 + bcd(b2) * 100 + bcd(b3)

    def bag_items(self) -> List[Tuple[int, int]]:
        """Return list of ``(item_id, quantity)`` currently in the bag."""
        count = min(self.read(ADDR_BAG_COUNT), BAG_CAPACITY)
        items = []
        for i in range(count):
            item_id = self.read(ADDR_BAG_ITEMS + i * 2)
            if item_id == 0xFF:
                break
            qty = self.read(ADDR_BAG_ITEMS + i * 2 + 1)
            items.append((item_id, qty))
        return items

    def bag_item_ids(self) -> set:
        return {iid for iid, _ in self.bag_items()}

    def has_item(self, item_id: int) -> bool:
        return item_id in self.bag_item_ids()

    def pokeball_count(self) -> int:
        return sum(qty for iid, qty in self.bag_items() if iid in ITEM_POKEBALL_IDS)

    def hm_ids_owned(self) -> set:
        return {iid for iid in self.bag_item_ids() if iid in HM_ITEM_IDS}

    def tm_ids_owned(self) -> set:
        lo, hi = TM_ITEM_ID_RANGE
        return {iid for iid in self.bag_item_ids() if lo <= iid <= hi}

    def has_cut(self) -> bool:
        return 0xC4 in self.bag_item_ids()  # HM01

    def can_use_cut(self) -> bool:
        """True when Cut is actually usable in the field: a party member knows Cut
        and the Cascade Badge (badge bit 1) -- required to use it -- is owned."""
        return self.has_badge(1) and self.party_knows_move(MOVE_CUT)

    def has_surf(self) -> bool:
        return 0xC6 in self.bag_item_ids()  # HM03

    def can_use_surf(self) -> bool:
        """True when Surf is usable in the field: a party member knows Surf and the
        Soul Badge (badge bit 4), required to use it, is owned."""
        return self.has_badge(4) and self.party_knows_move(MOVE_SURF)

    def has_strength(self) -> bool:
        return 0xC7 in self.bag_item_ids()  # HM04

    def can_use_strength(self) -> bool:
        """True when Strength is usable in the field: a party member knows Strength and
        the Rainbow Badge (badge bit 3), required to use it, is owned."""
        return self.has_badge(3) and self.party_knows_move(MOVE_STRENGTH)

    def has_poke_flute(self) -> bool:
        return ITEM_POKE_FLUTE in self.bag_item_ids()

    def pokedex_owned_count(self) -> int:
        return sum(bin(self.read(a)).count("1")
                   for a in range(POKEDEX_OWNED_START, POKEDEX_OWNED_END + 1))

    def pokedex_seen_count(self) -> int:
        return sum(bin(self.read(a)).count("1")
                   for a in range(POKEDEX_SEEN_START, POKEDEX_SEEN_END + 1))

    def has_pokedex(self) -> bool:
        # event 0xD74B-5 "Got Pokedex"
        return self.event_flag(0xD74B, 5)
