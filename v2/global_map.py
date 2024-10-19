# adapted from https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py

import os
import json

MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}

# Handle KeyErrors
def local_to_global(r: int, c: int, map_n: int):
    try:
        (
            map_x,
            map_y,
        ) = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
