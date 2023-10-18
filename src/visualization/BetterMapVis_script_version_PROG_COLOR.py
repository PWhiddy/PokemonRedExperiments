import pandas as pd
from pathlib import Path
from colorcet.plotting import swatch, swatches
import holoviews as hv
hv.extension('matplotlib')
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
import requests
from multiprocessing import Pool
import io
import json
from tqdm import tqdm
import mediapy as media
import numpy as np


def make_all_coords_arrays(filtered_dfs):
    return np.array([tdf[['x', 'y', 'map']].to_numpy().astype(np.uint8) for tdf in filtered_dfs]).transpose(1,0,2)

def load_tex(name):
    resp = requests.get(sprites[name])
    return np.array(Image.open(io.BytesIO(resp.content)))

def get_sprite_by_coords(img, x, y):
    sy = 34+17*y
    sx = 9 +17*x
    alpha_v = np.array([255, 127,  39, 255], dtype=np.uint8)
    sprite = img[sy:sy+16, sx:sx+16]
    return np.where((sprite == alpha_v).all(axis=2).reshape(16,16,1), np.array([[[0,0,0,0]]]), sprite).astype(np.uint8)

def game_coord_to_pixel_coord(
    x, y, map_idx, base_y):
    
    global_offset = np.array([1056-16*12, 331]) #np.array([790, -29])
    map_offsets = {
        # https://bulbapedia.bulbagarden.net/wiki/List_of_locations_by_index_number_(Generation_I)
        0: np.array([0,0]), # pallet town
        1: np.array([-10, 72]), # viridian
        2: np.array([-10, 180]), # pewter
        12: np.array([0, 36]), # route 1
        13: np.array([0, 144]), # route 2
        14: np.array([30, 172]), # Route 3
        15: np.array([80, 190]), #Route 4
        33: np.array([-50, 64]), # route 22
        37: np.array([-9, 2]), # red house first
        38: np.array([-9, 25-32]), # red house second
        39: np.array([9+12, 2]), # blues house
        40: np.array([25-4, -6]), # oaks lab
        41: np.array([30, 47]), # Pokémon Center (Viridian City)
        42: np.array([30, 55]), # Poké Mart (Viridian City)
        43: np.array([30, 72]), # School (Viridian City)
        44: np.array([30, 64]), # House 1 (Viridian City)
        47: np.array([21,136]), # Gate (Viridian City/Pewter City) (Route 2)
        49: np.array([21,108]), # Gate (Route 2)
        50: np.array([21,108]), # Gate (Route 2/Viridian Forest) (Route 2)
        51: np.array([-35, 137]), # viridian forest
        52: np.array([-10, 189]), # Pewter Museum (floor 1)
        53: np.array([-10, 198]), # Pewter Museum (floor 2)
        54: np.array([-21, 169]), #Pokémon Gym (Pewter City)
        55: np.array([-19, 177]), #House with disobedient Nidoran♂ (Pewter City)
        56: np.array([-30, 163]), #Poké Mart (Pewter City)
        57: np.array([-19, 177]), #House with two Trainers (Pewter City)
        58: np.array([-25, 154]), # Pokémon Center (Pewter City)
        59: np.array([83, 227]), # Mt. Moon (Route 3 entrance)
        60: np.array([123, 227]), # Mt. Moon
        61: np.array([152, 227]), # Mt. Moon
        68: np.array([65, 190]), # Pokémon Center (Route 4)
        193: None # Badges check gate (Route 22)
    }
    if map_idx in map_offsets.keys():
        offset = map_offsets[map_idx]
    else:
        offset = np.array([0,0])
        x, y = 0, 0
    coord = global_offset + 16*(offset + np.array([x,y]))
    coord[1] = base_y - coord[1]
    return coord

def add_sprite(overlay_map, sprite, coord):
    raw_base = (overlay_map[coord[1]:coord[1]+16, coord[0]:coord[0]+16, :])
    intermediate = raw_base
    mask = sprite[:, :, 3] != 0
    if (mask.shape != intermediate[:,:,0].shape):
        #print(f'requested coords: {coord[1]}-{coord[1]+16}, {coord[0]}-{coord[0]+16}')
        #print(f'overlay_map.shape {overlay_map.shape}')
        #print(f'mask.shape {mask.shape} intermediate[:,:,0].shape {intermediate[:,:,0].shape}')
        #print(f'x {x} y {y} map {map_idx}')
        return {'coords': coord}
    else:
        intermediate[mask] = sprite[mask]
    overlay_map[coord[1]:coord[1]+16, coord[0]:coord[0]+16, :] = intermediate
    
def blend_overlay(background, over):
    al = over[...,3].reshape(over.shape[0], over.shape[1], 1)
    ba = (255-al)/255
    oa = al/255
    return (background[..., :3]*ba + over[..., :3]*oa).astype(np.uint8)

def split(img):
    return img

def render_video(fname, all_coords, walks, bg, inter_steps=4, add_start=True):
    debug = False
    errors = []
    sprites_rendered = 0
    turbo_map = get_cmap("cet_isoluminant_cgo_80_c38")._resample(8) #mpl.colormaps['turbo']._resample(8)
    with media.VideoWriter(
        f'{fname}.mov', split(bg).shape[:2], codec='prores_ks', 
        encoded_format='yuva444p', input_format='rgba', fps=60
    ) as wr:
        step_count = len(all_coords)
        state = [{'dir': 0, 'map': 40} for _ in all_coords[0]]
        pbar = tqdm(range(0, step_count))
        for idx in pbar:
            step = all_coords[idx]
            if idx > 0:
                prev_step = all_coords[idx-1]
            elif add_start:
                prev_step = np.tile(np.array([5, 3, 40]), (all_coords.shape[1], 1))
            else:
                prev_step = all_coords[idx]
            if debug:
                print('-- step --')
            for fract in np.arange(0,1,1/inter_steps):
                over = np.zeros_like(bg, dtype=np.uint8)
                for run in range(len(step)):
                    cur = step[run]
                    prev = prev_step[run]
                    # cast to regular int from uint8
                    cx, cy, px, py = map(int, [cur[0], cur[1], prev[0], prev[1]])
                    dx = cx - px
                    dy = cy - py
                    total_delta = abs(dx) + abs(dy)
                    if total_delta > 1:
                        state[run]['map'] = cur[2]
                    dx = min(max(dx, -1), 1)
                    dy = -1*min(max(dy, -1), 1)
                    if debug:
                        print(f'x: {cx} y: {cy} dx: {dx} dy: {dy}')
                    # only change direction if not moving between maps
                    if cur[2] == prev[2]:
                        if dx > 0:
                            state[run]['dir'] = 3
                        elif dx < 0:
                            state[run]['dir'] = 2
                        elif dy > 0:
                            state[run]['dir'] = 1
                        elif dy < 0:
                            state[run]['dir'] = 0

                    p_coord = game_coord_to_pixel_coord(
                        cx, -cy, state[run]['map'], over.shape[0]
                    )
                    prev_p_coord = game_coord_to_pixel_coord(
                        px, -py, prev[2], over.shape[0]
                    )
                    diff = p_coord - prev_p_coord
                    interp_coord = prev_p_coord + (fract*(diff.astype(np.float32))).astype(np.int32)
                    if np.linalg.norm(diff) > 16:
                        continue
                    agent_version_float = (run // 44) / 610
                    error = add_sprite(
                        over, np.array(turbo_map(agent_version_float)) * walks[state[run]['dir']],
                        interp_coord
                    )
                    if error is not None:
                        errors.append(error)
                    else:
                        sprites_rendered += 1
                wr.add_image(split(over[:,:,:]))
                perc = len(errors) / (sprites_rendered + len(errors))
                pbar.set_description(f"draws: {sprites_rendered} errors: {len(errors)}, {perc:.2%}")
    return errors

def test_render(name, dat, walks, bg):
    print(f'processing chunk with shape {dat.shape}')
    return render_video(
        name,
        dat,
        walks,
        bg, inter_steps=8
    )

if __name__ == '__main__':
    
    run_dir = Path('baselines/session_4da05e87_main_full') # Path('baselines/session_ebdfe818')
# original session_e41c9eff, main session_4da05e87, extra session_e1b6d2dc
    
    coords_save_pth = Path('base_coords.npz')
    
    if coords_save_pth.is_file():
        print(f'{coords_save_pth} found, loading from file')
        base_coords = np.load(coords_save_pth)['arr_0']
    else:
        print(f'{coords_save_pth} not found, building...')
        dfs = []
        for run in tqdm(run_dir.glob('*.gz')):
            tdf = pd.read_csv(run, compression='gzip')
            dfs.append(tdf[tdf['map'] != 'map'])

        base_coords = make_all_coords_arrays(dfs)
        print(f'saving {coords_save_pth}')
        np.savez_compressed(coords_save_pth, base_coords)
    
    print(f'initial data shape: {base_coords.shape}')

    main_map = np.array(Image.open('poke_map/pokemap_full_calibrated_CROPPED_1.png'))
    chars_img = np.array(Image.open('poke_map/characters.png'))
    alpha_val = get_sprite_by_coords(chars_img, 1, 0)[0,0]
    walks = [get_sprite_by_coords(chars_img, x, 0) for x in [1, 4, 6, 8]]
        
    start_bg = main_map.copy()

    procs = 8#16
    with Pool(procs) as p:
        run_steps = 16385
        base_data = rearrange(base_coords, '(v s) r c -> s (v r) c', v=base_coords.shape[0]//run_steps)
        base_data = base_data[:1024]
        print(f'base_data shape: {base_data.shape}')
        runs = base_data.shape[0] #base_data.shape[1]
        chunk_size = runs // procs
        #render_errors = test_render(
        #    f'map_vis_final_state', base_data[(base_data.shape[0]-5):], walks, start_bg)
        #f'map_vis_initial_state', base_data[:], walks, start_bg
        all_render_errors = p.starmap(
            test_render, 
            [(f'map_vis_color/map_vis_initial_state{i}', base_data[chunk_size*i:chunk_size*(i+1)+5], walks, start_bg) for i in range(procs)])
    
