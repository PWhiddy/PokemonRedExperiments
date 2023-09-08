import pandas as pd
from pathlib import Path
import matplotlib
import seaborn
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
from multiprocessing import Pool
import io
import json
import math
from tqdm import tqdm
import mediapy as media
import numpy as np


def make_all_coords_arrays(filtered_dfs):
    return np.array([tdf[['x', 'y', 'map']].to_numpy().astype(np.uint8) for tdf in filtered_dfs]).transpose(1,0,2)

def get_sprite_by_coords(img, x, y):
    sy = 34+17*y
    sx = 9 +17*x
    alpha_v = np.array([255, 127,  39, 255], dtype=np.uint8)
    sprite = img[sy:sy+16, sx:sx+16]
    return np.where((sprite == alpha_v).all(axis=2).reshape(16,16,1), np.array([[[0,0,0,0]]]), sprite).astype(np.uint8)

def game_coord_to_global_coord(
    x, y, map_idx):
    
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
    coord = offset + np.array([x,y])
    #coord[1] = base_y - coord[1]
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

def compute_flow(all_coords, inter_steps=1, add_start=True):
    debug = False
    errors = []
    sprites_rendered = 0
    all_flows = {}
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
            #over = np.zeros_like(bg, dtype=np.uint8)
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

                p_coord = game_coord_to_global_coord(
                    cx, -cy, state[run]['map']
                )
                prev_p_coord = game_coord_to_global_coord(
                    px, -py, prev[2]
                )
                diff = p_coord - prev_p_coord
                #interp_coord = prev_p_coord + (fract*(diff.astype(np.float32))).astype(np.int32)
                if np.linalg.norm(diff) > 2:
                    continue
                coords_tup = tuple(prev_p_coord.tolist())
                if coords_tup in all_flows.keys():
                    all_flows[coords_tup] += diff
                else:
                    all_flows[coords_tup] = diff
                #error = add_sprite(
                #    over, walks[state[run]['dir']],
                #    interp_coord
                #)
                error = None
                if error is not None:
                    errors.append(error)
                else:
                    sprites_rendered += 1
            pbar.set_description(f"draws: {sprites_rendered}")
    
    return all_flows

def render_arrows(fname, all_flows, arrow_sprite):
    print("Rendering arrows")
    min_x = min([k[0] for k in all_flows.keys()])
    max_x = max([k[0] for k in all_flows.keys()])
    min_y = min([k[1] for k in all_flows.keys()])
    max_y = max([k[1] for k in all_flows.keys()])
    grid_dims = (max_x - min_x, max_y - min_y)
    cell_dim = arrow_sprite.size[0] # use x only, assuming square
     
    #colmap = matplotlib.cm.get_cmap('husl')
    colmap = seaborn.husl_palette(h=0.1, s=0.95, l=0.75, as_cmap=True)
    
    full_img = np.zeros( ((grid_dims[0]+1) * cell_dim, (grid_dims[1]+1) * cell_dim, 4 ), dtype=np.uint8)
    
    print("computing curl")
    dense_flow = np.zeros( ((grid_dims[0]+1), (grid_dims[1]+1), 2), dtype=np.float32)
    for coord, total_move in tqdm(all_flows.items()):
        dense_flow[coord[0], coord[1]] = total_move
    dFy_dx, dFy_dy = np.gradient(dense_flow[:, :, 1])
    dFx_dx, dFx_dy = np.gradient(dense_flow[:, :, 0])

    # Compute curl
    curl_F = dFy_dx - dFx_dy
    
    print(f"total curl: {curl_F.sum()}")
    
    with open('map_flow_run1/flow_dense.npy', 'wb') as f:
        np.save(f, dense_flow)

        # find interior flows and remove them
    coords_to_remove = set()
    for coord, total_move in tqdm(all_flows.items()):
        if (
            (coord[0]-1, coord[1]) in all_flows.keys() and 
            (coord[0]+1, coord[1]) in all_flows.keys() and 
            (coord[0], coord[1]-1) in all_flows.keys() and 
            (coord[0], coord[1]+1) in all_flows.keys()
        ):
            coords_to_remove.add(coord)
    
    for cord in coords_to_remove:
        del all_flows[cord]
    
    for coord, total_move in tqdm(all_flows.items()):
        angle = math.atan2(-total_move[0], total_move[1])
        #mag = math.sqrt(coord[0]**2 + coord[1]**2)
        rotated_arrow = arrow_sprite.rotate(180*angle/math.pi, resample=Image.Resampling.BICUBIC)
        nx = coord[0] - min_x
        ny = coord[1] - min_y
        #color = hsv2rgb(np.array([0.5*angle/math.pi+0.5, 1.0, 1.0]))
        color = colmap(0.5*angle/math.pi+0.5)
        full_img[
            nx * cell_dim : (nx + 1) * cell_dim, 
            ny * cell_dim : (ny + 1) * cell_dim
        ] = np.array(rotated_arrow) * np.array([color[0], color[1], color[2], 1.0])
    print("Writing file")
    final_img = Image.fromarray(full_img)
    final_img.save(f"{fname}.png")
    
    '''
    print("generating coords")
    fig, ax = plt.subplots(figsize=grid_dims)
    u = np.array([v[0] for v in all_flows.values()])
    v = np.array([v[1] for v in all_flows.values()])
    u, v = np.tanh(u), np.tanh(v)
    cols = []
    for i in range(len(u)):
        mag = (u[i]**2 + v[i]**2)**0.25 + 0.0001
        #u[i] /= mag
        #v[i] /= mag
        cols.append(mag)
    print("rendering")
    ax.quiver(
        [k[0] for k in all_flows.keys()], 
        [k[1] for k in all_flows.keys()], 
        u, v,
        cols
    )
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    #ax.axis([-0.3, 2.3, -0.3, 2.3])
    ax.set_aspect('equal')
    #plt.title('Matplotlib Example')
    print("saving")
    plt.savefig(f"{fname}.png")
    '''
    
def compute_flow_wrap(dat):
    print(f'processing chunk with shape {dat.shape}')
    return compute_flow(
        dat,
        inter_steps=1
    )

if __name__ == '__main__':
    
    run_dir = Path('baselines/session_4da05e87') # Path('baselines/session_ebdfe818')
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
    arrow_size = 16 #32
    arrow_img = Image.open('poke_map/transparent_arrow.png').resize((arrow_size, arrow_size))
    #alpha_val = get_sprite_by_coords(chars_img, 1, 0)[0,0]
    #walks = [get_sprite_by_coords(chars_img, x, 0) for x in [1, 4, 6, 8]]
        
    procs = 8
    with Pool(procs) as p:
        run_steps = 16385
        base_data = rearrange(base_coords, '(v s) r c -> s (v r) c', v=base_coords.shape[0]//run_steps)
        #base_data = base_data[:, ::110, :] # (16385, 26840, 3)
        print(f'base_data shape: {base_data.shape}')
        runs = base_data.shape[0] #base_data.shape[1]
        chunk_size = runs // procs
        batches_all_flows = p.map(
            compute_flow_wrap, 
            [base_data[chunk_size*i:chunk_size*(i+1)+5] for i in range(procs)])
        
        print(f"merging {len(batches_all_flows)} batches")
        merged_flows = {}
        for batch in tqdm(batches_all_flows):
            for cell, flow in batch.items():
                if cell in merged_flows.keys():
                    merged_flows[cell] += flow
                else:
                    merged_flows[cell] = flow
        
        render_arrows("map_flow_run1/full_combined_test_inter", merged_flows, arrow_img)