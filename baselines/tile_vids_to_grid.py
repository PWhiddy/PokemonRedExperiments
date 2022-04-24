import sys
import stat
import subprocess
from pathlib import Path

# inspired by https://www.jimby.name/techbits/recent/xstack/
# and https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos%20using%20xstack

def run_ffmpeg_grid(out_path, files, screen_res_str, full_res_string, gx, gy, short_test=False):
    cmd = ['ffmpeg']
    for file in files:
        cmd.append("-i")
        cmd.append(str(file.resolve()))
    cmd.append("-filter_complex")
    fltr = ""
    fltr = '"'
    for idx, file in enumerate(files):
        fltr += f"[{idx}:v] setpts=PTS-STARTPTS, scale={screen_res_str} [a{idx}]; "

    for idx in range(len(files)):
        fltr += f"[a{idx}]"
    fltr += f"xstack=inputs={len(files)}:layout="
    layout = []
    for y in range(gy):
        if y == 0:
            cur_y = "0"
        else:
            cur_y = "+".join([f"h{cy}" for cy in range(y)])
        for x in range(gx):
            if x == 0:
                cur_x = "0"
            else:
                cur_x = "+".join([f"w{cx}" for cx in range(x)])
            layout.append(f"{cur_x}_{cur_y}")
    fltr += "|".join(layout)
    fltr += "[out]"
    fltr += '" '
    cmd.append(fltr)
    cmd.append("-map")
    cmd.append("[out]")
    #cmd.append("-c:v")
    #cmd.append("libx264")
    if short_test:
        cmd.append("-t")
        cmd.append("10")
    cmd.append(str(out_path.resolve()))
    
    #-f matroska -
    
    #proc = subprocess.Popen(cmd)
    '''
    while True:
        line = proc.stdout.readline()
        if not line: break
        print(line)
    '''
    
    return ' '.join(cmd)
              
def make_script(path):
    sess_dir = path
    print(f"generating grid script for {sess_dir.name}")
    rollout_dir = sess_dir / "rollouts"
    all_files = list(rollout_dir.glob("full_reset_1*.mp4"))
    return run_ffmpeg_grid(
        (sess_dir / sess_dir.name).with_suffix('.mp4'), all_files, 
        "160x144", "1280x720", 8, 5, short_test=False)

def make_outer_script(out_file, paths):
    return run_ffmpeg_grid(
        out_file, paths, 
        "1280x720", "10240x5760", 8, 8, short_test=False)

def write_file(out_file, script):
    with open(out_file, "w") as f:
        print(f"writing to {f}")
        print(script, file=f)
    out_file.chmod(out_file.stat().st_mode | stat.S_IEXEC)

if __name__ == "__main__":
    inner_mosaic = False
    if inner_mosaic:
        outer_dir = Path(sys.argv[1])
        all_sessions = list(outer_dir.glob("session_*"))
        scripts = [make_script(sess) for sess in all_sessions]
        for script, sess in zip(scripts, all_sessions):
            out_file = Path(outer_dir / Path("parallel_scripts") / sess.with_suffix('.sh').name)
            write_file(out_file, script)
    else:
        base = Path('grid_renders')
        all_input_vids = list(base.glob("session_*/session_*.mp4"))
        print(len(all_input_vids))
        output_dir = base / "outer_mosaic"
        script = make_outer_script(output_dir / "big_boi.mp4", all_input_vids)
        write_file(output_dir / "big_boi.sh", script)