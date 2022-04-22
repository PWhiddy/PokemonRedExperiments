import sys
import subprocess
from pathlib import Path

# inspired by https://www.jimby.name/techbits/recent/xstack/
# and https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos%20using%20xstack

def run_ffmpeg_grid(out_path, files, screen_res_str, full_res_string, gx, gy, short_test=False):
    cmd = ['ffmpeg']
    for file in files:
        cmd.append("-i")
        cmd.append(str(file))
    cmd.append("-filter_complex")
    fltr = "" #fltr = '"'
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
    #fltr += '" '
    cmd.append(fltr)
    cmd.append("-map")
    cmd.append("[out]")
    #cmd.append("-c:v")
    #cmd.append("libx264")
    if short_test:
        cmd.append("-t")
        cmd.append("10")
    cmd.append(str(out_path))
    
    #-f matroska -
    
    proc = subprocess.Popen(cmd)
    while True:
        line = proc.stdout.readline()
        if not line: break
        print(line)
    
    #print(' '.join(cmd))

if __name__ == "__main__":
    sess_dir = Path(sys.argv[1])
    rollout_dir = sess_dir / "rollouts"
    all_files = list(rollout_dir.glob("full_reset_1*.mp4"))
    run_ffmpeg_grid((sess_dir / sess_dir.name).with_suffix('.mp4'), all_files, "160x144", "1280x720", 8, 5, short_test=False)