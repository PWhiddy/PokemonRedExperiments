import argparse
import os
import re
import requests
import json
import subprocess
import glob
from pathlib import Path
import uuid
import datetime
import time
import sys  # Make sure to import the sys module
import gzip
import pandas as pd  # Add this import statement

DEFAULT_BASE_URL = "http://127.0.0.1:5000"
directory_path = 'downloaded_checkpoints'

# Add arguments for metrics and params
# upload and download included tensordata
# index.html session_xxxx instead of current folder to match the tensor setup
# for URL accept --refresh to set rate
# integrate hosting server in this location? then into script
#       Hosting and download to downloaded checkpoints script integration
# Print out a location html that show the furthest going instance.

# Newest Menu.py updates
# menu.py --info 
# added git update detection
# Checkpoint monitoring scans the newest session for newly created zips and reports them with a timestamp
# Allow --URL for a custom external server
# match new imports from source
# Review need for HTML view file due to new patch
	# it’s possible tensorboard does not allow “watching”
# index.html index information about json and other info
# Show highest value of the trained models in index.html


# Possible style upgrades
# consider If statement for Local and Downloaded sessions. Don’t display if nothing is found.

#Line 155 run_script = f"python3 {selected_run}" used to run scripts in the folder


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def monitor_zip_files(directory_to_watch):
    last_checkpoint_time = 0
    def handle_new_zip(file_path):
        nonlocal last_checkpoint_time 
        current_time = time.time()
        relative_path = os.path.relpath(file_path, start=directory_to_watch)
        #if current_time - last_checkpoint_time >= print_interval:
        interval = current_time - last_checkpoint_time
        minutes, seconds = divmod(interval, 60)
        print(f"Checkpoint {relative_path} created at {time.ctime()} interval : {int(minutes)} min {int(seconds)} sec")
        last_checkpoint_time = current_time  # Update the last checkpoint time

    # Watchdog event handler
    class NewFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".zip"):
                handle_new_zip(event.src_path)

    # Start the watchdog observer to monitor the directory
    observer = Observer()
    event_handler = NewFileHandler()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    observer.start()
    
    print(f"\nMonitoring {directory_to_watch} for new checkpoints.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if not os.path.exists(directory_path):
    os.makedirs(directory_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url')
    #parser.add_argument('--upload')
    #parser.add_argument('--restore')
    parser.add_argument('--view')
    parser.add_argument('--info', action='store_true', help='Display help information')

    args = parser.parse_args()

    if args.url:
        DEFAULT_BASE_URL = args.url
        # does not error out with a faulty IP
    #elif args.restore:
        #List a checkpount menu without a file specified
        #  no runs should be lists
        #selected_checkpoint = remote_actions()
    #elif args.upload:
        # if no file is offered you should list a menu
        #  #no runs should be listed
        #upload(selection, session_dict)
    elif args.view:
        simple_create_index('index.html')
        sys.exit()
    elif args.info:
        info()
        sys.exit()

    return parser.parse_args()

def show_menu(selected_checkpoint):
    update_optional = 0
    # Loop the menu indefinately 
    while True:
        # Check if your branch is up to date with 'origin/master'
        up_to_date = is_up_to_date()
        print("Your PokemonRedExperiments installation is up to date!" if up_to_date else "Your branch is not up to date with 'origin/master'.")

        session_dict, downloaded_checkpoints = list_all_sessions_and_pokes()
        if not session_dict:
            print("No checkpoints found.")
            return selected_checkpoint
        downloaded_checkpoint_count = len(session_dict)
        print(f"\nAvailable sessions sorted by their largest checkpoints:")
        for i, (session, largest_step) in enumerate(session_dict.items()):
            print(f"  {i + 1}. {session}/poke-{largest_step}_steps.zip")
        print("\n  Downloaded checkpoints:")
        for i, checkpoint in enumerate(downloaded_checkpoints, start=downloaded_checkpoint_count + 1):
            print(f"  {i}. {checkpoint}")
        print("\n  Default Runs:")
        matching_files = [file for file in os.listdir(os.getcwd()) if file.startswith("run_") and file.endswith(".py")]
        for i, file in enumerate(matching_files, start=downloaded_checkpoint_count + 1):
            print(f"  {i}. {file}")
        print("\n  95. Resume from remote")
        print("  96. Upload to remote")
        print("  97. Load a custom interactive checkpoint.")
        print("  98. Checkpoint creation live monitor.")
        print("  99. View progress using index.html")
        print("  999. View map progress map.html")
        #print("  999. View progress using Tailwind index.html")
        if update_optional == 1:
            print(f"  \n9999. Sync with code base update.")
            subprocess.run(["git", "merge", "origin/master"])
            print(f"Your branch has been synced with 'origin/master'.\n")
        menu_selection = input("Enter the number of the menu option: ")

        # Menu Logic
        if menu_selection.isdigit():
            selection = int(menu_selection)
            if 1 <= selection <= len(session_dict):
                selected_session = list(session_dict.keys())[selection - 1]
                selected_step = session_dict[selected_session]
                selected_checkpoint = f"{selected_session}/poke_{selected_step}_steps.zip"
                return selected_checkpoint
            elif downloaded_checkpoint_count + 1 <= selection <= downloaded_checkpoint_count + len(downloaded_checkpoints):
                selected_checkpoint = os.path.join('downloaded_checkpoints', downloaded_checkpoints[selection - downloaded_checkpoint_count - 1])
                return selected_checkpoint
            elif downloaded_checkpoint_count + len(downloaded_checkpoints) + 1 <= selection <= downloaded_checkpoint_count + len(downloaded_checkpoints) + len(matching_files):
                selected_run = matching_files[selection - downloaded_checkpoint_count - len(downloaded_checkpoints) - 1]
                run_script = f"python3 {selected_run}"
                subprocess.run(run_script, shell=True)
            elif menu_selection == '95':
                selected_checkpoint = remote_actions()
                if selected_checkpoint:
                    return selected_checkpoint     
            elif menu_selection == '96':
                selection = int(input("Enter your selection for remote upload: "))
                upload(selection, session_dict)
            elif menu_selection == '97':
                #custom restore
                return
            elif menu_selection == '98':
                all_folders = os.listdir()
                session_folders = [folder for folder in all_folders if re.match(r'session_[0-9a-fA-F]{8}', folder)]
                def get_creation_time(folder):
                    return os.path.getctime(folder)
                session_folders.sort(key=get_creation_time, reverse=True)
                if session_folders:
                    newest_session = session_folders[0]
                    monitor_zip_files(newest_session)
            elif menu_selection == '99':
                #create_index('index.html')
                simple_create_index('index.html')
            elif menu_selection == '999':
                create_map('map.html')
            #elif menu_selection == '999':
            #    tailwind_create_index('index.html')
            elif menu_selection == '9999':
                subprocess.run(["git", "pull", "origin/master"])
                print("Your branch has been synced with 'origin/master'.")
            else:
                print("Invalid selection.")
        else:
            print("Invalid input. Please enter a valid number.")

def list_all_sessions_and_pokes():
    all_folders = os.listdir()
    session_folders = [folder for folder in all_folders if re.match(r'session_[0-9a-fA-F]{8}', folder)]
    session_dict = {}
    downloaded_checkpoints = []
    for session_folder in session_folders:
        poke_files = glob.glob(f"{session_folder}/poke_*_steps.zip")
        if poke_files:
            largest_poke_file = max(poke_files, key=lambda x: int(re.search(r'poke_(\d+)_steps', x).group(1)))
            largest_step = int(re.search(r'poke_(\d+)_steps', largest_poke_file).group(1))
            session_dict[session_folder] = largest_step

    downloaded_checkpoints = [file for file in os.listdir('downloaded_checkpoints') if file.endswith('.zip')]
    sorted_session_dict = {k: v for k, v in sorted(session_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_session_dict, downloaded_checkpoints

def remote_actions():
    BASE_URL = DEFAULT_BASE_URL
    response = requests.get(f"{BASE_URL}/uploads/metadata.txt")
    if response.status_code != 200:
        print("Failed to fetch metadata from the server.")
        return None
    server_metadata = response.text.strip()
    if not server_metadata:
        print("No checkpoint metadata found. Is this an empty server?")
        return None
    try:
        server_metadata = json.loads(server_metadata)
    except json.decoder.JSONDecodeError as e:
        print("Error decoding JSON:", str(e))
        return None
    print(f"\nAvailable remote checkpoints:")
    for i, entry in enumerate(server_metadata):
        print(f"{i + 1}. Filename: {entry['filename']}, Steps: {entry['steps']}")
    server_selection = input("Enter the number of the checkpoint you want to download: ")
    try:
        server_selection = int(server_selection)
        if 1 <= server_selection <= len(server_metadata):
            selected_server_entry = server_metadata[server_selection - 1]
            filename = selected_server_entry['filename']
            download_response = requests.get(f"{BASE_URL}/uploads/{filename}")
            if download_response.status_code == 200:
                with open(f"downloaded_checkpoints/{filename}", 'wb') as f:
                    f.write(download_response.content)
                print(f"Downloaded checkpoint: {filename}")
            else:
                print(f"Failed to download the selected checkpoint: {filename}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    return None

def restore(url, download_selection):
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split("/")[-1]
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded checkpoint: {filename}")
        return filename
    else:
        print("Failed to download checkpoint.")
        return None

def upload(selection, session_dict):
    try:
        selected_session = list(session_dict.keys())[selection - 1]
        selected_step = session_dict[selected_session]
        file_path = f"{selected_session}/poke_{selected_step}_steps.zip"
        upload_command = f"curl -X POST -F file=@{file_path} http://127.0.0.1:5000/upload"
        subprocess.run(upload_command, shell=True)
    except (ValueError, IndexError):
        print("Invalid selection")

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def main(selected_checkpoint):
    from red_gym_env import RedGymEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from tensorboard_callback import TensorboardCallback

    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')
    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
        'explore_weight': 3
    }
    print(env_config)
    num_cpu = 16  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')    
    callbacks = [checkpoint_callback, TensorboardCallback()]
    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())
    learn_steps = 40

    print('\nLoading checkpoint', selected_checkpoint, ' ... \n')
    model = PPO.load(selected_checkpoint, env=env)
    model.n_steps = ep_length
    model.n_envs = num_cpu
    model.rollout_buffer.buffer_size = ep_length
    model.rollout_buffer.n_envs = num_cpu
    model.rollout_buffer.reset()
    for i in range(learn_steps):
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks))
    if use_wandb_logging:
        run.finish()

def info():
    help_message = """
===============================================================================
                        Pokemon Red Experiments - Help Menu
===============================================================================

Welcome to the Pokémon Plays AI script. This menu provides you with helpful
information about available options and actions. Use the following options:

--url: Specify a custom external server URL for remote interactions.
--view --refresh NUM: View progress using the default HTML interface.
--info: Display this help menu.

Additional Actions:
  - '96' to resume from a remote checkpoint
        allowing downloading from the Flask app.py in /www.
  - '97' to upload your checkpoint to the server
        enabling easy sharing of your checkpoints with yourself or others.
  - '98' to monitor checkpoint creation live, allowing you to see when
        a checkpoint is created so you can exit, minimizing loss.
  - '99' to view progress creates an index.html file in the running session 
        folder that updates to display your JPEGs for the current run.
  - '999' to view progress using Tailwind CSS-enhanced HTML. (Work in Progress)

To keep your script up to date with the code base, enter '9999'. If the script
detects you are out of date, it will append this option, but it is always available.

Enjoy using Pokémon Plays AI! https://github.com/PWhiddy/PokemonRedExperiments

===============================================================================
"""
    print(help_message)

def is_up_to_date():
    return subprocess.run(["git", "pull", "origin"]) or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == subprocess.run(["git", "rev-parse", "origin/master"], capture_output=True, text=True).stdout.strip()
import os
import pandas as pd
import gzip

def create_map(output_file='map.html'):
    # Find all session folders within the current working directory
    session_folders = [folder for folder in os.listdir() if folder.startswith('session_')]
    if not session_folders:
        print("No 'session_' folders found in the current working directory.")
        return

    # Create the initial HTML content
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Map Locations</title>
        <style>
            .run-container {{
                float: left;
                width: 30%;
                margin: 10px;
                padding: 10px;
            }}  
        </style>
    </head>
    <body>
        <h1>Map Locations and Counts</h1>
    """

    # Loop through session folders
    newest_session = max(session_folders, key=lambda folder: os.path.getctime(folder))
        # Create an empty dictionary to store unique locations and their counts for this session
    unique_location_counts = {}

    # Loop through files in the session directory
    image_dir = os.path.join(newest_session)
    for filename in os.listdir(image_dir):
        if filename.endswith('.csv.gz'):
            print(f"Found CSV file: {filename}")
            with gzip.open(os.path.join(newest_session, filename), 'rt') as file:
                df = pd.read_csv(file)
                if 'map_location' in df.columns:
                    total_lines = len(df)
                    unique_locations = df['map_location'].unique()
                    for location in unique_locations:
                        count = len(df[df['map_location'] == location])
                        percentage = (count / total_lines) * 100
                        unique_location_counts[location] = (count, percentage)
                else:
                    print(f"'map_location' column not found in file: {filename}")

            # Add the unique location counts to the HTML content for this session
            html_content += f"""
            <div class="run-container">
                <h2>{newest_session}</h2>
                <table>
            """
            
            for location, (count, percentage) in unique_location_counts.items():
                html_content += f"""
                <tr>
                <td><strong>Loc:</strong> {location}</td>
                <td><strong>Count:</strong> {count}</td>
                <td><strong>Percentage:</strong> {percentage:.2f}%</td>
                </tr>
                </div>
                """

            html_content += "</table></div>"

    # Complete the HTML content and save it to the file
    html_content += """</body></html>"""
    session_html_file = os.path.join(newest_session, output_file)
    with open(session_html_file, 'w') as file:
        file.write(html_content)

    print(f"You can now open '{newest_session}/{output_file}' in the session directory to view your results.")


def simple_create_index(output_file='index.html', max_items=20):
    # Find all session folders within the current working directory
    pair_count = 0

    session_folders = [folder for folder in os.listdir() if folder.startswith('session_')]
    if not session_folders:
        print("No 'session_' folders found in the current working directory.")
        return

    # Sort the session folders by their names (timestamps) and get the newest one
    newest_session = max(session_folders, key=lambda folder: os.path.getctime(folder))
    image_names = []
    json_names = {}
    zip_names = []
    imageid_badges = {}  # Initialize a dictionary to store image_id and highest badge
    imageid_values = {}  # Initialize a dictionary to store image_id and values

    # Get a list of file names in the newest session folder based on patterns
    image_dir = os.path.join(newest_session)
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpeg'):
            image_names.append(filename)
        elif filename.startswith('poke_') and filename.endswith('_steps.zip'):
            zip_names.append(filename)
        elif filename.startswith('all_runs_') and filename.endswith('.json'):
            highest_badge_value = 0  # Initialize to 0 initially
            image_id = filename.split('_')[2].split('.')[0]
            json_names[image_id] = filename
            # print(newest_session, "/", json_names[image_id])
            filepath = os.path.join(newest_session, json_names[image_id])
            # Open and parse the JSON data from the file
            with open(filepath, 'r') as json_file:
                data = json.load(json_file)
                for entry in data:
                    badge_value = entry.get('badge', 0)  # Default to 0 if 'badge' key doesn't exist
                    if badge_value > highest_badge_value:
                        highest_badge_value = badge_value
                        imageid_badges[image_id] = badge_value
                        print("new high value :", badge_value)
                    # Extract and store all the values from the JSON for this image
                    image_values = {
                        'eve': round(entry.get('event', 0), 2),
                        'lev': round(entry.get('level', 0), 2),
                        'hea': round(entry.get('heal', 0), 2),
                        'op_': round(entry.get('op_lvl', 0.0), 2),
                        'dea': round(entry.get('dead', 0.0), 2),
                        'bad': round(entry.get('badge', 0), 2),
                        'exp': round(entry.get('explore', 0), 2)
                    }
                    imageid_values[image_id] = image_values
        #elif filename.endswith('.csv.gz'):
        #    print(f"Found CSV file: {filename}")
        #    with gzip.open(os.path.join(newest_session, filename), 'rt') as file:
        #        df = pd.read_csv(file)
        #    if 'map_location' in df.columns:
        #        unique_locations = df['map_location'].unique()
        #        for location in unique_locations:
        #            count = len(df[df['map_location'] == location])
        #            print(f"Location: {location}, Count: {count}")
        #    else:
        #        print(f"'map_location' column not found in file: {filename}")

            # Sort the ZIP files by the highest step value
    zip_names.sort(key=lambda zip_name: int(zip_name.split('_')[1].split('_')[0]), reverse=True)

    # Get the creation timestamp of the newest session folder
    session_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(newest_session)).strftime('%Y-%m-%d %H:%M:%S')

    # Find the timestamp of the oldest image file
    oldest_file_time = min([os.path.getctime(os.path.join(image_dir, f)) for f in os.listdir(image_dir) if f.endswith('.jpeg')])
    first_detected_checkpoint_time = datetime.datetime.fromtimestamp(oldest_file_time).strftime('%Y-%m-%d %H:%M:%S')

    # Calculate the time since the first detected checkpoint
    current_time = datetime.datetime.now()
    time_since_first_checkpoint = current_time - datetime.datetime.fromtimestamp(oldest_file_time)
    time_since_first_checkpoint_str = str(time_since_first_checkpoint)

    # Create the updated HTML content with a softer red banner
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pokemon Plays AI</title>
        <style>
            .container {{
                max-width: 800px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                background-color: #282C35; /* Softer red */
                color: #fff;
                padding: 20px;
            }}
            .content {{
                padding: 20px;
            }}
            table {{
                width: 100%;
            }}
            td {{
                text-align: center;
                padding: 10px;
            }}
            img {{
                max-width: 100%;
            }}
        </style>
        <script src="https://cdn.tailwindcss.com"></script>
        <meta http-equiv="refresh" content="1">
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Pokemon Red Experiments</h1>
                <p>First Detected Checkpoint: {first_detected_checkpoint_time}</p>
                <p>Current Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Time Since First Detected Checkpoint: {time_since_first_checkpoint_str}</p>
                    """
    if zip_names:
        html_content += f'<a id="download_zip" href="{zip_names[0]}" download>Download checkpoint ({zip_names[0]})</a>'
                #<button id="download_zip">Download Most Recent ZIP ({zip_names[0]})</button>
    html_content += f"""
            </div>
            <div class="content">
                <table>
    """

    # Calculate the number of columns in the grid based on the number of images
    num_cols = min(len(image_names), 4)
    html_content += '<style>'
    html_content += 'table { border-collapse: collapse; }'
    html_content += 'td { padding: 0; margin: 0; }'
    html_content += '</style>'
    # Display image names and JSON download buttons in a grid layout
    for i, image_name in enumerate(image_names, start=1):
        if i > max_items:
            break  # Limit the number of displayed items
        
        image_id = image_name.split('_')[1].split('.')[0]
        if i % num_cols == 1:
            html_content += f'        <tr>\n'
        html_content += f'            <td style="margin: 0; padding: 0; text-align: left;">'
        html_content += f'<img src="{image_name}" alt="Image {i}" style="margin-bottom: 0px;"><br>'
        html_content += f'<span style="margin-bottom: 0px; margin-top: 0px; ">{image_name}</span><br>'
        # Add a button to download the JSON file
        # Check if this image_id has extracted values
        if image_id in json_names:
            json_name = json_names[image_id]
            #html_content += f'<button id="download_json_{image_id}" style="margin-top: 0px;">{json_name}</button>'
            html_content += f'<a href="{json_name}" download target="_blank">{json_name}</a>'

            # Check if this image_id has extracted values
            html_content += '<table style="border-collapse: collapse; border-spacing: 0; line-height: 1; ">'
            if image_id in imageid_values:
                values = imageid_values[image_id]
                # Display the truncated keys and values in two columns
                pairs = [f'{key}: {value}' for key, value in values.items()]
                for i in range(0, len(pairs), 2):
                    pair1 = pairs[i]
                    pair2 = pairs[i + 1] if i + 1 < len(pairs) else ''  # Handle odd number of pairs
                    html_content += '<tr class="m-0 p-0" style="text-align: left;">'  # Use Tailwind classes to set margin and padding to 0 for rows
                    html_content += f'<td>{pair1}</td>'
                    html_content += f'<td>{pair2}</td></tr>'
                html_content += '</table><br>'
                #html_content += f'<span class="trophy-icon">🏆</span>'
  
    html_content += '</td>'
    html_content += """                </table>
            </div>
        </div>
    </body>
    </html>
    """

    session_html_file = os.path.join(newest_session, output_file)

    # Save the updated HTML content to a file
    with open(session_html_file, 'w') as file:
        file.write(html_content)
    print(f"You can now open '{newest_session}/{output_file}' in the session directory to view your results.")

if __name__ == '__main__':
    parse_args()
    selected_checkpoint = None
    selected_checkpoint = show_menu(selected_checkpoint)
    main(selected_checkpoint)
