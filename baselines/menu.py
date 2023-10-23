import argparse
import os
import re
import requests
import json
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--menu', action='store_true', help='Show the menu')
    parser.add_argument('--restore', help='Restore from a URL or use the default URL')
    parser.add_argument('--upload', help='Upload to a URL or use the default URL')
    return parser.parse_args()

def list_all_sessions_and_pokes():
    all_folders = os.listdir()
    session_folders = [folder for folder in all_folders if re.match(r'session_[0-9a-fA-F]{8}', folder)]
    session_dict = {}

    for session_folder in session_folders:
        poke_files = glob.glob(f"{session_folder}/poke_*_steps.zip")
        if poke_files:
            largest_poke_file = max(poke_files, key=lambda x: int(re.search(r'poke_(\d+)_steps', x).group(1)))
            largest_step = int(re.search(r'poke_(\d+)_steps', largest_poke_file).group(1))
            session_dict[session_folder] = largest_step

    # Add downloaded checkpoints to the session_dict
    downloaded_checkpoints = os.listdir('downloaded_checkpoints')
    for downloaded_checkpoint in downloaded_checkpoints:
        if downloaded_checkpoint.endswith('.zip'):
            session_name = 'downloaded_checkpoints/' + downloaded_checkpoint
            print('downloaded_checkpoints/' + downloaded_checkpoint)
            session_dict[session_name] = 'downloaded_checkpoints/' + downloaded_checkpoint
            #largest_step = int(re.search(r'poke_(\d+)_steps', downloaded_checkpoint).group(1))
            session_dict[session_name] = largest_step

    sorted_session_dict = {k: v for k, v in sorted(session_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_session_dict

def remote_actions():
    url = input("Enter the URL for resuming or leave it blank for the default: ")
    if url.strip() == "":
        response = requests.get("http://127.0.0.1:5000/metadata.txt")
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

        print(f"\nAvailable checkpoints from the server:")
        for i, entry in enumerate(server_metadata):
            print(f"{i + 1}. {entry['filename']}")

        server_selection = int(input("Enter the number of the checkpoint you want to download: "))
        download_directory = "downloaded_checkpoints"
        os.makedirs(download_directory, exist_ok=True)

        download_response = requests.get(f"http://127.0.0.1:5000/uploads/{server_metadata[server_selection - 1]['filename']}")
        if download_response.status_code != 200:
            print("Failed to download the selected checkpoint.")
            return None

        selected_server_entry = server_metadata[server_selection - 1]['filename']
        with open(f"downloaded_checkpoints/{selected_server_entry}", 'wb') as f:
            f.write(download_response.content)
        return f"downloaded_checkpoints/{selected_server_entry}"

def show_menu(sess_path):
    selected_checkpoint = None
    session_dict = list_all_sessions_and_pokes()

    if not session_dict:
        print("No checkpoints found.")
        return selected_checkpoint

    print(f"\nAvailable sessions sorted by their largest checkpoints:")
    for i, (session, largest_step) in enumerate(session_dict.items()):
        print(f"  {i + 1}. {session}/poke-{largest_step}_steps.zip")

    print("\n  95. Future-Delete Saved Files")
    print("  96. Resume from remote")
    print("  97. Upload to remote")
    print("  98. Exit")
    print("  99. Start a new run")
    menu_selection = input("Enter the number of the menu option: ")

    if menu_selection == '96':
        selected_checkpoint = remote_actions()
    elif menu_selection.isdigit():
        # If a numeric option is selected, try to use it as a checkpoint
        selection = int(menu_selection)
        if 1 <= selection <= len(session_dict):
            selected_session = list(session_dict.keys())[selection - 1]
            selected_step = session_dict[selected_session]
            selected_checkpoint = f"{selected_session}/poke_{selected_step}_steps.zip"
        else:
            print("Invalid selection.")
    elif menu_selection == '97':
        selection = input("Enter your selection for remote upload: ")
        upload(selection, session_dict)
    elif menu_selection == '98':
        print("Exiting the menu.")
    elif menu_selection == '99':
        selected_checkpoint = None
    else:
        print("Invalid selection.")
        return selected_checkpoint

    return selected_checkpoint

# Define a function to restore from a URL or the default URL
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

# Define a function to upload to a URL or the default URL
def upload(selection, session_dict):
    try:
        upload_selection = int(selection)
        selected_session = list(session_dict.keys())[upload_selection - 1]
        selected_step = session_dict[selected_session]
        file_path = f"{selected_session}/poke_{selected_step}_steps.zip"

        print(file_path)
        upload_command = f"curl -X POST -F file=@{file_path} http://127.0.0.1:5000/upload"
        subprocess.run(upload_command, shell=True)
    except (ValueError, IndexError):
        print("Invalid selection.")

if __name__ == '__main__':
    args = parse_args()
    if args.menu:
        selected_checkpoint = show_menu(sess_path)
        if args.restore and selected_checkpoint is None:
            selected_checkpoint = restore(args.restore, None)
        if args.upload:
            upload(args.upload, None)
