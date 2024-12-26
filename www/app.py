import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import hashlib
from datetime import datetime
import re

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['METADATA_FILE'] = './uploads/metadata.txt'

# Initialize the files_data list with metadata on startup
files_data = []
@app.route('/uploads')
def list_files():
    """Display a list of uploaded files for download."""
    read_metadata()
    sorted_files = sorted(files_data, key=lambda x: x.get('steps', 0), reverse=True)
    return render_template('list_files.html', files=sorted_files)

def read_metadata():
    """Read metadata from the metadata file."""
    global files_data
    try:
        with open(app.config['METADATA_FILE'], 'r') as f:
            files_data = json.load(f)
    except FileNotFoundError:
        files_data = []
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")

def write_metadata(data):
    """Write metadata to the metadata file."""
    with open(app.config['METADATA_FILE'], 'w') as f:
        json.dump(data, f)

# Read metadata from file on startup
read_metadata()

@app.route('/')
def index():
    """Display a list of uploaded files with metadata sorted by steps."""
    read_metadata()
    sorted_files = sorted(files_data, key=lambda x: x.get('steps', 0), reverse=True)
    return render_template('index.html', files=sorted_files)

@app.route('/uploads/<filename>')
def download_file(filename):
    """Download an uploaded file by providing the filename."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a file, extract metadata, and save it with metadata."""
    global files_data

    uploaded_file = request.files['file']

    # Generate a unique filename using timestamp and SHA1 hash
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    sha1 = hashlib.sha1(uploaded_file.read()).hexdigest()[:10]
    uploaded_file.seek(0)
    original_filename = secure_filename(uploaded_file.filename)

    # Extract the 'steps' from the original filename using regex
    match = re.search(r'poke_(\d+)_steps\.zip', original_filename)
    if match:
        steps = int(match.group(1))
    else:
        steps = None  # Default value if not found

    filename = f"poke_{steps}_steps_{sha1}_{timestamp}.zip"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if a file with the same 'steps' value already exists
    existing_entry = next((entry for entry in files_data if entry.get('steps') == steps), None)

    if existing_entry:
        # Update the existing entry
        existing_entry['filename'] = filename
        existing_entry['filepath'] = filepath
        existing_entry['timestamp'] = timestamp
    else:
        # Create a new entry
        file_info = {'filename': filename, 'filepath': filepath, 'timestamp': timestamp, 'steps': steps}
        files_data.append(file_info)

    # Save the uploaded file to the specified filepath
    uploaded_file.save(filepath)

    # Sort the metadata by 'steps' in reverse order
    files_data.sort(key=lambda x: x.get('steps', 0), reverse=True)

    # Write metadata to the metadata file
    write_metadata(files_data)

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
