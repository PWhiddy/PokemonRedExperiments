import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import hashlib
from datetime import datetime
from flask import send_from_directory
import re  # Import the regular expression module

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'

# In-memory list to hold file metadata
files_data = []

@app.route('/metadata.txt')
def metadata():
    try:
        return send_from_directory('./uploads', 'metadata.txt')
    except FileNotFoundError:
        return "metadata.txt not found.", 404

@app.route('/highest_steps')
def highest_steps():
    read_metadata()  # Read metadata from file

    if not files_data:
        return "No files uploaded.", 404
    
    highest_steps_file = max(files_data, key=lambda x: x['steps'])
    directory, filename = os.path.split(highest_steps_file['filepath'])
    return send_from_directory(directory=directory, filename=filename)

# Function to read metadata from file
def read_metadata():
    global files_data
    try:
        with open('./uploads/metadata.txt', 'r') as f:
            files_data = json.load(f)
    except:
        files_data = []

# Function to write metadata to file
def write_metadata(data):
    with open('./uploads/metadata.txt', 'w') as f:
        json.dump(data, f)

# Read metadata from file on startup
read_metadata()

@app.route('/')
def index():
    read_metadata()  # Read metadata from file
    sorted_files = sorted(files_data, key=lambda x: x['steps'], reverse=True)
    return render_template('index.html', files=sorted_files)

# ...

@app.route('/upload', methods=['POST'])
def upload_file():
    global files_data

    uploaded_file = request.files['file']
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    sha1 = hashlib.sha1(uploaded_file.read()).hexdigest()[:10]
    uploaded_file.seek(0)
    original_filename = secure_filename(uploaded_file.filename)

    # Extract the 'steps' from the original filename
    match = re.search(r'poke_(\d+)_steps\.zip', original_filename)
    if match:
        steps = int(match.group(1))
    else:
        steps = None  # Default value if not found

    filename = f"poke_{steps}_steps_{sha1}_{timestamp}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    uploaded_file.save(filepath)

    file_info = {'filename': filename, 'filepath': filepath, 'timestamp': timestamp, 'steps': steps}
    files_data.append(file_info)

    files_data.sort(key=lambda x: x.get('steps', 0), reverse=True)
    write_metadata(files_data)

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)

