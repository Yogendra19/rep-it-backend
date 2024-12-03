import os

def allowed_file(filename, extensions=['mp4', 'avi', 'mov']):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions
