from flask import Response
from app import app
from app.detection.detect import generate_frames

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
