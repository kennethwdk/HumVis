from camera import Camera
from video import generate_video_results
from humvis import HumVis

model = HumVis()
camera = Camera(model)

from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename

import os
import cv2
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
app.config['UPLOAD_DIR'] = 'static/uploads'
print('index')

# index
@app.route('/')
def index():
    return render_template('demo_index.html')

# live part
@app.route('/live/')
def live_index():
    return render_template('live_index.html')

@app.route('/live/receiveimage', methods=["POST"])
def receiveimage():
    if request.method == "POST":
        json_data = request.get_json()
        str_image = json_data.get("imgData")
        camera.enqueue_input(str_image)
    
    return Response('upload')

# get image data
def gen_stream(output_type='pose'):
    while True:
        if output_type == 'pose':
            frame = camera.get_pose_frame()
        elif output_type == 'mask':
            frame = camera.get_mask_frame()
        elif output_type == 'part':
            frame = camera.get_part_frame()
        elif output_type == 'mesh':
            frame = camera.get_mesh_frame()
        else:
            raise ValueError('Unknown output type: {}'.format(output_type))
        assert type(frame) == np.ndarray

        ret1, jpeg = cv2.imencode('.jpg', frame)
        jpg_frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+jpg_frame + b'\r\b')

@app.route('/live/video_pose_feed')
def video_pose_feed():
    return Response(gen_stream('pose'), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/live/video_mask_feed')
def video_mask_feed():
    return Response(gen_stream('mask'), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/live/video_part_feed')
def video_part_feed():
    return Response(gen_stream('part'), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/live/video_mesh_feed')
def video_mesh_feed():
    return Response(gen_stream('mesh'), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/upload/', methods=['GET', 'POST'])
def upload_index():
    if request.method == 'POST':
        is_video = False
        if 'image' in request.files:
            file = request.files['image']
        elif 'video' in request.files:
            file = request.files['video']
            is_video = True
        else:
            raise ValueError('Unknow upload')
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_DIR']):
            os.makedirs(app.config['UPLOAD_DIR'])
        filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
        if not os.path.exists(filepath):
            file.save(filepath)
        if is_video:
            return redirect(url_for('.upload_video_result', filename=filename))
        else:
            return redirect(url_for('.upload_result', filename=filename))
    return render_template('upload_index.html')

@app.route('/upload/result', methods=['GET'])
def upload_result():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
    frame = cv2.imread(filepath)
    frame_pose, frame_mask, frame_part, frame_mesh = model.get_analysis_results(frame)

    frame_pose = frame_pose[:, :, ::-1]
    im = Image.fromarray(frame_pose)
    data = io.BytesIO()
    im.save(data, "JPEG")
    frame_pose_stram = base64.b64encode(data.getvalue())

    frame_mask = frame_mask[:, :, ::-1]
    im = Image.fromarray(frame_mask)
    data = io.BytesIO()
    im.save(data, "JPEG")
    frame_mask_stram = base64.b64encode(data.getvalue())

    frame_part = frame_part[:, :, ::-1]
    im = Image.fromarray(frame_part)
    data = io.BytesIO()
    im.save(data, "JPEG")
    frame_part_stram = base64.b64encode(data.getvalue())

    frame_mesh = frame_mesh[:, :, ::-1]
    im = Image.fromarray(frame_mesh)
    data = io.BytesIO()
    im.save(data, "JPEG")
    frame_mesh_stram = base64.b64encode(data.getvalue())

    return render_template('upload_result.html', filename=filename, pose_img=frame_pose_stram.decode('utf-8'), mask_img=frame_mask_stram.decode('utf-8'), part_img=frame_part_stram.decode('utf-8'), mesh_img=frame_mesh_stram.decode('utf-8'))

@app.route('/upload/videoresult', methods=['GET'])
def upload_video_result():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
    ret_filenames = generate_video_results(filepath, model, app.config['UPLOAD_DIR'])
    # ret_filenames = [filename, filename, filename, filename]
    return render_template('upload_video_result.html', filename=filename, pose_video=ret_filenames[0], mask_video=ret_filenames[1], part_video=ret_filenames[2], mesh_video=ret_filenames[3])

print('app run')
app.run(host='0.0.0.0', port=65432, debug=True, threaded=True)
