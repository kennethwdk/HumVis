import threading
import binascii
from time import sleep
import numpy as np
import cv2
import base64


class Camera(object):
    def __init__(self, model):
        self.model = model
        self.to_process = []
        self.to_pose_output = []
        self.to_mask_output = []
        self.to_part_output = []
        self.to_mesh_output = []
        self.to_output = []

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        str_image = self.to_process.pop(0)
        img = base64.b64decode(str_image)
        img_np = np.fromstring(img, dtype='uint8')
        frame = cv2.imdecode(img_np, 1)
        assert type(frame) == np.ndarray

        ret = self.model.get_analysis_results(frame, tracking=True)
        frame_pose, frame_mask, frame_part, frame_mesh = ret
        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_pose_output.append(frame_pose)
        self.to_mask_output.append(frame_mask)
        self.to_part_output.append(frame_part)
        self.to_mesh_output.append(frame_mesh)

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.005)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_pose_frame(self):
        while not self.to_pose_output:
            sleep(0.02)
        return self.to_pose_output.pop(0)
    
    def get_mask_frame(self):
        while not self.to_mask_output:
            sleep(0.02)
        return self.to_mask_output.pop(0)
    
    def get_part_frame(self):
        while not self.to_part_output:
            sleep(0.02)
        return self.to_part_output.pop(0)
    
    def get_mesh_frame(self):
        while not self.to_mesh_output:
            sleep(0.02)
        return self.to_mesh_output.pop(0)
