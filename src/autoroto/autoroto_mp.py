
import cv2 
import numpy as np
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import MODEL_PATH

import logging
logger = logging.getLogger(__name__)



class AutoRoto_mp:
    '''Auto rotoscopy using Mediapipe image segmentation'''
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    BG_COLOR = (192, 192, 192) # gray
    MASK_COLOR = (255, 255, 255) # white
    # Create the options that will be used for ImageSegmenter
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    output_type = python.vision.ImageSegmenterOptions.OutputType.CATEGORY_MASK
    options = vision.ImageSegmenterOptions(base_options=base_options, output_type=output_type)  
    segmenter = vision.ImageSegmenter.create_from_options(options)

    def __init__(self, video_path, desired_height=1080, desired_width=480, std_scr_ratio=True):
        self.cv2cap = cv2.VideoCapture(video_path)
        ret, frame = self.cv2cap.read()
        #self.cv2cap.set(cv2.CAP_PROP_POS_FRAMES,1000)
        # Height and width that will be used by the model
        if std_scr_ratio:
            self.desired_height = desired_height
            self.desired_width = self.desired_height * 3840/2160
        else:
            self.desired_height = desired_height
            self.desired_width = desired_width

    def rem_bg(self):
        self.cv2cap = cv2.VideoCapture(0)
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            while self.cv2cap.isOpened():
                success, frame = self.cv2cap.read()
                if not success:
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                # Draw the pose annotation on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        self.cv2cap.release()

    def rem_bg2(self):
        while self.cv2cap.isOpened():
            success, frame = self.cv2cap.read()
            if not success:
                break
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            # Retrieve the masks for the segmented image
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)#[:,:,::-1])
            image_data = image.numpy_view()
            segmentation_result = self.segmenter.segment(image)
            #category_mask = segmentation_result.category_mask
            fg_img = np.zeros(image_data.shape, dtype=np.uint8)
            # frame.flags.writeable = False
            # frame = self.resize_image(frame)
            #frame = cv2.resize(frame, (self.desired_width, self.desired_height))
            show_img = self.resize_image(segmentation_result[0].numpy_view())
            org_img = self.resize_image(image_data)
            cv2.imshow('MediaPipe Pose', show_img)
            cv2.imshow('Original Image', org_img)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
    # Performs resizing and showing the image
    def resize_image(self, image):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (int(self.desired_width), math.floor(h/(w/self.desired_width))))
        else:
            img = cv2.resize(image, (math.floor(w/(h/self.desired_height)), self.desired_height))
        return img
