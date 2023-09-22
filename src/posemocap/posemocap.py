import cv2
from cvzone.PoseModule import PoseDetector
from src.constants.main_constants import SCREEN_RATIO

class PoseMoCap:
    '''Identifies poses from video input and returns motion capture bvh from poses'''
    scr_ratio = SCREEN_RATIO
    roto_vert_res = 1080
    roto_hor_res = int(roto_vert_res * scr_ratio)
    desired_vert_res = 2160
    desired_hor_res = int(desired_vert_res * scr_ratio)

    def __init__(self, video_path, save_path=None, display_img=False):
        pass

    def get_poses(self):
        pass