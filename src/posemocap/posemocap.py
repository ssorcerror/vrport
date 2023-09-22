import cv2
from cvzone.PoseModule import PoseDetector
from src.constants.main_constants import SCREEN_RATIO
from src.jobs.vrport_job import VRPortJob

class PoseMoCap:
    '''Identifies poses from video input and returns animation files fom poses.'''
    scr_ratio = SCREEN_RATIO
    roto_vert_res = 1080
    roto_hor_res = int(roto_vert_res * scr_ratio)
    desired_vert_res = 2160
    desired_hor_res = int(desired_vert_res * scr_ratio)

    def __init__(self, job:VRPortJob):
        self.job = job
        self.detector = PoseDetector()
        

    def get_poses(self):
        while (1):
            success, img = self.job.cv2cap.read()
            lmList, bbmoxInfo = self.detector.findPose(img)
            cv2.imshow("image",img)
            cv2.waitKey(1)