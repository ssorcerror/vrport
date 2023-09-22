from pathlib import Path
import cv2

class VRPortJob:
    '''Houses info necessary for a VR Port Job.'''
    def __init__(self, video_path, save_path=None, display_img=False):
        self.cv2cap = cv2.VideoCapture(video_path)
        self.video_path_object = Path(video_path)
        self.video_filename = self.video_path_object.name
        self.video_name = self.video_filename.split('.')[0]
        if save_path == None:
            self.save_path = Path(self.video_path_object.parent, self.video_name, self.video_name  + '.png')
        else:
            self.save_path = save_path
        self.display_img = display_img
