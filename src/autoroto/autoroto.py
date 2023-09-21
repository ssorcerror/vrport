
import cv2 
import numpy as np
import math
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
from config import AUTOROTO_PATH
from pathlib import Path

import logging
logger = logging.getLogger(__name__)



class AutoRoto:
    '''Auto rotoscopy using ML image segmentation'''

    scr_ratio = 3840/2160
    roto_vert_res = 1080
    roto_hor_res = int(roto_vert_res * scr_ratio)
    desired_vert_res = 2160
    desired_hor_res = int(desired_vert_res * scr_ratio)
    fcn = None
    gpu_avail = torch.cuda.is_available()
    gpu_ind = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name()

    def __init__(self, video_path, save_path=None):
        self.cv2cap = cv2.VideoCapture(video_path)
        self.video_path_object = Path(video_path)
        self.video_filename = self.video_path_object.name
        self.video_name = self.video_filename.split('.')[0]
        if save_path == None:
            self.save_path = Path(self.video_path_object.parent, self.video_name, self.video_name  + '.png')
        else:
            self.save_path = save_path

    def rem_bg(self):
        while (1):
            ret, frame = self.cv2cap.read()
            if frame is None:
                break
            num_frame = self.cv2cap.get(cv2.CAP_PROP_POS_FRAMES)
            pil_matte = self.createMatte(frame, Path(AUTOROTO_PATH,f'matte_{num_frame}.jpg'), self.roto_vert_res)
            if pil_matte != None:
                cv2_matte = np.array(pil_matte.convert('RGB'))

                #convert RGB to BGR
                cv2_matte = cv2_matte[:,:,::-1].copy()
                cv2_matte = cv2.cvtColor(cv2_matte, cv2.COLOR_BGR2GRAY)
                threshold, cv2_mask = cv2.threshold(cv2_matte, 30, 250,
                                            cv2.THRESH_BINARY)
                mask_rsz = cv2.resize(cv2_mask, (self.desired_hor_res, self.desired_vert_res))
                _, alpha = cv2.threshold(mask_rsz, 100, 255, cv2.THRESH_BINARY)
                frame_rsz = cv2.resize(frame, (self.roto_hor_res, self.roto_vert_res))
                #dst = cv2.bitwise_and(frame, frame, mask=mask_rsz)
                b, g, r = cv2.split(frame)
                rgba = [b, g, r, alpha]
                dst = cv2.merge(rgba, 4)
                cv2.imwrite(str(self.save_path.split('.')[0]) + '_' + str(int(num_frame)) + '.' + str(self.save_path.split('.')[1]), dst)
                disp = cv2.resize(dst, (self.roto_hor_res, self.roto_vert_res))
                cv2.imshow('frame',disp)

            if cv2.waitKey(1) == ord('q'): # press "q" to quit
                break
    
    def getRotoModel(self):
        global fcn
        model = models.segmentation.fcn_resnet101(pretrained=True).cuda()
        fcn =  model.eval()

    # Define the helper function
    def decode_segmap(self, image, nc=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitorep
                (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def createMatte(self, imginp, matteName, size):
        if type(imginp)==str:
            img = Image.open(imginp)
        else:
            img = Image.fromarray(imginp)

        trf0 = T.Compose([T.Resize(size),
                            T.ToTensor()])
        trf = T.Compose([T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inp0 = trf0(img).to(device)
        inp = trf(inp0).unsqueeze(0)
        inp = inp
        if (self.fcn == None): self.getRotoModel()
        out = fcn(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()  
        #om = torch.argmax(out.squeeze(), dim=0).detach().cuda().numpy()
        rgb = self.decode_segmap(om)
        im = Image.fromarray(rgb)
        im.save(matteName)
        return Image.fromarray(rgb)
    
    def save_frame(self):
        '''Saves given frame near asset path directory'''
        pass