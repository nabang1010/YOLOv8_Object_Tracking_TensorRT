from models import TRTModule
import argparse
from time import time
import cv2
from pathlib import Path
import torch
import ctypes
from bytetrack.byte_tracker import BYTETracker

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from datetime import datetime, timedelta
import json
import numpy as np
import random



class ROI:
    def __init__(self, x1, y1, x2, y2, roi_id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.roi_id = roi_id
        self.count = 0
        

DICT_ROIS = {}
DEBOUNCE_PERIOD = timedelta(seconds=2)
person_tracker = {}
debounce_tracker = {}

color_dict = {}

def get_random_color(id):
    if id not in color_dict:
        color_dict[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_dict[id]



def main(args):
    args_bytetrack = argparse.Namespace()
    args_bytetrack.track_thresh = 0.2
    args_bytetrack.track_buffer = 200
    args_bytetrack.mot20 = True
    args_bytetrack.match_thresh = 0.7

    tracker = BYTETracker(args_bytetrack)
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    fps = 0
    # input video
    cap = cv2.VideoCapture(args.vid)
    # input webcam
    # cap = cv2.VideoCapture(0)
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (video_width,video_height))
    while(True):
        ret, frame = cap.read()
        
        if frame is None:
            print('No image input!')
            continue
        
        start = float(time())
        fps_str = "FPS:"
        fps_str += "{:.2f}".format(fps)
        bgr = frame
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        tensor = blob(rgb, return_seg=False)
        
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        
        tensor = torch.asarray(tensor, device=device)
        
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        # print(labels)
        
        if bboxes.numel() == 0:
            continue
        
        bboxes -= dwdh
        bboxes /= ratio
        output = []
        for (bbox, score, label) in zip(bboxes, scores, labels):
            if label == 0 and score.item() > 0.2:
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                # x1, y1, x2, y2, conf
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], score.item()])
        output = np.array(output)
                
        info_imgs = frame.shape[:2]
        img_size = info_imgs
        
        if output != []:
            online_targets = tracker.update(output, info_imgs, img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                
                if args.show:
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), get_random_color(tid), 2)
                    cv2.putText(frame, str(tid), (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        end = float(time())
        
  
        

    
        fps = 1/(end - start)
        print(fps_str)
        cv2.putText(frame, "YOLOV8-BYTETrack", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, fps_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if args.show:
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
        out.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    # tracker_trt.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default='../models/engine/yolov8n.engine')
    parser.add_argument('--vid', type=str, help='Video file', default='../sample_video/sample_2.mp4')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the results')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

