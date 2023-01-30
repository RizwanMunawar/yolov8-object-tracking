#............... Importing Libraries ..............
import cv2
import sys
import json
import torch
import hydra
import datetime;
from sort import *
from pathlib import Path
from random import randint
sys.path.insert(0, os.getcwd())

from yolo.utils.checks import check_imgsz
from yolo.engine.predictor import BasePredictor
from yolo.utils import DEFAULT_CONFIG, ROOT, ops
from yolo.utils.plotting import Annotator, colors, save_one_box


#............ Declaration of Variables ............
tracker = None
save_json_dir = None
rand_color_list = []


def gen_timestamp():
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    return int(ts)


#............ Save Json Output Dir ................
def create_json_out_path():

    global save_json_dir
    save_json_dir = increment_json_out_path(Path("Output Json") / "results", exist_ok=False)
    (save_json_dir).mkdir(parents=True, exist_ok=True) 


#.............. Initialize Tracker ................
def init_tracker():
    
    global tracker
    
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker =Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)


#.............. Write Json File ...................
def write_json(json_data):

    json_file_path = os.path.join(save_json_dir,'Output.json')
    with open(json_file_path,'a+') as json_file_out:
     json.dump(json_data, json_file_out, sort_keys = True, indent = 4,
               ensure_ascii = False)


#.......... Creating Json Directory ...............
def increment_json_out_path(path, exist_ok=False, sep='', mkdir=False):
    
    path = Path(path) 
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
    
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
    
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path


#.......... Random Color for Every Track ..........
def random_color_list(num):
    
    global rand_color_list
    
    rand_color_list = []
    
    for i in rage(0,num):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)


#..... Draw UI Border for text on Detections .....
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2-r), 2, color, 12)

    return img


#........... Draw UI Box on Detections ...........
def draw_ui_box(img,x, label=None, categories = None,identities = None,names = None,line_thickness=None):
    
    for i, x in enumerate(x):
        
        cat = int(categories[i]) if categories is not None else 0
        trkid = int(identities[i]) if identities is not None else 0
        
        label = names[cat]
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
        
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2,  (190, 150, 37), thickness=tl, lineType=cv2.LINE_AA)

        timestmp = gen_timestamp()
        #... Writing Json ...
        json_data = {
                
            "Bbox":
            {
                "x1":int(x[0]),
                "y1":int(x[1]),
                "x2":int(x[2]),
                "y2":int(x[3]),
            },

            "label": label,
            "Trk Id": trkid,
            "TimeStamp": timestmp
        }

        write_json(json_data)

        if label:

            tf = max(tl - 1, 1)  
            t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
            
            img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),(c1[0] + t_size[0], c1[1]+3),(232,98,161) , 1, 8, 2)
            cv2.putText(img, str(label), (c1[0], c1[1] - 2), 0, tl / 3,[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


#................ Class for Detection Predictor ................
class DetectionPredictor(BasePredictor):
    
    #init annotator function
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    #add logic for preprocessing function
    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    #add logic for post processing function
    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,self.args.conf,self.args.iou,agnostic=self.args.agnostic_nms,max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    #add logic to write results
    def write_results(self, idx, preds, batch):

        p, im, im0 = batch
        log_string = ""
        
        if len(im.shape) == 3:
            im = im[None]  
        self.seen += 1
        im0 = im0.copy()
        
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        
        else:
            frame = getattr(self.dataset, 'frame', 0)
        
        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
    
    
        # .................. Object Tracking ....................
        dets_to_sort = np.empty((0,6))
        
        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
        
        tracked_dets = tracker.update(dets_to_sort)
        tracks =tracker.getTrackers()
        
        for track in tracks:
            
            [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                        (255,0,255), thickness=1) 
                        for i,_ in  enumerate(track.centroidarr) if i < len(track.centroidarr)-1 ] 
                
        if len(tracked_dets)>0:
            bbox_xyxy = tracked_dets[:,:4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            draw_ui_box(im0,bbox_xyxy,categories=categories,identities = identities,names=self.model.names,line_thickness=2)
            

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):

    #initialze tracker
    init_tracker()

    #generate random colors for tracks
    # random_color_list(5000)
    
    #create output json dir
    create_json_out_path()

    #other model configs
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()