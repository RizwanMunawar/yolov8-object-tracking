 # Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import cv2
import time

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top leftfrom collections import deque (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    
    # Bottom right
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


def draw_ui_box(img,x, label=None,line_thickness=None,color=None):
        
    # line/font thickness
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2,  color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:

        # font thickness
        tf = max(tl - 1, 1)  
        t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 4, thickness=tf)[0]
        
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),(c1[0] + t_size[0], c1[1]+3),(232,98,161) , 1, 8, 2)

        # cv2.line(img, c1, c2, color, 30)
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, str(label), (c1[0], c1[1] - 2), 0, tl / 4,[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # cv2.putText(mask_img, str(label), (c1[0], c1[1] - 2), 0, tl / 3,[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def init_fps():
    global fps_end_time 
    global count 
    global start_time
    count = 0
    # global fps_start_time 
    # fps_start_time = 0
    fps_end_time = 0
    
    
def draw_text(img,text,font=cv2.FONT_HERSHEY_SIMPLEX,pos=(0, 0),font_scale=1,font_thickness=2,text_color=(0, 255, 0),text_color_bg=(0, 0, 0),):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

class DetectionPredictor(BasePredictor):
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                # self.annotator.box_label(xyxy, label, color=colors(c, True))
                draw_ui_box(im0,xyxy,label,line_thickness=3,color=(0, 255, 0))
                        
                # x1 = int(xyxy[0])
                # y1 = int(xyxy[1])
                # x2 = int(xyxy[2])
                # y2 = int(xyxy[3])
                # label ="fplayer"
                # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # cv2.rectangle(im0, (x1,y1), (x2,y2), (0,147,255), 3)
                # cv2.rectangle(im0, (x1, y1 - 20), (x1 + w, y1), (0,255,73), -1)
                # cv2.putText(im0, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.6, [0,0, 0], 1)
        
                # cv2.rectangle(im0, p1, p2, (0,255,127), thickness=2)
                # tf = max(w - 1, 1)  # font thickness
                # w, h = cv2.getTextSize(label, 0, fontScale=w / 3, thickness=tf)[0]  # text width, height
                # # outside = p1[1] - h >= 3
                # # p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # cv2.rectangle(im0, p1, p2, (0,255,127), -1, cv2.LINE_AA)  # filled
                # cv2.putText(im0,
                #             label, 
                #             (p1[0], p1[1]),
                #             0,
                #             w / 3,
                #             (255,255,255),
                #             thickness=tf,
                            # lineType=cv2.LINE_AA)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
                
                # cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
                # cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                #             0.6, [255, 255, 255], 1)
                
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)
        

        fps_end_time = time.time()

        if frame == 1:
            global start_time
            global average_fps
            start_time = 0
            average_fps = []
        
        fps_diff_time = fps_end_time - start_time
        fps = 1/fps_diff_time
        average_fps.append(fps)
        start_time = fps_end_time
        
        av_fps = sum(average_fps)/len(average_fps)
        fps_text="FPS:{:.2f}".format(av_fps)
        draw_text(
                    im0,
                    f"FPS: {av_fps:0.1f}",
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

        # cv2.putText(im0, fps_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):

    init_fps()
    
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
