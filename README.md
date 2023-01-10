# yolov8-object-tracking

### Features
- Object Tracks
- Different Color for every track
- Video/Image/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Selection of specific class ID for tracking
- Development of dashboard for YOLOv8

### Train YOLOv8 on Custom Data
- https://chr043416.medium.com/train-yolov8-on-custom-data-6d28cd348262

### Steps to run Code
- Clone the repository
```
https://github.com/RizwanMunawar/yolov8-object-tracking.git
```

- Goto cloned folder
```
cd yolov8-object-tracking
```

- Install the ultralytics package
```
pip install ultralytics==8.0.0
```

- Do Tracking with mentioned command below
```
#video file
python yolo\v8\detect\detect_and_trk.py model=yolov8s.pt source="test.mp4" show=True

#imagefile
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source="yolo\data\datasets\images\bus.jpg" model=yolov8m.pt

#Webcam
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source=0 model=yolov8m.pt show=True

#External Camera
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source=1 model=yolov8m.pt show=True
```

- Output file will be created in the working-dir/runs/detect/obj-tracking with original filename
