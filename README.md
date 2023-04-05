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
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source="path to image"

#Webcam
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source=0 show=True

#External Camera
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source=1 show=True
```

- Output file will be created in the working-dir/runs/detect/train with original filename


### Results
<table>
  <tr>
    <td>YOLOv8s Object Tracking</td>
    <td>YOLOv8m Object Tracking</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/211671576-7d39829a-f8f5-4e25-b30a-530548c11a24.png"></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/211672010-7415ef8b-7941-4545-8434-377d94675299.png"></td>
  </tr>
 </table>

### References
- https://github.com/abewley/sort
- https://github.com/ultralytics/ultralytics


### My Articles
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/04/maximizing-efficiency-on-construction.html 🔥
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/instance-segmentation-vs-semantic.html ✅
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/mastering-image-classification.html 🔥
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/object-detection-in-agriculture.html ✅
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/techniques-for-accurate-data-annotation.html ✅
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/object-tracking-using-bytetrack.html 🔥
- https://muhammadrizwanmunawarvisionai.blogspot.com/2023/03/pose-estimation-computer-vision.html ✅
- https://chr043416.medium.com/train-yolov8-on-custom-data-6d28cd348262
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f

<a href= "https://www.linkedin.com/posts/muhammadrizwanmunawar_yolov8-aiinnovation-computervision-activity-7018504358329790465-qmde?utm_source=share&utm_medium=member_desktop">LinkedIn Post on YOLOv8 Release</a>

Don't forget to check out my <a href= "https://muhammadrizwanmunawar.com/">Services</a>

For more details, you can reach out to me on [Medium](https://chr043416.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)
