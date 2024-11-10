# yolov8-object-tracking 
#### [ultralytics==8.0.0]


### Features
- Object Tracks
- Different Color for every track
- Video/Image/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Selection of specific class ID for tracking
- Development of dashboard for YOLOv8

### 🙀 Train YOLOv8 on Custom Data
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


**Some of my articles/research papers | computer vision awesome resources for learning | How do I appear to the world? 🚀**

[Ultralytics YOLO11: Object Detection and Instance Segmentation🤯](https://muhammadrizwanmunawar.medium.com/ultralytics-yolo11-object-detection-and-instance-segmentation-88ef0239a811) ![Published Date](https://img.shields.io/badge/published_Date-2024--10--27-brightgreen)

[Parking Management using Ultralytics YOLO11](https://muhammadrizwanmunawar.medium.com/parking-management-using-ultralytics-yolo11-fba4c6bc62bc) ![Published Date](https://img.shields.io/badge/published_Date-2024--11--10-brightgreen)

[My 🖐️Computer Vision Hobby Projects that Yielded Earnings](https://muhammadrizwanmunawar.medium.com/my-️computer-vision-hobby-projects-that-yielded-earnings-7923c9b9eead) ![Published Date](https://img.shields.io/badge/published_Date-2023--09--10-brightgreen)

[Best Resources to Learn Computer Vision](https://muhammadrizwanmunawar.medium.com/best-resources-to-learn-computer-vision-311352ed0833) ![Published Date](https://img.shields.io/badge/published_Date-2023--06--30-brightgreen)

[Roadmap for Computer Vision Engineer](https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c)  ![Published Date](https://img.shields.io/badge/published_Date-2022--08--07-brightgreen)

[How did I spend 2022 in the Computer Vision Field](https://www.linkedin.com/pulse/how-did-i-spend-2022-computer-vision-field-muhammad-rizwan-munawar) ![Published Date](https://img.shields.io/badge/published_Date-2022--12--20-brightgreen)

[Domain Feature Mapping with YOLOv7 for Automated Edge-Based Pallet Racking Inspections](https://www.mdpi.com/1424-8220/22/18/6927) ![Published Date](https://img.shields.io/badge/published_Date-2022--09--13-brightgreen)

[Exudate Regeneration for Automated Exudate Detection in Retinal Fundus Images](https://ieeexplore.ieee.org/document/9885192) ![Published Date](https://img.shields.io/badge/published_Date-2022--09--12-brightgreen)

[Feature Mapping for Rice Leaf Defect Detection Based on a Custom Convolutional Architecture](https://www.mdpi.com/2304-8158/11/23/3914) ![Published Date](https://img.shields.io/badge/published_Date-2022--12--04-brightgreen)

[Yolov5, Yolo-x, Yolo-r, Yolov7 Performance Comparison: A Survey](https://aircconline.com/csit/papers/vol12/csit121602.pdf)  ![Published Date](https://img.shields.io/badge/published_Date-2022--09--24-brightgreen)

[Explainable AI in Drug Sensitivity Prediction on Cancer Cell Lines](https://ieeexplore.ieee.org/document/9922931)  ![Published Date](https://img.shields.io/badge/published_Date-2022--09--23-brightgreen)

[Train YOLOv8 on Custom Data](https://medium.com/augmented-startups/train-yolov8-on-custom-data-6d28cd348262)  ![Published Date](https://img.shields.io/badge/published_Date-2022--09--23-brightgreen)


**More Information**

For more details, you can reach out to me on [Medium](https://chr043416.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)
