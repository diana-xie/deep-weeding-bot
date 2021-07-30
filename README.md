# deep-weeding-bot

Experimenting with YOLO v3 and v4. Detecting corn vs. weeds, as well as simulating a “robot” in the field that combines Arduino NMEA location data with weed detection to trigger an action when a weed is detected.

See Medium article on some mAP and cost analysis I did: [link](https://medium.com/@dianaxie/evaluating-map-for-an-object-detection-agribot-3e7bcb52623b?source=friends_link&sk=7597fe77cdda033cca4a125b26bc7187)

<img src="documentation/weed-object-detection-gif.gif?raw=true" width="500px">

- Adapted Yolov3/v4 code: [https://github.com/diana-xie/darknet_yolo](https://github.com/diana-xie/darknet_yolo)
- Libraries used: 
  * Training & implementing YOLO v3: [ImageAI](https://imageai.readthedocs.io/en/latest/customdetection/)
  * Labelling images: [labelImg](https://github.com/tzutalin/labelImg)
- Tutorials used:
  * [How to create your own Custom Object Detector](https://towardsdatascience.com/how-to-create-your-own-custom-object-detector-766cb11ccd1c)
  * [Learn to Augment Images and Multiple Bounding Boxes](https://medium.com/@a.karazhay/guide-augment-images-and-multiple-bounding-boxes-for-deep-learning-in-4-steps-with-the-notebook-9b263e414dac)
  * [xmlAnnotation](https://gist.github.com/calisir/568190a5e55a79e08be318c285688457)
  * [Arduino-Python interface](https://learn.adafruit.com/adafruit-ultimate-gps/circuitpython-parsing)

# Overview

## 1. Using YOLO v3 & ImageAI for custom object detection
### Goal: 
Create an object detection algorithm that distinguishes corn from weeds, on real-life video data collected in the field.

<b>1. Distinguish corn from weeds</b>
* as long as weeds can be distinguished from "other" entities the majority of time

<b>2. Detect weeds with high recall</b>
* robot will only target weeds for action, therefore it is more critical to produce high recall; i.e. correctly predict positives (i.e. detect "weed") out of all the actual positives in the dataset

<img src="documentation/sample-detection.jpg?raw=true" width="650px">

### Results: 
- Notebook: [yolo_v3_weeds.ipynb](https://github.com/diana-xie/deep-weeding-bot/blob/master/yolo_v3_weeds.ipynb)
- Data augmentation code: [yolo_setup](https://github.com/diana-xie/deep-weeding-bot/tree/master/yolo_setup)

### Use case:
Use case is to simulate a scenario in which a robot is deployed in a corn field to kill the majority of weeds, while minimizing tradeoff of damage to corn crops. Thus, main objective is to "play it safe" and mainly target weeds with high recall, rather than produce a response to corn detection. 

### Approach: 
Load and use pre-trained YOLO models w/ transfer learning to train YOLOv3 models on custom data. Custom data is actual video footage of robot traversing corn field, with weeds in path. The objects trained on are corn and weed labels in the footage, separated into images.
### Data:
- Video from robot with mounted GoPro camera, traversing actual corn fields with weeds growing on paths. 
- Manually labelled corn & weeds in footage (i.e. images extracted from footage), using [labelImg](https://github.com/tzutalin/labelImg)

<img src="documentation/labelImg.JPG?raw=true" width="650px">

- Pre-trained YOLO v3 H5 [file](https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4), which was used for transfer learning and re-trained on this custom dataset
- Data augmentation to distort images and expand dataset

<img src="documentation/sample-augmentation.jpg?raw=true" width="300px">

## 2. Simulating weeding robot
### Description: 
Simulate the above robot with mounted camera, where incoming video feed is automatically subjected to YOLO v3 custom object detection to trigger an action on the weed detected.

### Results: 
- Robot simulator code: [simulation.py](https://github.com/diana-xie/deep-weeding-bot/blob/master/simulation.py) 
- Demo: see sample of video below. [full video](https://drive.google.com/file/d/1HZSUiqsbrtV2g-8c1dhrIwg-geZHtCbZ/view?usp=sharing)

<center><img src="documentation/robot-simulator-demo.gif?raw=true" width="650px"></center>

### Use case:
Simulate a scenario in which the above robot moves, sees a plant, classifies it as a weed, and "kills" it (or triggers some action).

### Approach:
Whenever Arduino GPS <i>(i.e. robot "moves")</i> moves <i>(video: right side)</i>, a snapshot of what robot supposedly sees in real-life is taken and classified as weed <i>(video: red lettering on image popup)</i> or not <i>(green lettering)</i>. This keeps repeating until running out of snapshots fed to simulator.

### Data:
(Same video data as #1)
