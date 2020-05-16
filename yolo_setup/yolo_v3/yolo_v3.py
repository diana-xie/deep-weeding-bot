import os
import shutil
import random
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection

images_path = 'images_train/'
labels_path = 'labels_train/'  # 'labels_path'
train_path = 'dataset_yolo_v3/train/'
validation_path = 'dataset_yolo_v3/validation/'

train_test_split = True

if train_test_split:
    # train test validation
    for image_file in os.listdir(images_path):
        labels_file = image_file.replace('.jpg', '.xml')

        try:
            if random.uniform(0, 1) > 0.2:
                shutil.copy(images_path + image_file, train_path + 'images/' + image_file)
                shutil.copy(labels_path + labels_file, train_path + 'annotations/' + labels_file)
            else:
                shutil.copy(images_path + image_file, validation_path +
                            'images/' + image_file)
                shutil.copy(labels_path + labels_file, validation_path +
                            'annotations/' + labels_file)

        except Exception as ex:
            print("Failed for image {}: ".format(ex))

"""
Train model
"""

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="dataset_yolo_v3/")
trainer.setTrainConfig(object_names_array=["corn", "weed"],
                       batch_size=2,
                       num_experiments=200,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

"""
Evaluate model
- evaluate how every model checkpoint performs via its mAP.
"""
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="dataset_yolo_v3")
metrics = trainer.evaluateModel(model_path="dataset_yolo_v3/models",
                                json_path="dataset_yolo_v3/json/detection_config.json",
                                iou_threshold=0.5,
                                object_threshold=0.9,
                                nms_threshold=0.5)

"""
Use model for detection
"""

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("dataset_yolo_v3/models/detection_model-ex-005--loss-0014.777.h5")
detector.setJsonPath("dataset_yolo_v3/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="test.jpg",
                                             output_image_path="ima-detected.jpg",
                                             minimum_percentage_probability=50)
