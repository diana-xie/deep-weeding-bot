import cv2
import numpy as np
import argparse
import time
import os
import shutil
import random
from imageai.Detection.Custom import DetectionModelTrainer

images_path = 'resized_images/'
labels_path = 'labels_path/'
train_path = 'dataset/train/'
validation_path = 'dataset/validation/'

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

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="dataset/")
trainer.setTrainConfig(object_names_array=["obj1", "obj2"],
                       batch_size=16,
                       num_experiments=200,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()