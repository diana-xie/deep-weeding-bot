import pandas as pd
from keras.models import load_model
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from time import sleep
import serial

# Simulator params
DEBUG_MODE = True
ARDUINO_PORT = 'COM5'

# Global paths
OUTPUT_DIRECTORY = "./outputs/"
LABEL_DIRECTORY = "./labels/"
MODEL_DIRECTORY = "./models/"
MODEL_URL = "https://nextcloud.qriscloud.org.au/index.php/s/Y7EhlkVMYCqxdg2/download"
MODEL_ZIP_FILE = "./models/models.zip"
IMG_DIRECTORY = "./images/"
IMG_URL = "https://nextcloud.qriscloud.org.au/index.php/s/a3KxPawpqkiorST/download"
IMG_ZIP_FILE = "./images/images.zip"

# Global variables
RAW_IMG_SIZE = (256, 256)
IMG_SIZE = (224, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
MAX_EPOCH = 2
BATCH_SIZE = 32
FOLDS = 5
STOPPING_PATIENCE = 32
LR_PATIENCE = 16
INITIAL_LR = 0.0001
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']

CLASS_DICT = dict(zip(CLASSES, CLASS_NAMES))

# Load model
model = load_model('models/resnet.hdf5')
# Load labels
data = pd.read_csv(LABEL_DIRECTORY + "labels.csv")

# Setup
filenames = list(data['Filename'])
actual_labels = list(data['Species'])
weed_labels = ['Siam Weed', 'Snake weed', 'Prickly acacia', 'Chinee Apple']  # Set weeds

""" """ """ """ """ """ """ """ """ """ """ """
""" Perform inference & show results """
""" """ """ """ """ """ """ """ """ """ """ """


def show_image(img: np.array, plant_name: str, is_weed: bool = False):
    plt.figure()
    plt.imshow(img[0] * 255)
    if is_weed:
        plt.text(15, 30, 'weed', fontsize=30, color='red')
    else:
        plt.text(15, 30, plant_name, fontsize=30, color='green')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')


def make_inference(file_idx: int, model):
    """
    :param file_idx: indx of file in file list
    :param model: loaded model
    :return: 
    """
    # Load image
    filename = filenames[file_idx]
    img = imread(IMG_DIRECTORY + filename)
    # Resize to 224x224
    img = resize(img, (224, 224))
    # Map to batch
    img = np.expand_dims(img, axis=0)
    # Scale from int to float
    img = img * 1. / 255

    # Predict label
    prediction = model.predict(img, batch_size=1, verbose=0)
    y_pred = np.argmax(prediction[0][:7], axis=1)
    # y_pred[np.max(prediction, axis=1) < 1 / 9] = 8

    # Show results
    is_weed = False
    plant_name = actual_labels[file_idx]
    predicted_name = CLASS_NAMES[y_pred]
    if predicted_name in weed_labels:
        is_weed = True

    show_image(img=img, plant_name=predicted_name, is_weed=is_weed)

    return predicted_name, plant_name, is_weed


# Blink the Arduino when the robot "sees" a "weed"
def blink_arduino():
    ser = serial.Serial(ARDUINO_PORT, 9800, timeout=1)
    ser.write(b'H')   # send the pyte string 'H'
    ser.write(b'L')


def robot_simulation():

    # store 'N' and 'W' coordinates
    list_n = []  # store 'N' coordinates
    list_w = []  # store 'W' coordinates

    # tracker to determine if last coord was same as latest
    track_lat = 0
    track_lon = 0

    # counter to present new image
    counter = 0

    # store labels_predicted
    labels_predicted = []
    labels_actual = []

    # for debugging inference
    if DEBUG_MODE:
        while counter < len(filenames):
            prediction, actual = make_inference(file_idx=counter, model=model)
            labels_predicted.append(prediction)
            labels_actual.append(actual)
            counter += 1

    # for regular run with Arduino
    else:

        with serial.Serial(ARDUINO_PORT, baudrate=115200, timeout=1) as ser:

            # read from the serial output
            # while counter < len(filenames):
            while counter < len(filenames[:100]):

                line = ser.readline().decode('ascii', errors='replace')
                response = line.split(',')

                if len(response) > 4 and line.find("$GPRMC") == 0:

                    try:
                        lat = ((float(response[3]) / 100.00) - (int(float(response[3]) / 100.00))) / 0.6 + int(
                            float(response[3]) / 100.00)  # response[4]  # 'N'
                        lon = ((float(response[5]) / 100.00) - (int(float(response[5]) / 100.00))) / 0.6 + int(
                            float(response[5]) / 100.00)  # response[6]  # 'W'
                        message = "{} {}, {} {}".format(str(lat), response[4], str(lon), response[6])
                        print(message)
                        list_n.append(lat)
                        list_w.append(lon)

                        # robot moved, snapshots new "scene", now classify weeds
                        if (lat != track_lat) or (lon != track_lon):
                            print("'Saw' image, so running plant detection and classification")
                            # prediction, actual = make_inference(file_idx=counter, model=model)
                            prediction, actual, is_weed = make_inference(file_idx=counter, model=model)
                            labels_predicted.append(prediction)
                            labels_actual.append(actual)

                            counter += 1

                        # track, to see if robot "moves" next round
                        track_lat, track_lon = lat, lon

                    except Exception as ex:
                        print('Could not get lat & lon, due to {}'.format(ex))

                sleep(.05)

    # pd.DataFrame([labels_predicted, labels_actual], columns=['predicted', 'actual'])
    print("Finished simulation")


robot_simulation()


