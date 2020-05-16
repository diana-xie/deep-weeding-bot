"""
From:
- https://github.com/bjornstenger/xml2yolo
- https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
"""

from xml.dom import minidom
import os
import glob
import shutil

lut = {}
lut["corn"] = 0
lut["weed"] = 1


def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_xml2yolo(lut, path):
    for fname in glob.glob(path + '*.xml'):

        xmldoc = minidom.parse(fname)

        fname_out = (fname[:-4] + '.txt')

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid = (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print("warning: label '%s' not in look-up table" % classid)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width, height), b)
                # print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')


# get list of txt filenames, for Darknet implementation of YOLO v4
def get_filenames(path_train: str, path_validate: str):
    """
    :param path_train:
    :param path_validate:
    :return:
    """

    """ Train dataset"""

    # List all the jpg's as specified by Darknet setup instructions
    files = glob.glob(path_train + '/*.jpg')
    with open('train.txt', 'w') as in_files:  # named according to Darknet requirements
        for eachfile in files:
            in_files.write('data/obj/' + os.path.basename(eachfile) + '\n')

    # move file to specified location in Darknet repo
    shutil.move(r"C:\Users\Diana\project_code\red-barn\yolo_setup\yolo_v4\train.txt",
                r"C:\Users\Diana\project_code\darknet\data\train.txt")

    """ Validate dataset"""

    # List all the jpg's as specified by Darknet setup instructions
    files = glob.glob(path_validate + '/*.jpg')
    with open('test.txt', 'w') as in_files:  # named according to Darknet requirements
        for eachfile in files:
            in_files.write('data/obj/' + os.path.basename(eachfile) + '\n')

    # move file to specified location in Darknet repo
    shutil.move(r"C:\Users\Diana\project_code\red-barn\yolo_setup\yolo_v4\test.txt",
                r"C:\Users\Diana\project_code\darknet\data\test.txt")


def main():
    convert_xml2yolo(lut, os.path.dirname(__file__) + '/labels_validate/')
    get_filenames(path_train=os.path.dirname(__file__) + '/images_train/',
                  path_validate=os.path.dirname(__file__) + '/images_validate/')


if __name__ == '__main__':
    main()
