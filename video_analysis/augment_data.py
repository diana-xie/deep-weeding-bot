import glob
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import xml.etree.cElementTree as ET
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug import augmenters as iaa
import imageio
import re
import os
from PIL import Image

resize_images = False

# Image augmentor
aug = iaa.SomeOf(2, [
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(rotate=(-60, 60)),
    iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
    iaa.Fliplr(1),
    iaa.Multiply((0.5, 1.5)),
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.03 * 255, 0.05 * 255)),
    iaa.Add((-25, 25)),
    iaa.MotionBlur(k=15),
    iaa.MultiplySaturation((0.5, 1.5)),
    iaa.LogContrast(gain=(0.6, 1.4)),
    iaa.Flipud(1)
])


def get_resized_images():
    # Define here your paths
    original_images_path = 'images/'
    resized_images_path = 'resized_images/'
    new_width = 416
    new_height = 416
    for image_file in os.listdir(original_images_path):
        ima = Image.open(original_images_path + image_file)
        new_ima = ima.resize((new_width, new_height), Image.ANTIALIAS)
        new_ima.save(resized_images_path + image_file)


# Convert XML files to CSV
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def csv_to_xml(CSV_PATH, RESIZED_IMAGES_PATH, LABELS_PATH, folder):
    f = open(CSV_PATH, 'r')
    reader = csv.reader(f)
    header = next(reader)
    old_filename = None
    for row in reader:
        filename = row[0]
        # if filename == 'aug0_frame920.jpg':
        #     print(filename)
        #     print(filename)
        if filename == old_filename:
            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = row[3]
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = row[4]
            ET.SubElement(bndbox, 'ymin').text = row[5]
            ET.SubElement(bndbox, 'xmax').text = row[6]
            ET.SubElement(bndbox, 'ymax').text = row[7]
        else:
            if old_filename is not None:
                labels_file = old_filename.replace('.jpg', '.xml')
                tree = ET.ElementTree(annotation)
                tree.write(LABELS_PATH + labels_file)

            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'folder').text = folder
            ET.SubElement(annotation, 'filename').text = filename
            ET.SubElement(annotation, 'path').text =  RESIZED_IMAGES_PATH + filename
            source = ET.SubElement(annotation, 'source')
            ET.SubElement(source, 'database').text = 'Unknown'
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = row[1]
            ET.SubElement(size, 'height').text = row[2]
            ET.SubElement(size, 'depth').text = '3'
            ET.SubElement(annotation, 'segmented').text = '0'

            object = ET.SubElement(annotation, 'object')
            ET.SubElement(object, 'name').text = row[3]
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = row[4]
            ET.SubElement(bndbox, 'ymin').text = row[5]
            ET.SubElement(bndbox, 'xmax').text = row[6]
            ET.SubElement(bndbox, 'ymax').text = row[7]
        old_filename = filename
    f.close()


# function to convert BoundingBoxesOnImage object into DataFrame
def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs


def image_aug(df, IMAGES_PATH, AUG_IMAGES_PATH, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                              )
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():
        #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        #   read the image
        image = imageio.imread(IMAGES_PATH + filename)
        #   get bounding boxes coordinates and write into array
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        #   disregard bounding boxes which have fallen out of image pane
        bbs_aug = bbs_aug.remove_out_of_image()
        #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()

        #   don't perform any actions with the image if there are no bounding boxes left in it
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass

        #   otherwise continue
        else:
            #   write augmented image to a file
            imageio.imwrite(AUG_IMAGES_PATH + image_prefix + filename, image_aug)
            #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
            #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix + x)
            #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
            #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

            # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


if __name__ == '__main__':

    # IMAGES_PATH = './images/'
    # CSV_PATH = IMAGES_PATH + 'xml_objects.csv'
    # RESIZED_IMAGES_PATH = 'resized_images/'
    # LABELS_PATH = 'labels/'
    # folder = 'images_train_resized_annotation'

    if resize_images:
        get_resized_images()

    xml_df = xml_to_csv('labels_original/')

    for i in range(10):
        augmented_images_df = image_aug(df=xml_df,
                                        IMAGES_PATH='resized_images/',
                                        AUG_IMAGES_PATH='resized_images/',
                                        image_prefix='aug{}_'.format(i),
                                        augmentor=aug)
        augmented_images_df.to_csv('aug{}_images.csv'.format(i), index=False)
        csv_to_xml(CSV_PATH='aug{}_images.csv'.format(i),
                   RESIZED_IMAGES_PATH='resized_images/',
                   LABELS_PATH='labels_original/',
                   folder='resized_images')
        os.remove('aug{}_images.csv'.format(i))
