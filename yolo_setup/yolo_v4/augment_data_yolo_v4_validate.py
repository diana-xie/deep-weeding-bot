"""
Temp file to generate validation image prep. Should eventually be done automatically.
"""

"""
Tutorials used:
- https://medium.com/@a.karazhay/guide-augment-images-and-multiple-bounding-boxes-for-deep-learning-in-4-steps-with-the-notebook-9b263e414dac
- https://gist.github.com/calisir/568190a5e55a79e08be318c285688457
- https://github.com/asetkn/Tutorial-Image-and-Multiple-Bounding-Boxes-Augmentation-for-Deep-Learning-in-4-Steps/blob/master/Tutorial-Image-and-Multiple-Bounding-Boxes-Augmentation-for-Deep-Learning-in-4-Steps.ipynb

Image resize dim originally 600 x 600.

Steps:
1. Hand-label training/validation set, using labelImg "type labelImg in Terminal"

2. Perform train-test split.

"""
import pandas as pd
import numpy as np
import re
import os
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage  # imgaug uses plt backend for displaying images
from imgaug import augmenters as iaa
import imageio  # imageio library will be used for image input/output
import glob
import xml.etree.ElementTree as ET  # this library is needed to read XML files for converting it into CSV
import shutil

ia.seed(1)


# Function that will extract column data for our CSV file as pandas DataFrame
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
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
            except:
                pass
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


# function to convert BoundingBoxesOnImage object into DataFrame
def bbs_obj_to_df(bbs_object):
    # convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
    # convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs


def resize_imgaug(df: pd.DataFrame,
                  images_path: str,
                  aug_images_path: str,
                  image_prefix: str,
                  height: int,
                  width: int = None):
    # to resize the images we create two augmenters
    # one is used when the image height is more than 416px and the other when the width is more than 416px
    if width is None:
        width = 'keep-aspect-ratio'
    height_resize = iaa.Sequential([
        iaa.Resize({"height": height, "width": width})
    ])
    width_resize = iaa.Sequential([
        iaa.Resize({"height": width, "width": height})
    ])

    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                              )
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():

        #   Get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)

        # The only difference between if and elif statements below is the use of height_resize and width_resize
        # augmentors defined previously.

        #   If image height is greater than or equal to image width
        #   AND greater than 416px perform resizing augmentation shrinking image height to 416px.
        if group_df['height'].unique()[0] >= group_df['width'].unique()[0] and group_df['height'].unique()[0] > 416:
            #   read the image
            image = imageio.imread(images_path + filename)
            #   get bounding boxes coordinates and write into array
            bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
            #   pass the array of bounding boxes coordinates to the imgaug library
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
            #   apply augmentation on image and on the bounding boxes
            image_aug, bbs_aug = height_resize(image=image, bounding_boxes=bbs)
            #   write augmented image to a file
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
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

        #   if image width is greater than image height
        #   AND greater than 416px perform resizing augmentation shrinking image width to 416px
        elif group_df['width'].unique()[0] > group_df['height'].unique()[0] and group_df['width'].unique()[0] > 416:
            #   read the image
            image = imageio.imread(images_path + filename)
            #   get bounding boxes coordinates and write into array
            bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
            #   pass the array of bounding boxes coordinates to the imgaug library
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
            #   apply augmentation on image and on the bounding boxes
            image_aug, bbs_aug = width_resize(image=image, bounding_boxes=bbs)
            #   write augmented image to a file
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
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

        #     append image info without any changes if it's height and width are both less than 416px
        else:
            aug_bbs_xy = pd.concat([aug_bbs_xy, group_df])
    # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)

    return aug_bbs_xy


def prepare_images(labels_raw_validate_path: str,
                   images_raw_path: str,
                   images_validate_path: str,
                   height: int,
                   width: int):
    """
    :param labels_raw_validate_path: path of folder containing original xml labels for images
    :param images_raw_path: path of folder containing raw images
    :param images_validate_path: path of folder containing images that were labelled for training + later augmented
    images
    :param height: height of resized image
    :param width: width of resized image
    :return:
    labels_df = dataframe of original images xml labels
    resize_images_df = dataframe of resized images xml labels
    """

    # apply the function to convert all XML files in images/ folder into labels_validate.csv
    labels_df = xml_to_csv(os.path.dirname(labels_raw_validate_path))
    labels_df.to_csv('labels_validate.csv', index=False)
    print('Successfully converted xml to csv.')

    # Copy images that were labelled
    files = glob.glob(labels_raw_validate_path + '/*.xml')
    files = [re.findall(r'\d+', x)[0] for x in files]
    files = [images_raw_path + '/frame' + str(x) + '.jpg' for x in files]
    for file in files:
        shutil.copy(file, images_validate_path)  # file, destination

    # apply resizing augmentation to our images and write the updated images and bounding boxes annotations to the
    # DataFrame. we will not apply prefix to our files and will overwrite images in the same directory
    resized_images_validate_df = resize_imgaug(df=labels_df,
                                               images_path=images_validate_path,
                                               aug_images_path=images_validate_path,  # source & aug folders same
                                               image_prefix='',
                                               height=height,
                                               width=width)
    resized_images_validate_df.to_csv('resized_images_validate.csv', index=False)

    # visualise the resized valentin-petkov-loL9nnBK-fE-unsplash.jpg image with bounding boxes
    # to make sure our bounding boxes were resized correctly as well
    grouped = resized_images_validate_df.groupby('filename')
    group_df = grouped.get_group('frame243.jpg')
    group_df = group_df.reset_index()
    group_df = group_df.drop(['index'], axis=1)
    bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    image = imageio.imread('images_validate/frame243.jpg')
    bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    ia.imshow(bbs.draw_on_image(image, size=2))

    return labels_df, resized_images_validate_df


def image_aug(df: pd.DataFrame,
              images_path: str,
              aug_images_path: str,
              image_prefix: str,
              augmentor):
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
        image = imageio.imread(images_path + filename)
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
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
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


def examine_aug_images(filename: str,
                       resized_images_validate_df: pd.DataFrame,
                       augmented_images_df: pd.DataFrame):
    grouped_resized = resized_images_validate_df.groupby('filename')
    grouped_augmented = augmented_images_df.groupby('filename')

    group_r_df = grouped_resized.get_group(filename)
    group_r_df = group_r_df.reset_index()
    group_r_df = group_r_df.drop(['index'], axis=1)
    bb_r_array = group_r_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    resized_img = imageio.imread('images_validate/' + filename)
    bbs_r = BoundingBoxesOnImage.from_xyxy_array(bb_r_array, shape=resized_img.shape)

    group_a_df = grouped_augmented.get_group('aug1_' + filename)
    group_a_df = group_a_df.reset_index()
    group_a_df = group_a_df.drop(['index'], axis=1)
    bb_a_array = group_a_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    augmented_img = imageio.imread('images_aug_validate/' + 'aug1_' + filename)
    bbs_a = BoundingBoxesOnImage.from_xyxy_array(bb_a_array, shape=augmented_img.shape)

    ia.imshow(np.hstack([
        bbs_r.draw_on_image(resized_img, size=2),
        bbs_a.draw_on_image(augmented_img, size=2)
    ]))

    # Examine random example
    filename = resized_images_validate_df['filename'].unique()[
        np.random.randint(0, len(resized_images_validate_df['filename'].unique()), 1)][0]
    examine_aug_images(filename, resized_images_validate_df, augmented_images_df)


def augment_images(use_prepare_images: bool = False,
                   height: int = 416,
                   width: int = None):
    """
    :param use_prepare_images: if True, apply the function to convert all XML files in images/ folder into
    labels_validate.csv
    :param height: height of resized image
    :param width: width of resized image
    :return: none; all outputs automatically saved in folders
    """
    # TODO: add automatic train-test-split option

    # apply the function to convert all XML files in images/ folder into labels_validate.csv
    if use_prepare_images:
        labels_df, resized_images_validate_df = prepare_images(
            labels_raw_validate_path=os.path.dirname(os.path.dirname(__file__)) + '/labels_raw_validate/',
            images_raw_path=os.path.dirname(os.path.dirname(__file__)) + '/images_raw/',
            images_validate_path='images_validate/',
            height=height,
            width=width
        )
    else:
        resized_images_validate_df = pd.read_csv('resized_images_validate.csv')

    # augment images

    # This setup of augmentation parameters will pick two of four given augmenters and apply them in random order
    aug = iaa.SomeOf(2, [
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.Affine(rotate=(-60, 60)),
        iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
        iaa.Fliplr(1),
        iaa.Multiply((0.5, 1.5)),
        iaa.GaussianBlur(sigma=(1.0, 3.0)),
        iaa.AdditiveGaussianNoise(scale=(0.03 * 255, 0.05 * 255))
    ])
    # Apply augmentation to our images and save files into 'images_aug_validate/' folder with 'aug1_' prefix.
    # Write the updated images and bounding boxes annotations to the augmented_images_df dataframe.
    augmented_images_df = image_aug(df=resized_images_validate_df,
                                    images_path='images_validate/',
                                    aug_images_path='images_aug_validate/',
                                    image_prefix='aug1_',
                                    augmentor=aug)

    # Concat resized_images_validate_df and augmented_images_df together and save in a new all_labels_validate.csv file
    all_labels_df = pd.concat([resized_images_validate_df, augmented_images_df])
    all_labels_df.to_csv('all_labels_validate.csv', index=False)

    # Lastly we can copy all our augmented images in the same folder as original resized images
    for file in os.listdir('images_aug_validate'):
        shutil.copy('images_aug_validate/' + file, 'images_validate/' + file)


# convert single csv to xml files
# Based off https://gist.github.com/calisir/568190a5e55a79e08be318c285688457
def csv_to_xml(csv_file: str,
               labels_validate_path: str):
    """
    :param csv_file: filename of csv file input
    :param labels_validate_path: path of folder where output xml's will be saved
    :return: none
    """
    # get data
    data = pd.read_csv(csv_file)
    # data['filename'] = data['filename'].apply(nameChange)  # change filename to save
    data = data[data['xmin'].notnull()]  # remove annotations with null locations

    # get unique filenames, to perform bounding box xml appending per image
    filenames = data['filename'].unique()

    for filename in filenames:

        df = data[data['filename'] == filename]

        for i in range(0, len(df)):

            height = df['height'].iloc[i]
            width = df['width'].iloc[i]
            depth = 3

            if i == 0:

                # for file
                annotation = ET.Element('annotation')
                ET.SubElement(annotation, 'folder').text = 'images'
                ET.SubElement(annotation, 'filename').text = str(df['filename'].iloc[i])
                ET.SubElement(annotation, 'segmented').text = '0'
                size = ET.SubElement(annotation, 'size')
                ET.SubElement(size, 'width').text = str(width)
                ET.SubElement(size, 'height').text = str(height)
                ET.SubElement(size, 'depth').text = str(depth)

                # for first bounding box annotation
                ob = ET.SubElement(annotation, 'object')
                ET.SubElement(ob, 'name').text = df['class'].iloc[i]
                ET.SubElement(ob, 'pose').text = 'Unspecified'
                ET.SubElement(ob, 'truncated').text = '0'
                ET.SubElement(ob, 'difficult').text = '0'
                bbox = ET.SubElement(ob, 'bndbox')
                ET.SubElement(bbox, 'xmin').text = str(df['xmin'].iloc[i])
                ET.SubElement(bbox, 'ymin').text = str(df['ymin'].iloc[i])
                ET.SubElement(bbox, 'xmax').text = str(df['xmax'].iloc[i])
                ET.SubElement(bbox, 'ymax').text = str(df['ymax'].iloc[i])

            else:

                # for bounding box annotations
                ob = ET.SubElement(annotation, 'object')
                ET.SubElement(ob, 'name').text = df['class'].iloc[i]
                ET.SubElement(ob, 'pose').text = 'Unspecified'
                ET.SubElement(ob, 'truncated').text = '0'
                ET.SubElement(ob, 'difficult').text = '0'
                bndbox = ET.SubElement(ob, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(df['xmin'].iloc[i])
                ET.SubElement(bndbox, 'ymin').text = str(df['ymin'].iloc[i])
                ET.SubElement(bndbox, 'xmax').text = str(df['xmax'].iloc[i])
                ET.SubElement(bndbox, 'ymax').text = str(df['ymax'].iloc[i])

        tree = ET.ElementTree(annotation)
        tree.write(labels_validate_path + filename.replace('.jpg', '.xml'), encoding='utf8')


def main():
    # augment images and generate xml annotation files for them
    augment_images(use_prepare_images=True,
                   height=416,
                   width=416)

    csv_to_xml(csv_file='all_labels_validate.csv',
               labels_validate_path='labels_validate/')

    return 0


if __name__ == '__main__':
    main()
