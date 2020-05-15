"""
Tutorial: https://medium.com/@a.karazhay/guide-augment-images-and-multiple-bounding-boxes-for-deep-learning-in-4-steps-with-the-notebook-9b263e414dac
Notebook: "Tutorial-Image-and-Multiple-Bounding-Boxes-Augmentation-for-Deep-Learning-in-4-Steps.ipynb"
"""

import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil


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


def resize_imgaug(df, images_path, aug_images_path, image_prefix):

    # to resize the images we create two augmenters
    # one is used when the image height is more than 600px and the other when the width is more than 600px
    height_resize = iaa.Sequential([
        iaa.Resize({"height": 600, "width": 'keep-aspect-ratio'})
    ])
    width_resize = iaa.Sequential([
        iaa.Resize({"height": 'keep-aspect-ratio', "width": 600})
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
        #   AND greater than 600px perform resizing augmentation shrinking image height to 600px.
        if group_df['height'].unique()[0] >= group_df['width'].unique()[0] and group_df['height'].unique()[0] > 600:
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
        #   AND greater than 600px perform resizing augmentation shrinking image width to 600px
        elif group_df['width'].unique()[0] > group_df['height'].unique()[0] and group_df['width'].unique()[0] > 600:
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

        #     append image info without any changes if it's height and width are both less than 600px
        else:
            aug_bbs_xy = pd.concat([aug_bbs_xy, group_df])
    # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)

    return aug_bbs_xy


def prepare_images():

    # apply the function to convert all XML files in images/ folder into labels.csv
    labels_df = xml_to_csv('labels_original/')
    labels_df.to_csv('labels.csv', index=False)
    print('Successfully converted xml to csv.')

    # Copy images that were labelled
    files = glob.glob('labels_original/' + '/*.xml')
    files = [re.findall(r'\d+', x)[0] for x in files]
    files = ['data/' + 'frame' + str(x) + '.jpg' for x in files]
    for file in files:
        shutil.copy(file, 'images/')  # file, destination

    # apply resizing augmentation to our images and write the updated images and bounding boxes annotations to the
    # DataFrame. we will not apply prefix to our files and will overwrite images in the same directory
    resized_images_df = resize_imgaug(labels_df, 'images/', 'images/', '')  # source & aug folders same = overwrite
    resized_images_df.to_csv('resized_images.csv', index=False)

    # visualise the resized valentin-petkov-loL9nnBK-fE-unsplash.jpg image with bounding boxes
    # to make sure our bounding boxes were resized correctly as well
    grouped = resized_images_df.groupby('filename')
    group_df = grouped.get_group('frame100.jpg')
    group_df = group_df.reset_index()
    group_df = group_df.drop(['index'], axis=1)
    bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    image = imageio.imread('images/frame100.jpg')
    bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    ia.imshow(bbs.draw_on_image(image, size=2))

    return labels_df, resized_images_df


def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
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
                       resized_images_df: pd.DataFrame,
                       augmented_images_df: pd.DataFrame):

    grouped_resized = resized_images_df.groupby('filename')
    grouped_augmented = augmented_images_df.groupby('filename')

    group_r_df = grouped_resized.get_group(filename)
    group_r_df = group_r_df.reset_index()
    group_r_df = group_r_df.drop(['index'], axis=1)
    bb_r_array = group_r_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    resized_img = imageio.imread('images/' + filename)
    bbs_r = BoundingBoxesOnImage.from_xyxy_array(bb_r_array, shape=resized_img.shape)

    group_a_df = grouped_augmented.get_group('aug1_' + filename)
    group_a_df = group_a_df.reset_index()
    group_a_df = group_a_df.drop(['index'], axis=1)
    bb_a_array = group_a_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    augmented_img = imageio.imread('aug_images/' + 'aug1_' + filename)
    bbs_a = BoundingBoxesOnImage.from_xyxy_array(bb_a_array, shape=augmented_img.shape)

    ia.imshow(np.hstack([
        bbs_r.draw_on_image(resized_img, size=2),
        bbs_a.draw_on_image(augmented_img, size=2)
    ]))

    # Examine random example
    filename = resized_images_df['filename'].unique()[
        np.random.randint(0, len(resized_images_df['filename'].unique()), 1)][0]
    examine_aug_images(filename, resized_images_df, augmented_images_df)


def augment_images(use_prepare_images: bool = False):
    """

    :param use_prepare_images:
    :return:
    """

    # apply the function to convert all XML files in images/ folder into labels.csv
    if use_prepare_images:
        labels_df, resized_images_df = prepare_images()
    else:
        labels_df = pd.read_csv('labels.csv')
        resized_images_df = pd.read_csv('resized_images.csv')

    # augment images

    # This setup of augmentation parameters will pick two of four given augmenters and apply them in random order
    aug = iaa.SomeOf(2, [
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.Affine(rotate=(-60, 60)),
        iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
        iaa.Fliplr(1),
        iaa.Multiply((0.5, 1.5)),
        iaa.GaussianBlur(sigma=(1.0, 3.0)),
        iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
    ])
    # Apply augmentation to our images and save files into 'aug_images/' folder with 'aug1_' prefix.
    # Write the updated images and bounding boxes annotations to the augmented_images_df dataframe.
    augmented_images_df = image_aug(resized_images_df, 'images/', 'aug_images/', 'aug1_', aug)

    # Concat resized_images_df and augmented_images_df together and save in a new all_labels.csv file
    all_labels_df = pd.concat([resized_images_df, augmented_images_df])
    all_labels_df.to_csv('all_labels.csv', index=False)

    # Lastly we can copy all our augmented images in the same folder as original resized images
    for file in os.listdir('aug_images'):
        shutil.copy('aug_images/' + file, 'images/' + file)


def main():
    augment_images(use_prepare_images=False)

    return 0


if __name__ == '__main__':
    main()


