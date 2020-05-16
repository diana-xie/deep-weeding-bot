"""
Based off https://gist.github.com/calisir/568190a5e55a79e08be318c285688457
"""
import pandas as pd
import xml.etree.ElementTree as ET

# get data
data = pd.read_csv('all_labels.csv')
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
    tree.write('labels_csv_to_xml/' + filename.replace('.jpg', '.xml'), encoding='utf8')