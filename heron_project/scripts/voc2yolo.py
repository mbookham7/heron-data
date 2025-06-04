import os
import xml.etree.ElementTree as ET

classes = ["heron"]  # Add more classes if needed

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(image_id, xml_dir, label_output_dir):
    in_file = open(f"{xml_dir}/{image_id}.xml")
    out_file = open(f"{label_output_dir}/{image_id}.txt", 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [float(xmlbox.find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + '\n')

# Usage
xml_dir = '../xmls'
label_dir = '../labels/train'
os.makedirs(label_dir, exist_ok=True)

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        image_id = xml_file.split(".")[0]
        convert_annotation(image_id, xml_dir, label_dir)
