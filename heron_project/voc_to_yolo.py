import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# EDIT THESE PATHS IF NEEDED
VOC_ANN_DIR = "annotations"
IMAGE_DIR = "images"
OUTPUT_LABEL_DIR = "labels"
CLASS_MAP = {"heron": 0}

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

for xml_file in tqdm(os.listdir(VOC_ANN_DIR)):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(VOC_ANN_DIR, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_file = xml_file.replace(".xml", ".txt")
    with open(os.path.join(OUTPUT_LABEL_DIR, label_file), "w") as out:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASS_MAP:
                continue

            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert((w, h), b)
            out.write(f"{CLASS_MAP[cls]} {' '.join(map(str, bb))}\n")