# ✅ Step-by-Step: Train YOLOv8 on Your Heron Dataset
Below is a **complete, end-to-end, copy-pasteable guide** assuming:

I’ll assume:

* 1 class: `heron`
* Python ≥ 3.9

## 1️⃣ Clone your dataset repo

```bash
git clone https://github.com/mbookham7/heron-data.git
cd heron-data
```

Your repo should look roughly like:

```
heron-data/
  images/
    image1.jpg
    image2.jpg
  annotations/
    image1.xml
    image2.xml
```

(If the structure is slightly different, that’s fine.)

---

## 2️⃣ Create a Python virtual environment (recommended)

```bash
python3 -m venv yolo-env
source yolo-env/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

## 3️⃣ Install YOLOv8 (Ultralytics)

```bash
pip install ultralytics
```

Verify install:

```bash
yolo version
```

---

## 4️⃣ Convert Pascal VOC XML → YOLO format

YOLOv8 **does not train directly on XML** — it needs YOLO `.txt` labels.

### Install conversion dependencies

```bash
pip install lxml tqdm
```

### Create conversion script

Create a file called `voc_to_yolo.py`:

```
cd heron_project
vi python voc_to_yolo.py
```

Paste the code below into the file, save and close

```python
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
```

### Run conversion

```bash
python voc_to_yolo.py
```

You should now have:

```
labels/
  image1.txt
  image2.txt
```

---

## 5️⃣ Split into train / validation sets

Create this structure:

```bash
mkdir -p dataset/train/images dataset/train/labels
mkdir -p dataset/val/images dataset/val/labels
```

### Simple 80/20 split

```bash
ls images | sort -R | awk '
NR % 5 == 0 {print > "val.txt"; next}
{print > "train.txt"}
'
```

### Move files

```bash
while read f; do
  mv images/$f dataset/train/images/
  mv labels/${f%.jpg}.txt dataset/train/labels/
done < train.txt

while read f; do
  mv images/$f dataset/val/images/
  mv labels/${f%.jpg}.txt dataset/val/labels/
done < val.txt
```

---

## 6️⃣ Create YOLO dataset config (`data.yaml`)

Create `data.yaml` in the project root:

```yaml
path: dataset
train: train/images
val: val/images

nc: 1
names: ["heron"]
```

---

## 7️⃣ Train YOLOv8 🚀

Start with the nano model (fast, good baseline):

```bash
yolo train \
  model=yolov8n.pt \
  data=data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16
```

💡 On Apple Silicon, YOLOv8 automatically uses **Metal (MPS)**.

---

## 8️⃣ Monitor training results

After training finishes:

```bash
ls ../runs/detect/train/
```

Important files:

* `weights/best.pt` ← your fine-tuned heron model
* `results.png` ← loss & mAP curves
* `confusion_matrix.png`

---

## 9️⃣ Test your trained model

```bash
yolo predict \
  model=runs/detect/train/weights/best.pt \
  source=heron_project/dataset/val/images \
  save=True
```

Predictions saved to:

```
runs/detect/predict/
```

---

## 🔟 Use the model in Python

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("test_image.jpg", conf=0.25)
results[0].show()
```

---

## ✅ Tips to improve heron detection

* Add **negative images** (no herons)
* Increase epochs to 100+
* Try a larger model:

  ```bash
  model=yolov8s.pt
  ```
* Use augmentation:

  ```bash
  yolo train ... augment=True
  ```

---

## 🧠 What you’ve achieved

✔ Converted VOC → YOLO
✔ Fine-tuned a pretrained detector
✔ Created a reusable heron detection model
