import torch
import os
import PIL
import cv2
from torch import nn
from PIL import Image, ImageDraw
from modelConstruction import yolo_null

a = float('inf')
print(a)

'''
class cl:
    def __init__(self, va):
        self.val = va

    def __traceback__(self):
        file_name = self.file_name
        line = self.line_num
        module = self.module
        code = self.code

        message = f'File {file_name}, line {line}, in {module}\n {code}'
        return message


i = cl('sup')
a = i.__namai__()
print(a)

torch.cuda.empty_cache()
image = Image.open('/home/ikase/labs/RSBOT/sources/raw img/yun_0/yun_0_frame_00929.png')
imagee = image.resize((800, 800))
image = ImageDraw.Draw(imagee)
image.rectangle((306.8853, -62.4249, 456.5326, 108.6651), outline ="red")
image.rectangle((394.4760,  86.8750, 463.9260, 123.1250), outline ="red")
image.rectangle(( 96.0225, 129.1460,  98.3066, 152.7817), outline ="black")
image.rectangle((91.6740, 150.6250, 187.5150, 166.8750), outline ="black")
image.text((91.6740, 150.6250), 'glasBoden', outline ="black")
imagee.show()
exit()
image.text((537.543, 790.625), 'glasBoden', outline ="black")
image.text((13.89, 791.25), 'glasBoden', outline ="black")
image.text((408.366, 251.25), 'ich', outline ="red")
image.text((526.431, 45.625), 'flnHammer', outline ="red")
image.text((509.763, 10.625), 'flnHammer', outline ="red")
image.text((276.411, 61.25), 'hDrbHammer', outline ="red")
image.text((369.474, 16.25), "hDrbHammer", outline ="red")
image.text((316.692, 0.625), "hDrbHammer", outline ="red")
image.text((169.458, 46.25), 'flnHammer', outline ="red")
image.text((211.12800000000001, 12), 'flnHammer', outline ="red")
image.text((275.022, 63.75), 'sprungbett', outline ="red")
image.text((238.90800000000002, 109.375), 'sprungbett', outline ="red")
imagee.show()
'''

'''
except Exception as exc:
    ovl, rec_code, rec_mes = push_message(message = f'--error: {exc}--')
    print(f'\n{WARNING_TEXT} --- an error message has been sent with status code {rec_code} --- {RESET_TEXT}\n')
    total_fails += 1
    if total_fails > 4:
        push_message(message = '__failed attempts > 5, program interrupted__')
        raise SystemExit(f'\n{BOLD_TEXT}failed attempts > 5, program interrupted{RESET_TEXT}\n')
    else:
        continue
print(f'average epoch loss per batch: {epoch_loss / batchLoss}')
'''

'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, torch.cuda.get_device_name())
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
import numpy as np

labelledDirP = '/home/ikase/labs/RSBOT/sources/labelled/yun0Lab/'
labelledDir = os.listdir(labelledDirP)

t = torch.random(1, 2, 3, 4)
print(t)
def extBoxes(files, path):
    boxDims = []
    for filename in files:
        file_path = os.path.join(path, filename)
        if not filename.endswith('.xml'):  # Only process XML files
            continue
        if os.path.isfile(file_path):  # Check if it's a file
            tree = ET.parse(file_path)
            root = tree.getroot()
            for ob in root.findall('object'):  # Correct tag is 'object'
                name = ob.find('name').text
                bbox = ob.find('bndbox')  # 'bndbox' is under 'object'

                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)

                boxDims.append({
                    'name': name,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax
                })
        widths_heights = []
    for box in boxDims:
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']
        widths_heights.append([width, height])

    return widths_heights

bx = extBoxes(files = labelledDir, path = labelledDirP)
print(len(bx))

wh_np = np.array(bx)

# Perform k-means clustering to find `num_a` anchor boxes
kmeans = KMeans(n_clusters = 9)
kmeans.fit(wh_np)

# The cluster centers are your anchor box dimensions
anchor_dims = kmeans.cluster_centers_
print(anchor_dims)

aList = [[ 78.58419358,  60.08147408],
          [157.4320781,  206.24965132],
         [223.74660633, 754.67873303],
[281.65401863,  67.91617799],
[108.94087738, 128.48303181],
[168.37449615,  34.06284353],
[449.56449376, 143.01386963],
[195.55507088, 436.47655398],
[ 46.41646819,  21.15251786]]

bList = []

for i in aList:
    w, h = i[0], i[1]
    wa, ha = round(w, 3), round(h, 3)
    bList.append((wa, ha))

bList.sort(key = lambda x: x[0])
print(bList)
'''
