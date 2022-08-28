import json
from turtle import width
import numpy as np
import os
from PIL import Image
import shutil
import csv
import cv2
import math

#path to anotations(json) file
coco_file = 'C:\\Users\\52563\\mmpose\\data\\ap10k\\annotations\\ap10k-test-split1.json'
with open(coco_file, "r") as read_content:
    data = json.load(read_content)

images = data['images']
annos = data['annotations']

#path to the whole dataset
source_fd = 'F:\\ap-60k\\zebra\\raw'

#path to the splited dataset
target_fd = 'F:\\ap-60k\\zebra\\test1'

for i in range(len(images)):
    img_name = images[i]['file_name']
    file = os.path.join(source_fd, img_name)
    dst = os.path.join(target_fd, img_name)
    if os.path.isfile(file):
        shutil.copyfile(file, dst)


