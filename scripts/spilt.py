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
annotations = 'D:\\deeplabcut\\dataset\\testset\\ap10test\\SynAP\\annotations_99+3000 (SynAP)\\ap10k-train-split1.json'
with open(annotations, "r") as read_content:
    data = json.load(read_content)

images = data['images']
annos = data['annotations']

#path to the whole dataset
source_fd = 'F:\\ap-10k\\ap-10k\\data'

#path to the splited dataset
target_fd = 'F:\\ap-10k\\ap-10k\\data2'

for i in range(len(images)):
    img_name = images[i]['file_name']
    file = os.path.join(source_fd, img_name)
    dst = os.path.join(target_fd, img_name)
    if os.path.isfile(file):
        shutil.copyfile(file, dst)


