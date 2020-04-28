import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import glob

import urllib.request
import zipfile
import tarfile
import os
import cv2

# thư mục chứa data
data_dir = "data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url_voc = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_path = os.path.join(data_dir, "VOC-db.tar")  # đặt tên folder khi tải về

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url_voc, target_path)

    tar = tarfile.TarFile(target_path)
    tar.extractall(data_dir)  # giải nén tất cả vào thư mục này
    tar.close  # mở file ra thì phải đóng

# thư mục chứa các weight
weight_dir = "weight"
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
