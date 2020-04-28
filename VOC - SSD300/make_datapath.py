import os.path as osp
import random
import xml.etree.cElementTree as ET
import cv2
import torch.utils.data as data
import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def make_datapath_list(root_path):
    image_path_template = osp.join(root_path, "JPEGImages", "%s.jpg")  # path chung của ảnh
    annotation_path_template = osp.join(root_path, "Annotations", "%s.xml")  # path chung của annotation

    train_id_names = osp.join(root_path, "ImageSets/Main/train.txt")  # thông tin train
    val_id_names = osp.join(root_path, "ImageSets/Main/val.txt")  # thông tin val

    train_img_list = list()
    train_annotation_list = list()
    val_img_list = list()
    val_annotation_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # xóa các ký tự xuống dòng, xóa space
        img_train_path = (image_path_template % file_id)
        anno_train_path = (annotation_path_template % file_id)

        train_img_list.append(img_train_path)
        train_annotation_list.append(anno_train_path)

    for line in open(val_id_names):
        file_id = line.strip()
        img_val_path = (image_path_template % file_id)
        anno_val_path = (annotation_path_template % file_id)

        val_img_list.append(img_val_path)
        val_annotation_list.append(anno_val_path)

    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


if __name__ == "__main__":
    root_path = "data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    print(len(train_img_list))
    print(len(train_annotation_list))
    print(len(val_img_list))
    print(len(val_annotation_list))
