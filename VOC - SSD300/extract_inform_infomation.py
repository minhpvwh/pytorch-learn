import xml.etree.ElementTree as ET
import numpy as np
from make_datapath import make_datapath_list
import cv2


class AnnoXml(object):
    def __init__(self, classes):  # classes chứa 20 class của VOC
        self.classes = classes

    def __call__(self, xml_path, width, height):
        ret = list()  # chứa các annotation của ảnh
        print(xml_path)
        # xml_path = "data/VOCdevkit/VOC2012/JPEGImages/2008_000003.jpg"

        xml = ET.parse(xml_path).getroot()  # read file xml
        print(xml.iter('object'))
        for obj in xml.iter('object'):
            difficult = int(obj.find("difficult").text)  # tìm thẻ difficult trong xml
            if difficult == 1:
                continue

            bndbox = []  # chứa các thông tin của bb
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1  # -1 vì bộ data này bắt đầu từ (1,1) not (0,0)
                if pt in ["xmin", "xmax"]:
                    pixel /= width  # tỷ lệ của chiều ngang
                else:
                    pixel /= height  # tỷ lệ của chiều dọc

                bndbox.append(pixel)

            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += [bndbox]
        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_id], ....]


if __name__ == "__main__":
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    anno_xml = AnnoXml(classes)

    root_path = "data/VOCdevkit/VOC2012"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    idx = 1
    img_file_path = val_img_list[idx]

    img = cv2.imread(img_file_path)  # (height, width, 3 channel(RGB))

    h, w, c = img.shape

    anno_inform = anno_xml(val_annotation_list[idx], w, h)

    print(anno_inform)
