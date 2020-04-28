from augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
import cv2
from make_datapath import make_datapath_list
from extract_inform_infomation import AnnoXml
import matplotlib.pyplot as plt


# Compose là class để kết hợp các transform lại với nhau
# ConvertFromInts chuyển ảnh từ int sang float32
# ToAbsoluteCoords lấy thong số pixel chuẩn trên bức ảnh chưa chuẩn hóa về [0,1]
# PhotometricDistort đổi các thông số về màu sắc của nó
# Expand khi mở rộng bức ảnh ra thì sẽ chèn những cái pixel mà nó trống bị khoảng đen vào
# ToPercentCoords đưa annotation chuẩn hóa về dạng [0,1]
# SubtractMeans dùng để trừ đi các giá trị trung bình của mỗi channel - khi trừ đi có thể chuản hóa bức ảnh

# class Transform
# class Transform(object)
class Transform():
    # có nhiều k gian màu như RGB , ... có color_mean là bn
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),  # cắt 1 phần bất kỳ trong bức ảnh ra
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":
    root_path = "data/VOCdevkit/VOC2012"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)
    h, w, c = img.shape

    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    anno_trans = AnnoXml(classes)

    anno_info_list = anno_trans(val_annotation_list[0], w, h)
    print(anno_info_list)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # càn chuyển dạng vì matplot khác dạng đọc vs cv2
    plt.show()

    # prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300
    transform = Transform(input_size, color_mean)

    # transform train img
    phase = "train"
    # boxes - > (xmin, ymin, xmax, ymax)
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))  # càn chuyển dạng vì matplot khác dạng đọc vs cv2
    plt.show()

    # transform val img
    phase = "val"
    # boxes - > (xmin, ymin, xmax, ymax)
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))  # càn chuyển dạng vì matplot khác dạng đọc vs cv2
    plt.show()
