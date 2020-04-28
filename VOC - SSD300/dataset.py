import torch.utils.data as data
import cv2
import torch
import numpy as np
from make_datapath import make_datapath_list
from transform import Transform
from extract_inform_infomation import AnnoXml


class Mydataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # gt các thông tin liên quan đến nhãn của bức ảnh và thông tin liên quan đến bb
        img, gt, h, w = self.pull_item(index)
        return img, gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path)  # BGR
        h, w, c = img.shape

        # get anno infomation
        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path, w, h)

        # preprocessing
        img, boxes, labels = self.transform(img, self.phase, anno_info[:, :4], anno_info[:, -1])

        # BGR - > RGB, (h,w,c) -> (c,h,w)
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # ground truth
        gt = gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, h, w


def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])  # sample[0] =  ảnh
        targets.append(torch.FloatTensor(sample[1]))  # sample[1] = annotation

    # imgs đang dạng list phải chuyển nó về dạng Tensor
    # [3, 300 , 300] => [batch_size, 3, 300, 300]
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets


if __name__ == "__main__":
    root_path = "data/VOCdevkit/VOC2012"
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    color_mean = (104, 117, 123)
    input_size = 300
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    phase = "train"
    transform = Transform(input_size, color_mean)
    anno_xml = AnnoXml(classes)
    train_dataset = Mydataset(train_img_list, train_annotation_list, phase, transform, anno_xml)
    phase = "val"
    val_dataset = Mydataset(val_img_list, val_annotation_list, phase, transform, anno_xml)

    print(train_dataset.__getitem__(2))

    batch_size = 4  # người ta thường để các số chẵn để xử lý song song trong thread của các core GPU,CPU
    # nếu trong classifier thì chả cần gọi hàm này ra thôi k cần custom collate_fn
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    print(batch_iter)
    images, targets = next(batch_iter)
    print(images.size())  # torch.Size([4, 3, 300, 300])
    print(len(targets))  # 4 nhóm annotation
    print(targets)
    print(targets[0].size())  # torch.Size([1, 5]) có 1 obj và mỗi obj có 5 t.phan xmin, ymin, xmax, ymax, label
