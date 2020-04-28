import itertools
from math import sqrt
import torch
import pandas as pd

cfg = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # tỷ lệ khung hình source 1 -> 6
    "feature_maps": [38, 19, 10, 5, 3, 1],  # size của feature map như 38x38, 19x19,...
    "steps": [8, 16, 32, 64, 108, 300],  # các độ lớn của default box
    "min_size": [30, 60, 111, 162, 213, 264],  # size của default box
    "max_size": [60, 111, 162, 213, 264, 315],  # size của default box
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # ứng vs feature map có thể có 4 or 6 khung hình
}


class DefaultBox():
    def __init__(self, cfg):
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg["steps"]

    def create_default_box(self):
        defaultbox_list = []

        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k

                # small square box
                s_k = self.min_size[k] / self.img_size  # first case : 30/300
                defaultbox_list += [cx, cy, s_k, s_k]

                # big square box
                s_k_ = sqrt(s_k * (self.max_size[k] / self.img_size))
                defaultbox_list += [cx, cy, s_k_, s_k_]

                # rectengle box
                for ar in self.aspect_ratios[k]:
                    # có 1 phần tử tạo dc 2 default box còn 2 phần tử tạo dc 4 default box rectengle
                    defaultbox_list += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    defaultbox_list += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(defaultbox_list).view(-1, 4)
        # trong output trong quá trình tính toán có nhũng phần tử bé hơn 0 or lớn hơn 1
        output.clamp_(max=1, min=0)  # _ ở dưới là nó lưu giá trị vào chính nó ở đây là output
        return output


if __name__ == "__main__":
    default_box = DefaultBox(cfg)
    default_box_list = default_box.create_default_box()
    # print(default_box.create_default_box())

    print(pd.DataFrame(default_box_list.numpy()))
    #             cx        cy        h         w
    #
    #             0         1         2         3
    # 0     0.013333  0.013333  0.100000  0.100000
    # 1     0.013333  0.013333  0.141421  0.141421
    # 2     0.013333  0.013333  0.141421  0.070711
    # 3     0.013333  0.013333  0.070711  0.141421
    #       ............................
    # 8730  0.500000  0.500000  1.000000  0.622254
    # 8731  0.500000  0.500000  0.622254  1.000000
