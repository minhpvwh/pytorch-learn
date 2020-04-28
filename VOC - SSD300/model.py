import torch.nn as nn
import torch
from l2_norm import L2Norm
from default_box import DefaultBox
from torch.autograd import Function
import torch.nn.functional as F


# print mạng VGG16 được lấy từ torchvision ra thì sẽ thấy tất cả mode đều là False.
# Còn tác giả SSD thì họ setting lại một số mode=True, mục đích là để keep được thông tin của edge (cạnh) của feature map.
# Nếu em để tất cả ceil_mode=False thì lúc lấy maxpooling sẽ bị mất thông tin của edge.
# Em tham khảo đoạn code này nhé. Chạy nó sẽ thấy kết quả khác nhau giữa 2 mode.
# x = torch.tensor([[-2, 1, 2, 6, 4], [-3, 1, 7, 2, -2], [-4, 2, 3, -1 , -3], [-7, 1, 2, 3, 11], [5, -7, 8, 12, -9]]).float()
# x = x.unsqueeze(0)
# y_1 = nn.MaxPool2d(kernel_size=2,stride=2, padding=0)
# y_2 = nn.MaxPool2d(kernel_size=2,stride=2, padding=0, ceil_mode=True)
# print(y_1(x))
# print(y_2(x))

def create_vgg():
    layers = []
    in_channels = 3

    cfgs = [64, 64, 'M', 128, 128, 'M',
            256, 256, 256, 'MC', 512, 512, 512, 'M',
            512, 512, 512]

    for cfg in cfgs:
        if cfg == 'M':  # # floor làm tròn lên
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC':  # ceiling làm tròn xuống
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)

            layers += [conv2d, nn.ReLU(inplace=True)]  # inplace=True để k lưu giá trị input để giảm memory
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)


def create_extras():
    layers = []
    in_channels = 1024
    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)


def create_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []

    # source1
    # loc
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]

    # source2
    # loc
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]

    # source3
    # loc
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # source4
    # loc
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # source5
    # loc
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # source6
    # loc
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


cfg = {
    "num_classes": 21,  # VOC data include 20 class + 1 background class
    "input_size": 300,  # SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],  # size của feature map như 38x38, 19x19,...
    "steps": [8, 16, 32, 64, 100, 300],  # các độ lớn của default box
    "min_size": [30, 60, 111, 162, 213, 264],  # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315],  # Size of default box
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # ứng vs feature map có thể có 4 or 6 khung hình
}


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg["num_classes"]

        # create main modules
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])
        self.L2Norm = L2Norm()

        # create default box
        dbox = DefaultBox(cfg)
        self.dbox_list = dbox.create_default_box()

        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        sources = list()  # chứa source 1 -> 6
        loc = list()  # loc = []  # chúa thông tin bb
        conf = list()  # conf = []  # chứa confident của bb

        # k là lớp thứ k trong mạng
        for k in range(23):
            x = self.vgg[k](x)

        # source1
        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        # source2
        sources.append(x)

        # source3~6
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            # aspect_ratio_num = 4 or 6
            # (batch_size, 4*aspect_ratio_num, featuremap_h, featuremap_w) ==> (batch_size, featuremap_h, featuremap_w, 4*aspect_ratio_num)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # .contiguous() để lưu các phần tử liên tục trên memory
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # (batch_size, 34928) 4*8732
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)  # (batch_size, 8732*21)

        loc = loc.view(loc.size(0), -1, 4)  # (batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)  # (batch_size, 8732, 21)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])
        else:
            return output


def decode(loc, defbox_list):
    """
    :param loc: [8732, 4] (delta_x, delta_x, delta_w, delta_h)
   :param default_box_list: [8732, 4] (cx_d, cy_d, w_d, h_d)
   :return: boxes [xmin, ymin, xmax, ymax]
    """

    # dim = 1 là hàng ngang xếp vào nhau, dim = 0 là xếp theo hàng dọc
    # X = torch.rand([3, 4])
    # Y = torch.rand([3, 4])
    # w, h = X.shape
    # print(w, h)
    #
    # dim_0 = torch.cat([X, Y], dim=0)
    # print(dim_0)
    # dim_1 = torch.cat([X, Y], dim=1)
    # print(dim_1)
    # tensor([[0.2647, 0.5893, 0.8238, 0.5097],
    #         [0.6052, 0.6083, 0.9908, 0.6694],
    #         [0.0106, 0.6093, 0.5622, 0.7940],
    #         [0.1573, 0.1139, 0.1755, 0.1955],
    #         [0.1253, 0.2754, 0.5962, 0.9850],
    #         [0.0670, 0.0303, 0.1126, 0.5473]])
    # tensor([[0.2647, 0.5893, 0.8238, 0.5097, 0.1573, 0.1139, 0.1755, 0.1955],
    #         [0.6052, 0.6083, 0.9908, 0.6694, 0.1253, 0.2754, 0.5962, 0.9850],
    #         [0.0106, 0.6093, 0.5622, 0.7940, 0.0670, 0.0303, 0.1126, 0.5473]])

    boxes = torch.cat((
        defbox_list[:, :2] + 0.1 * loc[:, :2] * defbox_list[:, 2:],
        defbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)

    boxes[:, :2] -= boxes[:, 2:] / 2  # calculate xmin, ymin
    boxes[:, 2:] += boxes[:, :2]  # calculate xmax, ymax

    return boxes


# scores là confident của bb
def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    :param boxes: [num_boxes, 4] # 4 là xmin, ymin, xmax, ymax
    :param scores: [num_boxes] # 1 cột chứa các xác suất của classes
    :param overlap:
    :param top_k:
    :return:
    """
    count = 0
    # giữ các thông tin mình cần bb nào, id bằng bn, k thì loại
    keep = scores.new(scores.size(0)).zero_().long()  # tạo ra 1 Tensor có định dạng giống scores

    # tạo các biến để lưu thông tin của bb - coordinate
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # diện tích bb - area of boxes
    # x2-x1 chiều dài
    # y2-y1 chiều cao
    area = torch.mul(x2 - x1, y2 - y1)

    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(0)  # thằng có xác suất cao nhất thì xuống dưới cùng
    idx = idx[-top_k:]  # id của top 200 boxes có độ tự tin cao nhất

    # numel() là nếu idx còn tồn tại các phần tử
    while idx.numel() > 0:
        i = idx[-1]  # id của box có độ tự tin cao nhất
        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]  # id của boxes ngoại trừ box có độ tự tin cao nhất
        # information boxes
        # lấy ra các thằng index có giá trị trong khoảng idx này từ trong x1 vs dim = 0
        torch.index_select(x1, 0, idx, out=tmp_x1)  # x1
        torch.index_select(y1, 0, idx, out=tmp_y1)  # y1
        torch.index_select(x2, 0, idx, out=tmp_x2)  # x2
        torch.index_select(y2, 0, idx, out=tmp_y2)  # y2

        # Y = torch.tensor([[0.8000, 0.5587, 0.0972],
        #                   [0.8000, 0.3000, 0.6500]])
        #
        # print(torch.clamp(Y, min=0.5))
        # tensor([[0.8000, 0.5587, 0.5000],
        #         [0.8000, 0.5000, 0.6500]])

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])  # =x1[i] if tmp_x1 < x1[1]
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])  # =y2[i] if tmp_y2 > y2[i]

        # chuyển về tensor cái size mà index được giảm đi 1 vì chúng ta đang tính của bb còn lại, bỏ thằng chính đi r
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # overlap area
        inter = tmp_w * tmp_h
        others_area = torch.index_select(area, 0, idx)  # diện tích của mỗi bbox
        union = area[i] + others_area - inter
        iou = inter / union
        idx = idx[iou.le(overlap)]  # chỉ giữ lại nhũng bb có độ trùng diện tích bé hơn 0.45
    # count sl các bb có thể giữ lại dc
    return keep, count


class Detect(Function):
    # lấy nhũng confident nào > 0.01 (1%)
    # overlap nào lớn hơn 0.45 sẽ bỏ nó đi
    def __init__(self, conf_thresh=0.01, top_k=200, nsm_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nsm_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        num_batch = loc_data.size(0)  # batch_size (2,4,6,...32, 64, 128) - sl ảnh
        num_dbox = loc_data.size(1)  # 8732
        num_classe = conf_data.size(2)  # 21

        conf_data = self.softmax(conf_data)  # (batch_num, 8732, num_class) -> (batch_num, num_class, num_bbox)
        conf_preds = conf_data.transpose(2, 1)

        output = torch.zeros(num_batch, num_classe, self.top_k, 5)  # 5 là xmin,..., label

        # xử lý từng bức ảnh trong một batch các bức ảnh
        for i in range(num_batch):
            # Tính bbox từ offset information và default box
            decode_boxes = decode(loc_data[i], dbox_list)

            # copy confidence score của ảnh thứ i
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classe):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # chỉ lấy những confidence > 0.01
                scores = conf_scores[cl][c_mask]
                # nếu trong scores k có phần tử nào
                # có thể dùng hàm numel()
                if scores.nelement() == 0:
                    continue

                # đưa chiều về giống chiều của decode_boxes để tính toán
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)  # (8732, 4)
                boxes = decode_boxes[l_mask].view(-1, 4)  # (số box có độ tự tin lớn hơn > 0.01, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output


if __name__ == "__main__":
    # vgg = create_vgg()
    # extras = create_extras()
    # loc = create_loc()
    # conf = create_conf()
    # print(loc)
    # print(conf)
    ssd = SSD(phase="train", cfg=cfg)
    print(ssd)
