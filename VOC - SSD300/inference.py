from model import SSD
from transform import Transform
import torch
import cv2
import matplotlib.pyplot as plt

cfg = {
    "num_classes": 21,  # VOC data include 20 class + 1 background class
    "input_size": 300,  # SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],  # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264],  # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315],  # Size of default box
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
}

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = SSD(phase="inference", cfg=cfg)
net_weights = torch.load('weight/ssd300_100.pth', map_location={"cuda:0": "cpu"})
net.load_state_dict(net_weights)


def show_predict(img_file_path):
    img = cv2.imread(img_file_path)

    color_mean = (104, 117, 123)
    input_size = 300
    transform = Transform(input_size, color_mean)
    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, "", "")
    img_tensor = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

    net.eval()
    input = img_tensor.unsqueeze(0)  # (1, 3, 300, 300) -  1 ảnh 3 channels 300x300
    output = net(input)
    plt.figure(figsize=[10, 10])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data  # (1, 21 ,200, 5) 5: score, cx, cy, w, h
    # đưa cái ảnh về dạng ban đầu k phải 300x300
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        # class i , bb thứ j, 0 là thể hiện cho score đang đứng đầu
        # 0.6 tự chọn, lấy lớn quá thì dectect thiếu, nhỏ quá thì ra hết nhưng mà có khả năng detect nhầm
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            # point
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(img,
                          (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), colors[i % 3], 2
                          )
            display_text = "%s: %.2f" % (classes[i - 1], score)
            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            j += 1
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_file_path = "image_test/cowboy.jpg"
    show_predict(img_file_path)
