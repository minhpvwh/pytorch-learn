from lib import *

class ImageTransform():
    def __init__(self, resize, mean, std):
        # có thể xử lý vs tập tập train khác và val khác
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                # scale về nửa bức ảnh đến giữ nguyên sau để resize
                transforms.RandomHorizontalFlip(0.7),  # xác suất 70% xoay theo chiều ngang
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),  # *
                transforms.Normalize(mean, std)  # *
            ])
        }

    def __call__(self, img, phase='train'):  # phase là chúng ta muốn transform trên gì?
        return self.data_transform[phase](img)