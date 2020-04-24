from lib import *


class Mydataset(data.Dataset):
    def __init__(self, file_list, transform_func=None, phase="train"):
        self.file_list = file_list
        self.transform_func = transform_func  # đây là hàm
        self.phase = phase

    # hàm này trả về độ dài của dataset
    # có bn ảnh chẳng hạn
    def __len__(self):
        return len(self.file_list)

    # hàm này nó trả cho mình cái ảnh đầu ra truyền vào trong network của mình
    # idx nó là cái ảnh thứ bn trong file list
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transed = self.transform_func(img, self.phase)

        if self.phase in ["train", "val"]:
            label = img_path.split("/")[-2]

        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transed, label
