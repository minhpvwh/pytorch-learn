import os.path as osp
import glob
import torch
from tqdm import tqdm


def make_datapath_list(phase="train"):
    root_path = "hymenoptera_data/"
    target_path = osp.join(root_path + phase + "/**/*.jpg")

    path_list = list()

    # glob sẽ lấy ra tất cả các link có định dạng như target_path
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad()  # gán tất cả các grad = 0 vì mỗi lần qua một epoch ms thì phải để nó học lại k thì nó vẫn sẽ giữ grad của epoch trước

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    # outputs gồm 1 ma trận có số hàng là sl batch size
                    # số cột là số class
                    # hàm max để lấy giá trị lớn nhất trong hàng đó úng vs cột nào
                    _, preds = torch.max(outputs, 1)  # 1 là axis lấy max của từng hàng, preds là index của cột

                    # tính backward của hàm loss để update parameter
                    if phase == "train":
                        loss.backward()
                        optimizer.step()  # update parameter cho optimizer

                    # nó dạng tensor nên muốn lấy value ra thì phải dùng hàm item()
                    epoch_loss += loss.item() * inputs.size(0)  # Tensor (batch_size, channels, height, width)

                    epoch_corrects += torch.sum(preds == labels.data)  # cái dự đoán đúng

            # len(dataloader_dict[phase].dataset) số batch_size trong epoch
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Accuracy: {:.4f}".format(phase, epoch_loss, epoch_accuracy))
