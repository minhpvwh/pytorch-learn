from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model
from dataset import Mydataset

train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

# dataset
train_dataset = Mydataset(train_list, transform_func=ImageTransform(resize, mean, std), phase="train")
val_dataset = Mydataset(val_list, transform_func=ImageTransform(resize, mean, std), phase="val")

# dataloader
batch_size = 2
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# loss
criterior = nn.CrossEntropyLoss()

# optimizer
params_to_update = []  # những phần tử cần update
up_params_name = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in up_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

train_model(net, dataloader_dict, criterior, optimizer, num_epoch)
