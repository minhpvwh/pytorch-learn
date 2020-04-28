import torch.nn as nn
import torch


class L2Norm(nn.Module):
    # mỗi channel sẽ nhân vs các hệ số khác nhau để bất kỳ như scale = 20
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        # dùng cái tensor để khởi tạo parameter cho thằng weight
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()

        self.eps = 1e-10  # là 1 con số rất nhỏ để khi chia tránh chia cho 0

    def reset_parameters(self):
        # tất cả những cái thông số scale vào toàn bộ weight
        nn.init.constant_(self.weight, self.scale)

    # x là giá trị từ net đến L2
    def forward(self, x):
        # L2
        # x.size() = (batch_size, channels, h, w) <=> (dim0, dim1, dim2, dim3)
        # dim = 1 là tính theo chiều dọc
        # để giữ nguyên các dim mà mình k tính  như dim0, 2 3 thì thêm thông số keepdim=True
        # vì phần tử ban đầu sẽ chia cho l2norm để ra con số đầu ra nên phải cộng thêm eps cho TH lỡ l2norm = 0
        l2norm = x.pow(2).sum(dim=1,
                              keepdim=True).sqrt() + self.eps  # bình phương tất cả các số trong x sau đó tính tổng
        x = torch.div(x, l2norm)
        # weight.size = (512) -> (1,512, 1,1)
        # expand_as bắt trước size
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights * x
