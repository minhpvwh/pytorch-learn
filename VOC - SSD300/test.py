import torch

# X = torch.rand([3, 4])
# Y = torch.rand([3, 4])
# w, h = X.shape
# print(w, h)
#
#
# dim_0 = torch.cat([X, Y], dim=0)
# print(dim_0)
# dim_1 = torch.cat([X, Y], dim=1)
# print(dim_1)

Y = torch.tensor([[0.8000, 0.5587, 0.0972],
                  [0.8000, 0.3000, 0.6500]])

print(torch.clamp(Y, min=0.5))

