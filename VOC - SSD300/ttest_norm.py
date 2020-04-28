import numpy as np

X = [1, -5, 4, 0]
# Norm chuẩn hóa để tính ra  một đại lượng biểu diễn độ lớn của một vector
# L0norm: Số lượng các phần tử khác 0
l0 = np.linalg.norm(X, ord=0)
print(l0)
# L1norm Khoảng cách Mahattan
l1 = np.linalg.norm(X, ord=1)
print(l1)
# L2norm Khoảng cách eclulid sqrt(a**2 + b**2)
l2 = np.linalg.norm(X, ord=2)
print(l2)