import numpy as np
import math
from alignment import *
import torch
def find_nanchor(A,B,a):
    W=cosineSimilartydis(A, B)#表示距离
    n = math.ceil(W.shape[0]/a)
    # print(n)
    # 复制矩阵A以避免修改原始矩阵
    modified_matrix_A = W.clone()
    # print(torch.max(W))
    # 遍历每列，将最小的n个数置为0
    for col in range(modified_matrix_A.shape[1]):
        min_indices = np.argpartition(modified_matrix_A[:, col], n)[:n]
        modified_matrix_A[min_indices, col] = 0
    # for col in range(modified_matrix_A.shape[1]):
    #     max_indices = np.argpartition(-modified_matrix_A[:, col], n)[:n]
    #     modified_matrix_A[max_indices, col] = 1
    # modified_matrix_A = torch.where(modified_matrix_A != 0, torch.tensor(1.0), torch.tensor(0.0))
    # print(type(modified_matrix_A))
    return modified_matrix_A