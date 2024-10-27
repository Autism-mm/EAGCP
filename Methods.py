import numpy as np
import torch

def Mh_tch_anchor(map_X,unmap_X):
    '''Manhattan matrix'''
    #不开根号比较好
    z = map_X.unsqueeze(1) - unmap_X.unsqueeze(0)
    print(z)
    z = torch.abs(z)
    pair_dist = torch.sum(z, 2, False)
    print(pair_dist)
    return pair_dist

def numpy_kernel2(x1, x2, h):
    m, n = x1.shape[0], x2.shape[0]
    print(m, n)  # 获取行数
    dist_matrix = np.zeros((m, n), dtype=float)  # 全零核矩阵
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)  # 向量差的平方和
    print(dist_matrix)
    # return np.exp(-0.5/(h**2)*dist_matrix)
    return dist_matrix


def numpy_kernel(x1, y2, h):
    '''The denominator is euclidean_dist'''
    m,n = x1.shape[0], y2.shape[0]
    x_temp=np.sum(np.multiply(x1,x1),axis=1)
    xx = x_temp.reshape(len(x_temp), 1)
    xxx = np.tile(xx, n)
    y_temp = np.sum(np.multiply(y2, y2), axis=1)
    yy = y_temp.reshape(len(y_temp), 1)
    yyy = np.tile(yy, m).T
    dist=xxx+yyy
    m=np.dot(x1,y2.T)
    dist_matrix=dist-2*m
    return np.exp(-0.5/(h**2)*dist_matrix)

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(x, y.t(),beta=1, alpha=-2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

if __name__=="__main__":
    a=torch.tensor([[1,2,3],[2,3,4],[2,1,0]])
    b = torch.tensor([[1, 2, 0],[2, 0, 4]])
    a1 = np.array([[1, 2, 3], [2, 3, 4], [2, 1, 0]])
    b2= np.array([[1, 2, 0], [2, 0, 4]])
    # euclidean_dist(a,b)
    # Anchor_S(a,b)
