import torch
import numpy as np
from Methods import *
def mapins(X,Y,SX,SY,unmaplabel_X, unmaplabel_Y,device):
    '''Hungarian algorithm aligns the unaligned part'''
    #C=Hyperbole_dist(X,Y,device)
    #C=euclidean_dist(X,Y)
    C=cosineSimilartydis(X,Y)
    unmap_num=len(unmaplabel_X)
    align_out0=[]
    align_out1 = []
    align_out2=[]
    align_out3=[]
    align_labels = torch.zeros(unmap_num)
    for i in range(unmap_num):
        idx = torch.argsort(C[i, :])
        C[:, idx[0]] = float("inf")
        align_out0.append((X[i, :].cpu()).numpy())
        align_out2.append((SX[i, :].cpu()).numpy())
        align_out1.append((Y[idx[0], :].cpu()).numpy())
        align_out3.append((SY[idx[0], :].cpu()).numpy())
        if unmaplabel_X[i] == unmaplabel_Y[idx[0]]:
            align_labels[i] = 1
    count = torch.sum(align_labels)
    inference_acc = count.item() / unmap_num
    print(inference_acc)
    return C,align_out0,align_out1,align_out2,align_out3,inference_acc
def MD_dist(x,y):
    '''
    Manhattan distance
    Similarity matrix
    '''
    z = x.unsqueeze(1) - y.unsqueeze(0)  # p[3, 2, 4]
    z = torch.abs(z)
    pair_dist = torch.sum(z, 2, False)
    return pair_dist
def Hyperbole_dist(x,y,device):
    md_dist= MD_dist(x, y)

    m, n = x.size(0), y.size(0)
    unit=np.ones((m,n))
    unit_matrix=torch.tensor(unit,dtype=torch.float32)
    unit_matrix=unit_matrix.to(device)

    dist=np.zeros((m,n))
    dist=torch.tensor(dist,dtype=torch.float32)
    #print(dist)
    dist=dist.to(device)
    dist.addmm_(x, y.t(), alpha=1)
    dist_l=torch.abs(unit_matrix-dist)
    hyperbole_r=torch.div(md_dist,dist_l)
    hyperbole_r=hyperbole_r.to(device)

    ones=np.ones((m,n))
    ones=torch.tensor(ones,dtype=torch.float32)
    ones=ones.to(device)
    sum=torch.add(ones,hyperbole_r)
    sub=torch.sub(ones,hyperbole_r)
    dis_div=torch.div(sum,sub)
    #dis_div = torch.abs(dis_div)
    hyperbole_dist=torch.log10(dis_div)
    return hyperbole_dist

def cosineSimilartydis(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)

    W=torch.mm(A,B.t())
    max_values, _ = torch.max(W, axis=0)
    min_values, _ = torch.min(W, axis=0)
    normalized_matrix = (W - min_values) / (max_values - min_values)
    return 1-normalized_matrix

def cosineSimilarty(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    # A2 = A / (torch.norm(A, dim=0, p=2, keepdim=True) + 0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)
    # B2 = B / (torch.norm(B, dim=0, p=2, keepdim=True) + 0.000001)


    W=torch.mm(A,B.t())
    max_values,_ = torch.max(W, axis=0)
    min_values,_ = torch.min(W, axis=0)
    normalized_matrix = (W - min_values) / (max_values - min_values)
    return normalized_matrix
def cosineSimilartyn(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    # A2 = A / (torch.norm(A, dim=0, p=2, keepdim=True) + 0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)
    # B2 = B / (torch.norm(B, dim=0, p=2, keepdim=True) + 0.000001)


    W=torch.mm(A,B.t())

    return W
def find_k_nearst_AnchorCos(Dist,k):
    '''K nearest neighbor view for each view'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    Dist=Dist.float()
    Dist = Dist.to(device)

    sigema=1
    nSmp=Dist.shape[0]
    # val, idx = torch.sort(Dist,descending=True)
    # val = torch.reshape(val[:, k-1], shape=(nSmp, 1))
    # Dist = torch.where(Dist >= val, torch.cuda.FloatTensor([1.0]), Dist).float()
    # Dist = torch.where(Dist == torch.cuda.FloatTensor([1.0]), Dist,torch.cuda.FloatTensor([0.0])).float()
    # Dist = torch.where(Dist == torch.cuda.FloatTensor([1.0]), Dist.t(), Dist).float()
    val, idx = torch.sort(Dist)
    val = torch.reshape(val[:, k - 1], shape=(nSmp, 1))
    Dist = torch.where(Dist > val, torch.cuda.FloatTensor([0.0]), Dist).float()
    Dist = torch.where(Dist == torch.cuda.FloatTensor([0.0]), Dist.t(), Dist).float()
    Dist = torch.where(Dist == torch.cuda.FloatTensor([0.0]), torch.cuda.FloatTensor([0.0]), torch.cuda.FloatTensor([1.0])).float()

    num_nodes = Dist.size(0)
    # 创建单位阵
    identity_matrix = torch.eye(num_nodes, device='cuda')
    # 邻接矩阵加上单位阵
    Dist = Dist + identity_matrix
    # print(Dist, 'kk')
    Dist=Dist.to(device)
    return Dist