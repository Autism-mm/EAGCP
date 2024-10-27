import warnings

import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class GCNNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim,output_dim):
        dropout=0.001
        super(GCNNet, self).__init__()
        allGCN1=[]
        allGCN2 = []
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.dropout = dropout

        self.gcn1 = GraphConvolution(input_dim, int(input_dim/2))
        self.gcn12 = GraphConvolution(int(input_dim/2), output_dim)
        self.dgcn1=GraphConvolution(output_dim,int(input_dim/2))
        self.dgcn12 = GraphConvolution(int(input_dim/2), input_dim)
        allGCN1.append(self.gcn1)
        allGCN1.append(self.gcn12)
        allGCN1.append(self.dgcn1)
        allGCN1.append(self.dgcn12)
        self.GCN =nn.Sequential(*allGCN1)
        self.gcn2 = GraphConvolution(input_dim, int(input_dim/2))
        self.gcn22 = GraphConvolution(int(input_dim/2), output_dim)
        self.dgcn2 = GraphConvolution(output_dim, int(input_dim/2))
        self.dgcn22=GraphConvolution(int(input_dim/2),input_dim)
        allGCN2.append(self.gcn2)
        allGCN2.append(self.gcn22)
        allGCN2.append(self.dgcn2)
        allGCN2.append(self.dgcn22)
        self.GCN2 = nn.Sequential(*allGCN2)
        # self.gcn2 = GraphConvolution(500,100)
        # self.gcn3 = GraphConvolution(20, output_dim)

    def forward(self, adjacency1, feature1,adjacency2,feature2):
        h0 = F.relu(self.gcn1(adjacency1, feature1))
        p0 = F.relu(self.gcn12(adjacency1, h0))
        p1=F.relu(self.dgcn1(adjacency1, p0))
        h1=F.relu(self.dgcn12(adjacency1, p1))
        # logits = self.gcn3(adjacency1, h0)

        h3 = F.relu(self.gcn2(adjacency2, feature2))
        p3=F.relu(self.gcn22(adjacency2, h3))
        p4 = F.relu(self.dgcn2(adjacency2, p3))
        h4 = F.relu(self.dgcn22(adjacency2, p4))
        # logits2 = self.gcn3(adjacency2, h3)

        # return F.log_softmax(logits, dim=1),F.log_softmax(logits2, dim=1)
        return p0,h1,p3,h4

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：H*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        # init.kaiming_uniform_神经网络权重初始化，神经网络要优化一个非常复杂的非线性模型，而且基本没有全局最优解，
        # 初始化在其中扮演着非常重要的作用，尤其在没有BN等技术的早期，它直接影响模型能否收敛。

        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)

        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, adjacency, input_feature):
#         support = torch.mm(input_feature, self.weight)
#         output = torch.sparse.mm(adjacency, support)
#         if self.bias is not None:
#             output += self.bias
#         # 使用数值稳定的计算方法
#         log_probs_stable = F.log_softmax(output - torch.max(output, dim=1, keepdim=True).values, dim=1)
#         return log_probs_stable