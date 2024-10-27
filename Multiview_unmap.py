from audioop import bias
from re import S
from typing import ValuesView
from torch.nn.modules.activation import LeakyReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math

# torch.manual_seed(5)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")  

def cosineSimilarty(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)
    W=torch.mm(A,B.t())
    return W

def Normalize(data):                    #零均值归一化，处理后数据均值为0，方差为1
    """
    :param data:Input datasets
    :return:normalized datasets
    """
    # m = np.mean(data)
    # mx = np.max(data)
    # mn = np.min(data)
    # return (data - m+0.001) / (mx - mn)
    m = torch.mean(data).item()
    mx = torch.max(data).item()
    mn = torch.min(data).item()
    # return data/(mx+0.0001)
    return (data - m) / (mx - mn)

class Discriminator(nn.Module):
    def __init__(self,view_num,views_feadim,latdim,lr=0.005):
        super().__init__()
        extdim=100
        self.Opt=[]
        self.Distor=nn.Sequential(
            nn.Linear(extdim,latdim),
            nn.ReLU(),
            nn.Linear(latdim,latdim),
            nn.ReLU(),
            nn.Linear(latdim,view_num)
        )
        

    def forward(self,realX,fakeX,Sn,view_num):
        X=dict()
        for v in range(view_num):
            X[v]=realX[v][Sn[:,v]==1,:]
        FL=dict()
        TL=dict()
        for v in range(view_num):
            fake_label=self.Disct[v](fakeX[v])
            fl=torch.zeros(fake_label.shape)
            true_label=self.Disct[v](X[v].float())
            tl=torch.ones(true_label.shape)
            # loss=loss+self.GANloss(true_label,tl)+self.GANloss(fake_label,fl)
            FL[v]=fake_label
            TL[v]=true_label
        
        return TL,FL

class Simple_GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(Simple_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.rand(in_features,out_features),requires_grad=True)
        self.Bias=bias
        self.bias = nn.Parameter(torch.rand(1, 1, out_features),requires_grad=True)
       


    def forward(self, input, adj):
        A=adj
        D=torch.diag(torch.sum(A,dim=0))
        D_hat=torch.inverse(D)**0.5
        DAD=torch.mm(D_hat,torch.mm(A,D_hat))
        support = torch.matmul(input, self.weight)
        output = torch.matmul(DAD, support)
        if self.Bias:
            output=output+self.bias
        return output

class GraphDecoder(nn.Module):
    def __init__(self):
        super(GraphDecoder, self).__init__()
        # self.act=nn.ReLU()
        self.act=nn.Sigmoid()

    def forward(self, input):
        output=self.act(torch.mm(input,input.t()))
        return output

class CrossViewGCN_Layer1(nn.Module):
    def __init__(self,min_in_features, out_features,nSmp,bias=False,lr=0.005):
        super().__init__()
        self.min_in_features = min_in_features
        self.out_features = out_features
        self.nSmp=nSmp
        self.Opt = []  # 优化器
        self.weight = nn.Parameter(torch.rand(min_in_features,out_features),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(), lr=lr))
        self.Bias=bias
        self.bias = nn.Parameter(torch.rand(1, 1, out_features),requires_grad=True)
        #self.Opt.append(torch.optim.Adam(self.parameters(), lr=lr))
    def forward(self, input):
        self.cuda()
        nSmp=self.nSmp
        na = self.min_in_features  # 锚点数
        X=input[:,0:na]
        #print(X[0:1, 0:5])
        #print(X[0:1,na-5:na])
        realna=input.shape[1]-nSmp
        #print(input[0:1,realna-5:realna+5])

        A=input[:,realna-8:realna+nSmp-8]
        #A = input[:, na:na + nSmp]
        #print(A[0:1, 0:5])
        #print(A[0:1, nSmp - 5:nSmp])
        A = A + torch.eye(nSmp).cuda()
        D=torch.diag(torch.sum(A,dim=0))
        D_hat=torch.inverse(D)**0.5
        DAD=torch.mm(D_hat,torch.mm(A,D_hat))
        support = torch.matmul(X, self.weight)
        output = torch.matmul(DAD, support)
        if self.Bias:
            output=output+self.bias
        output=torch.cat((output,A),dim=1)
        return output

    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()

class CrossViewGCN_Layer2(nn.Module):
    def __init__(self,in_features, out_features,nSmp,bias=False,lr=0.005):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nSmp = nSmp
        self.Opt = []  # 优化器
        self.weight = nn.Parameter(torch.rand(in_features,out_features),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(), lr=lr))
        self.Bias=bias
        self.bias = nn.Parameter(torch.rand(1, 1, out_features),requires_grad=True)
        #self.Opt.append(torch.optim.Adam(self.parameters(), lr=lr))
    def forward(self, input):
        self.cuda()
        d=self.in_features#输入维度
        nSmp=self.nSmp
        X=input[:,0:d]
        A=input[:,d:d+nSmp]
        D=torch.diag(torch.sum(A,dim=0))
        D_hat=torch.inverse(D)**0.5
        DAD=torch.mm(D_hat,torch.mm(A,D_hat))
        support = torch.matmul(X, self.weight)
        output = torch.matmul(DAD, support)
        if self.Bias:
            output=output+self.bias
        return output

    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()

class TwoLayersGCN(nn.Module):
    def __init__(self, in_features,  latdim,na, bias=False, lr=0.005):
        super().__init__()
        self.hiddim=100
        self.na=na
        self.GCNlayer1 = CrossViewGCN_Layer1(in_features, self.hiddim, bias, lr)
        self.GCNlayer2 = CrossViewGCN_Layer2(self.hiddim, latdim, bias, lr)

    def forward(self, input, adj):
        H0=input
        nSmp=input.shape[0]
        A=adj+torch.eye(nSmp)
        H1=self.GCNlayer1(H0,A)
        H1=torch.tensor(F.relu(H1),dtype=torch.float)
        H2=self.GCNlayer2(H1,A)
        H2=torch.tensor(F.relu(H2),dtype=torch.float)
        return H2

    def update_params(self):
        self.GCNlayer1.update_params()
        self.GCNlayer2.update_params()




class GCN_HW(nn.Module):
    def __init__(self,view_num,views_feadim,nSmp,W,min_in_features,latdim,LinkSet,MapW,bias=False,lr=0.005):
        super().__init__()
        self.Opt=[]
        self.GCN=dict()
        self.H=torch.rand(nSmp,latdim*view_num)
        #self.H = nn.Parameter(torch.rand(nSmp, latdim * view_num),requires_grad=True).contiguous()  # 这个模型其实没有encoder，这里的H起的就是encoder的作用，他并不是将X放到encoder里，学习出一个潜在表示H，而是直接随机化一个H出来，就把这个当作encoder后的潜在表示，然后依靠后续decoder时和X对比并反向传播一步步更新H，直到H decoder后和X相似，就认为得到了X的良好潜在表示H。H确实是用X训练出来的，但不是X变换得来的。
        #self.Opt.append(torch.optim.Adam(self.parameters(), lr=lr))
        self.hiddim = 100
        self.W = W
        self.view_num = view_num
        self.latdim = latdim
        self.min_in_features = min_in_features
        self.LinkSet=LinkSet
        self.MapW=MapW
        self.nSmp=nSmp
        for i in range(self.view_num):
            d = nn.Sequential(
                # 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。除此之外，一个包含神经网络模块的OrderedDict也可以被传入nn.Sequential()容器中。利用nn.Sequential()搭建好模型架构，模型前向传播时调用forward()方法，模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。然后，第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果。
                CrossViewGCN_Layer1(self.min_in_features, self.hiddim,self.nSmp),
                nn.ReLU(),
                CrossViewGCN_Layer2(self.hiddim, self.latdim,self.nSmp),
                nn.ReLU(),
            )
            d.to('cuda')
            d.cuda()
            self.Opt.append(torch.optim.Adam(d.parameters(), lr=lr))
            self.GCN[i]=d
        #self.GCN[i]=TwoLayersGCN(in_features,latdim,bias,lr)
        self.Decoder = []
        for v in range(view_num):
            vfeadim = views_feadim[v]
            #vfeadim2 = int(vfeadim * 0.8)

            d = nn.Sequential(
                nn.Linear(self.latdim, self.hiddim),
                nn.ReLU(),
                nn.Linear(self.hiddim, self.hiddim),
                nn.ReLU(),
                nn.Linear(self.hiddim, vfeadim),
                # nn.Linear(vfeadim2,vfeadim),
            )
            d.to('cuda')
            d.cuda()
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(), lr=lr))
    def forward(self, input):
        recX = dict()
        loss=0
        X=input
        W=self.W
        AG = dict()
        for v in range(self.view_num):
            # 计算每个视图视图内的距离（每个视图内的相似度邻接矩阵）
            ViewW = self.get_EudisMat(X[v], X[v])
            AG[v] = ViewW
        for i in range(self.view_num):
            #Hv = self.H[:, self.latdim * i:self.latdim * (i + 1)]  # 对每个视图取他自己的那部分随机化的潜在表示H
            #recX[i] = self.Decoder[i](Hv)  # 这里已经在对潜在表示解码了，得到的是与X大小一致的矩阵，目的是通过使recX与X相似，不断反向传播更新H，从而得到好的H
                #loss+=torch.norm(recX[i]-X[i].float())
            #loss += F.mse_loss(recX[i], X[i].float())  # 这里加的是论文中的Lr
            #Dist = torch.cdist(Hv, Hv)
            #loss += torch.sum(Dist * self.W[i, i])
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point = self.LinkSet[i, j]
                    val, idx = torch.sort(Anchor_Point, descending=True)
                    AIDX = idx[0:int(torch.sum(Anchor_Point))]
                    # 寻找最近的k个锚点，并保留和计算节点和他们的相似度
                    AWi = self.find_k_nearst_AnchorSim(AG[i][:, AIDX],self.nSmp)  # AG[i]代表第i个视图内相似度，是个nxn矩阵，AG[i][:,AIDX]相当于每行只保留该点与视图内所有已对齐点的的相似度，是个nxna矩阵。
                    AWj = self.find_k_nearst_AnchorSim(AG[j][:, AIDX], self.nSmp)  # 返回的是i视图内的锚点图相似度(Z)，nxna大小的矩阵，目前na个维度都有值(所有认为已对齐的实例都作为锚点)，距离已经转化为相似度，已列归一化
                    INPUTi=torch.cat((AWi.float().cuda(),W[i,i].cuda()),dim=1)
                    Hi=self.GCN[i](INPUTi)
                    Hi = torch.nn.functional.normalize(Hi, p=1, dim=1)
                    Hi = Hi * 5
                    self.H[:,i*self.latdim:(i+1)*self.latdim]=Hi
                    Wij = W[i, j]
                    INPUTj=torch.cat((torch.mm(Wij.cuda(),AWj.float().cuda()).cuda(),W[j,j].cuda()),dim=1)
                    Hj=self.GCN[j](INPUTj)
                    Hj=torch.nn.functional.normalize(Hj, p=1, dim=1)
                    Hj=Hj*5
                    self.H[:, j * self.latdim:(j + 1) * self.latdim]=Hj
                    #recXi = self.Decoder[i](Hi)
                    #loss += F.mse_loss(recXi, X[i].float())  # 这里加的是论文中的Lr
                    loss+=torch.norm(Hi-Hj,p='fro')*self.MapW[i,j]
                    #loss += F.mse_loss(Hi.float(), Hj.float())*self.MapW[i,j]
                    Dist = torch.cdist(Hi, Hi)
                    loss += 0.3*torch.sum(Dist * W[i, i].cuda())  # 这里加的是论文中的Lg


                    '''INPUTi = torch.cat((AWi.float(), W[i, i]), dim=1)
                    Hi = self.GCN[i](INPUTi)
                    self.H[:, i * self.latdim:(i + 1) * self.latdim] = Hi
                    INPUTj = torch.cat((AWj.float(), W[j, j]), dim=1)
                    Hj = self.GCN[j](INPUTj)
                    self.H[:, j * self.latdim:(j + 1) * self.latdim] = Hj
                    loss += torch.norm(Hi - torch.mm(W[i,j],Hj.float()), p='fro')'''
        return loss

    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()

    def getH(self):
        return self.H

    def get_EudisMat(self,A, B):
        # A nSmp_a*mfea
        # B nSmp_b*mfea
        # D nSmp_a*nSmp_b

        D = torch.cdist(A, B)  # 批量计算两个向量集合的距离，p 默认为2，为欧几里德距离。p=2的欧几里德距离也是L2范式，如果p=1即是L1范式
        # print(torch.norm(A[0,:]-B[0,:]))
        # print(torch.sum((A[0,:]-B[0,:])*(A[0,:]-B[0,:]))**0.5)
        return D

    def get_A(self,Z):
        Z = Z / (torch.norm(Z, dim=1, p=2,keepdim=True) + 0.000001)
        A = torch.mm(Z, Z.t())
        return A

    def find_k_nearst_AnchorSim(self,Dist, nSmp):
        # 这里对每个实例选取离他最近的K个已对齐实例作为锚点进行计算（如果所有的都作为锚点就不用选），k可以取别的值
        Dist = Dist.float()
        sigema = 1
        # HW K=int(Dist.shape[1])
        k = int(Dist.shape[1])  # k等目前于所有已对齐实例数
        # k=int(Dist.shape[1]*0.5)
        if k != int(Dist.shape[1]):
            # 如果不是选取所有已对齐实例作为锚点，那么其他的没选上的实例置负无穷
            # maxval,idx=torch.max(Dist,0)
            # minval,idx=torch.min(Dist,0)
            # Dist=(Dist-minval)/(maxval-minval+0.000001)
            val, idx = torch.sort(Dist, descending=False)
            val = torch.reshape(val[:, k], shape=(nSmp, 1))  # 这几步就是在选最近的k个，因为Dist是距离，距离越短越好，所以从小到大排序要前k个
            Dist = torch.where(Dist > val, torch.FloatTensor([float('-inf')]), -Dist / torch.mean(
                Dist)).float()  # 最后得到的就是nxna的矩阵，只不过每行只有距离该行对应实例距离最小的k个已对齐实例的相似度，其余都置为负无穷了
            # Dist=torch.softmax(Dist,1)     #这个地方很奇怪，-Dist/torch.mean(Dist)意思是用DIST该位置元素除以整个矩阵的均值（有负无穷之前的矩阵），不知道是什么操作         #并且可以发现根据代码，这里对每个实例选出的k个锚点其实不是固定的，而是针对每个实例分别选出k个据他最近的已对齐实例作为锚点
            Dist = torch.exp(Dist)
            # Dist=torch.where(Dist<0.01,torch.FloatTensor([0.0]),Dist)
        else:
            # 直接softmax就好了
            # 尝试归一化这个距离
            # val,idx=torch.max(Dist,0)
            # Dist=-Dist/val
            # Dist=torch.exp(Dist)
            # maxval,idx=torch.max(Dist,0)
            # minval,idx=torch.min(Dist,0)
            # Dist=(Dist-minval)/(maxval-minval+0.000001)
            Dist = torch.exp(
                -Dist / torch.mean(Dist, 0, keepdim=True))  # 特征缩放？让每个特征权重不因为特征值大小差距出现差距，也是归一化的一种，是在列（单个特征）上归一化
            # Dist=torch.softmax(Dist,1)
            maxval, idx = torch.max(Dist, 0)
            minval, idx = torch.min(Dist, 0)
            Dist = (Dist - minval) / (maxval - minval + 0.000001)  # +0.000001为了防止除零错误？底下这三行是归一化
        return Dist

class MNet_GAE_HW(nn.Module):
    def __init__(self,view_num, views_feadim, nSmp,latdim,A,LinkSet,alpha,beta,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=200
        self.A=A
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        self.alpha=alpha
        self.beta=beta
        # init encoder decoder
        self.GCNE1=[] 
        self.GCNE2=[]
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            # encoder
            e1=Simple_GraphConvolution(vfeadim,self.extdim)
            # e1=Simple_GraphConvolution(vfeadim,self.latdim)
            self.GCNE1.append(e1)
            self.Opt.append(torch.optim.Adam(e1.parameters(),lr=lr))

            e2=Simple_GraphConvolution(self.extdim,self.latdim)
            self.GCNE2.append(e2)
            self.Opt.append(torch.optim.Adam(e2.parameters(),lr=lr))
            # decoder
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim)
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
       
        # self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.G=dict()
    def ext_mat(self,A,Link):
        return A[Link][:,Link]


    def forward(self,X,view_num):
        Zv=dict()
        loss=0
        nSmp=X[0].shape[0]
        for v in range(view_num):
            Hv=self.GCNE1[v](X[v].float(),self.A[v,v])
            Hv=F.relu(Hv)
            Hv=self.GCNE2[v](Hv,self.A[v,v])
            Zv[v]=F.sigmoid(Hv)
            # Zv[v]=Hv
            # loss=loss+torch.norm(self.Decoder[v](Zv[v]).float()-X[v].float())
        
        self.H=Zv
        for v in range(view_num):
            self.G[v,v]=torch.exp(-torch.cdist(Zv[v],Zv[v])/2)
            loss=loss+self.alpha*self.GAEloss(self.A[v,v],self.G[v,v])/nSmp

        for i in range(view_num):
            for j in range(view_num):
                if i!=j:
                    loss=loss+torch.norm(Zv[i][self.LinkSet[i,j],:]-Zv[j][self.LinkSet[i,j],:])
                    Gij=torch.exp(-torch.cdist(Zv[i],Zv[j])/2)
                    self.G[i,j]=Gij
                    GAij=self.ext_mat(Gij,self.LinkSet[i,j])
                    AAij=self.ext_mat(self.A[i,j],self.LinkSet[i,j]).float()
                    loss=loss+self.beta*self.GAEloss(AAij,GAij)/nSmp
        return loss
    def GAEloss(self,Aij,Gij):
        # 正例向1 负例低于0.3即可
        UAij=(1-Aij)*torch.ones(Gij.shape)*0.3+Aij
        return torch.sum(torch.max(Aij-Gij,torch.zeros(Gij.shape)))+torch.sum(torch.max(Gij-UAij,torch.zeros(Gij.shape)))
        # loss=torch.sum(Gij*Aij)
        # return loss
    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        

    def getH(self):
        return self.H
    
    def getG(self):
        return self.G

class MNet_AE_HW(nn.Module):
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=100
        self.W=W
        
        self.view_num=view_num
        self.latdim=latdim
        # init encoder decoder
        self.Decoder=[]
        self.Encoder=[] 
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
            # encoder
            e=nn.Sequential(
                nn.Linear(vfeadim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.latdim),
            )
            self.Encoder.append(e)
            self.Opt.append(torch.optim.Adam(e.parameters(),lr=lr))
            # decoder
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(), 
                nn.Linear(vfeadim2,vfeadim)
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
           

        self.L=self.conL(W)

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g
        return L
    def forward(self,X,view_num,mode="Pretrain"):
        Hv=dict()
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        nSmp=X[0].shape[0]
        for v in range(view_num):
            Hv[v]=self.Encoder[v](X[v].float())
            recX[v]=self.Decoder[v](Hv[v])
            loss+=torch.norm(recX[v]-X[v])
            for vv in range(view_num):
                if vv!=v:
                    recX[vv]=self.Decoder[vv](Hv[v]).float()
                    W=self.W[v,vv]
                    W=W/torch.sum(W,1)
                    # hw 0.5
                    loss+=torch.norm(recX[vv]-torch.mm(W,X[vv].float()),p='fro')**2*0.5
                    

        for i in range(view_num):
            # loss+=torch.sum((torch.cdist(Hv[i],Hv[i])**2)*self.W[i,i,i])
            HLH=torch.mm(Hv[i].t(),torch.mm(self.L[i],Hv[i]))
            # hw 0.5
            loss+=torch.trace(HLH)*0.5
            for j in range(view_num):
                W=self.W[i,j]
                W=W/torch.sum(W,1)
                loss+=torch.norm(torch.mm(W,Hv[i])-Hv[j])

        self.H=Hv
        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        

    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL

class MNet_AE_ORL(nn.Module):
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.W=W
        
        self.view_num=view_num
        self.latdim=latdim
        # init encoder decoder
        self.Decoder=[]
        self.Encoder=[] 
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
            # encoder
            e=nn.Sequential(
                nn.Linear(vfeadim,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,self.latdim),
            )
            self.Encoder.append(e)
            self.Opt.append(torch.optim.Adam(e.parameters(),lr=lr))
            # decoder
            d=nn.Sequential(
                nn.Linear(self.latdim,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(), 
                nn.Linear(vfeadim2,vfeadim)
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
        self.L=self.conL(W)

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g
        return L
    def forward(self,X,view_num,mode="Pretrain"):
        Hv=dict()
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        nSmp=X[0].shape[0]
        for v in range(view_num):
            Hv[v]=self.Encoder[v](X[v].float())
            recX[v]=self.Decoder[v](Hv[v])
            loss+=torch.norm(recX[v]-X[v])
            for vv in range(view_num):
                if vv!=v:
                    recX[vv]=self.Decoder[vv](Hv[v]).float()
                    W=self.W[v,vv]
                    W=W/torch.sum(W,1)
                    # hw 0.5
                    loss+=torch.norm(recX[vv]-torch.mm(W,X[vv].float()),p='fro')**2
                    

        for i in range(view_num):
            # loss+=torch.sum((torch.cdist(Hv[i],Hv[i])**2)*self.W[i,i,i])
            HLH=torch.mm(Hv[i].t(),torch.mm(self.L[i],Hv[i]))
            # hw 0.5
            loss+=torch.trace(HLH)*0.05
            for j in range(view_num):
                W=self.W[i,j]
                W=W/torch.sum(W,1)
                loss+=torch.norm(torch.mm(W,Hv[i])-Hv[j])

        self.H=Hv
        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        

    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL


class MNet_SiaNetMAP_HW(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    # 训练使得有连接的相近，跨视图一致
    # 最终通过编码的表示计算相似度,得到跨视图连接图
    # 训练出来虽然能保证映射到同类实例,但是却也打乱了原先的空间分布

    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=100
        self.W=W
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init encoder
        self.Encoder=[] 
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
            # encoder
            e=nn.Sequential(
                nn.Linear(vfeadim,vfeadim2),
                nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(vfeadim2,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.latdim),
            )
            self.Encoder.append(e)
            self.Opt.append(torch.optim.Adam(e.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)

    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g
        return L
    def forward(self,X,view_num,mode="Pretrain"):
        Hv=dict()
        Dist=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        nSmp=X[0].shape[0]
        tao=1
        for v in range(view_num):
            Hv[v]=self.Encoder[v](X[v].float())
            D=torch.cdist(Hv[v],Hv[v])**2
            W=self.W[v,v]>1e-3
            loss+=torch.sum(D*W)
            # random select negative pair
            W=(self.W[v,v]<1e-3)*torch.rand(self.W[v,v].shape)
            val,idx=torch.sort(W,descending=True)
            val=torch.reshape(val[:,3],shape=(nSmp,1))
            W=torch.where(W<val,torch.FloatTensor([0.0]),torch.FloatTensor([1.0])).float()
            # loss
            W=(tao-D)*W
            loss+=torch.sum(torch.max(W,torch.zeros(W.shape)))
            Dist[v]=D

        for i in range(view_num):
            for j in range(i+1,view_num):
                if i!=j:
                    AIDX=self.AnchorIDX[i,j]
                    # Di=Dist[i][AIDX,AIDX]
                    # Dj=Dist[j][AIDX,AIDX]
                    # loss+=0.1*torch.norm(Di-Dj,'fro')**2
                    loss+=torch.norm(Hv[i][AIDX,:]-Hv[j][AIDX,:],'fro')**2
                    # W=self.W[i,j]
                    # W=W/(torch.sum(W,1,keepdim=True)+0.0001)
                    # loss+=torch.norm(torch.mm(W,Hv[i])-Hv[j],p='fro')**2
    
        self.H=Hv
        return loss

    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL

class MNet_AE_Reuters(nn.Module):
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=200
        self.W=W
        self.view_num=view_num
        self.latdim=latdim
        # init encoder decoder
        self.Decoder=[]
        self.Encoder=[] 
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
            # encoder
            e=nn.Sequential(
                nn.Linear(vfeadim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.latdim),
            )
            self.Encoder.append(e)
            self.Opt.append(torch.optim.Adam(e.parameters(),lr=lr))
            # decoder
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(), 
                nn.Linear(vfeadim2,vfeadim)
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
           

        self.L=self.conL(W)

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g
        return L
    def forward(self,X,view_num,mode="Pretrain"):
        Hv=dict()
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        nSmp=X[0].shape[0]
        for v in range(view_num):
            Hv[v]=self.Encoder[v](X[v].float())
            recX[v]=self.Decoder[v](Hv[v])
            loss+=torch.norm(recX[v]-X[v])
            for vv in range(view_num):
                if vv!=v:
                    recX[vv]=self.Decoder[vv](Hv[v]).float()
                    W=self.W[v,vv]
                    W=W/(torch.sum(W,1,keepdim=True)+0.0001)
                    # hw 0.5
                    loss+=torch.norm(recX[vv]-torch.mm(W,X[vv].float()),p='fro')**2*0.5
                    

        for i in range(view_num):
            # loss+=torch.sum((torch.cdist(Hv[i],Hv[i])**2)*self.W[i,i,i])
            HLH=torch.mm(Hv[i].t(),torch.mm(self.L[i],Hv[i]))
            # hw 0.5
            loss+=torch.trace(HLH)*0.5
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    # W=W/torch.sum(W,1)
                    W=W/(torch.sum(W,1,keepdim=True)+0.0001)
                    loss+=torch.norm(torch.mm(W,Hv[i])-Hv[j])

        self.H=Hv
        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        

    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL


class MNet_reNet_Cal(nn.Module):
    # epoch 200?
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,MapW,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=200
        self.W=W
        self.MapW=MapW
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),

                nn.Linear(self.extdim,self.extdim),
                # nn.Linear(vfeadim2,vfeadim2),
                nn.Linear(self.extdim,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()

        if mode=="Pretrain":
            loss=0
        loss=0
        nSmp=X[0].shape[0]

        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]
            recX[i]=self.Decoder[i](Hv)
            loss+=torch.norm(recX[i]-X[i].float())*10
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.5
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    W=W/torch.sum(W,1)
                    recX[j]=self.Decoder[j](Hv)
                    loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*self.MapW[i,j]
                    # AIDX=self.AnchorIDX[i,j]
                    # Hi=self.H[:,self.latdim*i:self.latdim*(i+1)][AIDX,:]
                    # Hj=self.H[:,self.latdim*j:self.latdim*(j+1)][AIDX,:]
                    # loss+=torch.norm(Hi-Hj,p='fro')
        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL
    
class MNet_reNet_Sce15(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=150
        self.W=W
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]    
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                # nn.Linear(vfeadim2,vfeadim2),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()

        if mode=="Pretrain":
            loss=0
        loss=0

        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]
            recX[i]=self.Decoder[i](Hv)
            loss+=torch.norm(recX[i]-X[i].float())
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.1
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    W=W/torch.sum(W,1)
                    recX[j]=self.Decoder[j](Hv)
                    loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*0.5
                    AIDX=self.AnchorIDX[i,j]
                    Hi=self.H[:,self.latdim*i:self.latdim*(i+1)][AIDX,:]
                    Hj=self.H[:,self.latdim*j:self.latdim*(j+1)][AIDX,:]
                    loss+=torch.norm(Hi-Hj,p='fro')*0.1

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL
  
class MNet_reNet_Reuters(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=500
        self.W=W
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                # nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                # nn.ReLU(),
                nn.Linear(self.extdim,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()

        if mode=="Pretrain":
            loss=0
        loss=0
        nSmp=X[0].shape[0]

        # for v in range(view_num):
        #     Hv=self.H[:,self.latdim*v:self.latdim*(v+1)]
        #     recX[v]=self.Decoder[v](Hv)
        #     loss+=torch.norm(recX[v]-X[v].float())
    
        # for v in range(view_num):
        #     if v==0:
        #         recX[v]=self.Decoder[v](self.H)
        #         loss+=torch.norm(recX[v]-X[v].float())
        #         Dist=torch.cdist(self.H,self.H)
        #         loss+=torch.sum(Dist*self.W[v,v])

        #     else:
        #         W=self.W[0,v]
        #         W=W/torch.sum(W,1)
        #         recX[v]=self.Decoder[v](self.H)
        #         loss+=torch.norm(torch.mm(W,recX[v])-X[v].float())

        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]
            recX[i]=self.Decoder[i](Hv)
            loss+=torch.norm(recX[i]-X[i].float())
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.05
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    W=W/torch.sum(W,1)
                    recX[j]=self.Decoder[j](Hv)
                    loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*0.05
                    AIDX=self.AnchorIDX[i,j]
                    Hi=self.H[:,self.latdim*i:self.latdim*(i+1)][AIDX,:]
                    Hj=self.H[:,self.latdim*j:self.latdim*(j+1)][AIDX,:]
                    loss+=torch.norm(Hi-Hj,p='fro')

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL
    
class MNet_reNet_BDGP(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=100
        self.W=W
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L
    def getHv(self,v):
        for j in range(self.view_num):
            AIDX=self.AnchorIDX[v,j]
            Hv=self.H[:,self.latdim*v:self.latdim*(v+1)][AIDX,:]
        return Hv

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]
            recX[i]=self.Decoder[i](Hv)
            loss+=torch.norm(recX[i]-X[i].float())
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.5
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    W=W/torch.sum(W,1)
                    recX[j]=self.Decoder[j](Hv)
                    loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*0.1
                    AIDX=self.AnchorIDX[i,j]
                    Hi=self.H[:,self.latdim*i:self.latdim*(i+1)][AIDX,:]
                    Hj=self.H[:,self.latdim*j:self.latdim*(j+1)][AIDX,:]
                    loss+=torch.norm(Hi-Hj,p='fro')

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL

class MNet_reNet_HW(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,MapW,lr=0.005):#latdim = latent dim = n_Class
        super().__init__()
        self.Opt=[]   #优化器
        self.extdim=100
        self.W=W
        self.MapW=MapW
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True).contiguous()   #这个模型其实没有encoder，这里的H起的就是encoder的作用，他并不是将X放到encoder里，学习出一个潜在表示H，而是直接随机化一个H出来，就把这个当作encoder后的潜在表示，然后依靠后续decoder时和X对比并反向传播一步步更新H，直到H decoder后和X相似，就认为得到了X的良好潜在表示H。H确实是用X训练出来的，但不是X变换得来的。
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init decoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            #vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(    #一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。除此之外，一个包含神经网络模块的OrderedDict也可以被传入nn.Sequential()容器中。利用nn.Sequential()搭建好模型架构，模型前向传播时调用forward()方法，模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。然后，第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果。
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim),
                # nn.Linear(vfeadim2,vfeadim),
            )
            d.cuda()
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)     #求每个视图内相似度的拉普拉斯矩阵
        self.getAnchor(LinkSet=LinkSet)     #求每两个视图间锚点的索引，以列表形式，存储在字典AnchorIDX中
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L
    def getHv(self,v):
        for j in range(self.view_num):
            AIDX=self.AnchorIDX[v,j]
            Hv=self.H[:,self.latdim*v:self.latdim*(v+1)][AIDX,:]
        return Hv

    def forward(self,X,view_num,mode="Pretrain"):
        self.cuda()
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]    #对每个视图取他自己的那部分随机化的潜在表示H
            recX[i]=self.Decoder[i](Hv).cuda()     #这里已经在对潜在表示解码了，得到的是与X大小一致的矩阵，目的是通过使recX与X相似，不断反向传播更新H，从而得到好的H
            # loss+=torch.norm(recX[i]-X[i].float())
            loss+=F.mse_loss(recX[i],X[i].float())  #这里加的是论文中的Lr
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i].cuda())   #这里加的是论文中的Lg
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j].cuda()
                    W=W/torch.sum(W,1).cuda()
                    recX[j]=self.Decoder[j](Hv).cuda()
                    loss+=torch.norm(recX[j].cuda() -torch.mm(W,X[j].float().cuda() ),p='fro').cuda() *self.MapW[i,j].cuda()   #这里加的是论文中的Lcvr
                    # AIDX=self.AnchorIDX[i,j]
                    # Hi=self.H[:,self.latdim*i:self.latdim*(i+1)][AIDX,:]
                    # Hj=self.H[:,self.latdim*j:self.latdim*(j+1)][AIDX,:]
                    # loss+=torch.norm(Hi-Hj,p='fro')

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL
 
class MNet_reNet_BBCsports3view(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    # map 0.1 epoch 100 alpha 0.5
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,MapW,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=50
        self.W=W
        self.MapW=MapW
        # self.MapW=torch.ones(view_num,view_num)
        # self.MapW=torch.zeros(view_num,view_num)
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                # nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                # nn.ReLU(),
                nn.Linear(self.extdim,vfeadim),
                # nn.Linear(vfeadim2,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L
    def getHv(self,v):
        if v==0:
            Hv=self.H[:,self.latdim*v:self.latdim*(v+1)]
        if v==1:
            W2=torch.diag(1-self.LinkSet[0,1]).float()
            H2=self.H[:,self.latdim*v:self.latdim*(v+1)]
            W1=torch.diag(self.LinkSet[0,1]).float()
            H1=self.H[:,self.latdim*0:self.latdim*1]
            Hv=W1@H1+W2@H2
        if v==2:
            W1=torch.diag(self.LinkSet[0,2]).float()
            H1=self.H[:,self.latdim*0:self.latdim*1]
            W2=torch.diag(torch.where((self.LinkSet[1,2]-self.LinkSet[0,2])>0,1,0)).float()
            H2=self.H[:,self.latdim*1:self.latdim*(2)]
            W3=torch.diag(torch.where((self.LinkSet[1,2]+self.LinkSet[0,2])>0,0,1)).float()
            H3=self.H[:,self.latdim*v:self.latdim*(v+1)]
            Hv=W1@H1+W2@H2+W3@H3
        return Hv

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]
            recX[i]=self.Decoder[i](Hv)
            loss+=torch.norm(recX[i]-X[i].float())
            loss+=F.mse_loss(recX[i],X[i].float())
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.5
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    W=W/(torch.sum(W,1)+0.0001)
                    recX[j]=self.Decoder[j](Hv)
                    # loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*self.MapW[i,j]
                    loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*2*self.MapW[i,j]
                    # loss+=F.mse_loss(recX[j],torch.mm(W,X[j].float()))*self.MapW[i,j]

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL
 
class MNet_reNet_3Sources(nn.Module):
    # 输入多视图unmap数据，和视图内相似度图，视图间相似度图，锚点连接
    # map 0.1 epoch 120 alpha 0.5
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,MapW,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=50
        self.W=W
        self.MapW=MapW
        # self.MapW=torch.ones(view_num,view_num)
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.extdim,self.extdim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.extdim,vfeadim),
                # nn.Linear(vfeadim2,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L
    def getHv(self,v):
        if v==0:
            Hv=self.H[:,self.latdim*v:self.latdim*(v+1)]
        if v==1:
            W2=torch.diag(1-self.LinkSet[0,1]).float()
            H2=self.H[:,self.latdim*v:self.latdim*(v+1)]
            W1=torch.diag(self.LinkSet[0,1]).float()
            H1=self.H[:,self.latdim*0:self.latdim*1]
            Hv=W1@H1+W2@H2
        if v==2:
            W1=torch.diag(self.LinkSet[0,2]).float()
            H1=self.H[:,self.latdim*0:self.latdim*1]
            W2=torch.diag(torch.where((self.LinkSet[1,2]-self.LinkSet[0,2])>0,1,0)).float()
            H2=self.H[:,self.latdim*1:self.latdim*(2)]
            W3=torch.diag(torch.where((self.LinkSet[1,2]+self.LinkSet[0,2])>0,0,1)).float()
            H3=self.H[:,self.latdim*v:self.latdim*(v+1)]
            Hv=W1@H1+W2@H2+W3@H3
        return Hv

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        for i in range(view_num):
            Hv=self.H[:,self.latdim*i:self.latdim*(i+1)]
            recX[i]=self.Decoder[i](Hv)
            # loss+=torch.norm(recX[i]-X[i].float())
            loss+=F.mse_loss(recX[i],X[i].float())
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.5
            for j in range(view_num):
                if i!=j:
                    if self.MapW[i,j]==0:
                        continue
                    else:
                        W=self.W[i,j]
                        W=W/(torch.sum(W,1)+0.0001)
                        recX[j]=self.Decoder[j](Hv)
                        # loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*0.5
                        # loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*self.MapW[i,j]
                        loss+=F.mse_loss(recX[j],torch.mm(W,X[j].float()))*self.MapW[i,j]

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        return self.H
    
    def getLH(self):
        return self.comL
 


class MNet_reNet_BBCsports3view_lose(nn.Module):
    # 废弃端口
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.extdim=40
        self.W=W
        self.H=nn.Parameter(torch.rand(nSmp,latdim*view_num),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))
        self.LinkSet=LinkSet
        self.view_num=view_num
        self.latdim=latdim
        # init encoder
        self.Decoder=[]
        for v in range(view_num):
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
        
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim),
                # nn.Linear(vfeadim2,vfeadim),
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
   
        self.L=self.conL(W)
        self.getAnchor(LinkSet=LinkSet)
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        for i in range(self.view_num):
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g       
        return L
    def getHv(self,v):
        if v==0:
            Hv=self.H[:,self.latdim*v:self.latdim*(v+1)]
        if v==1:
            W2=torch.diag(1-self.LinkSet[0,1]).float()
            H2=self.H[:,self.latdim*v:self.latdim*(v+1)]
            W1=torch.diag(self.LinkSet[0,1]).float()
            H1=self.H[:,self.latdim*0:self.latdim*1]
            Hv=W1@H1+W2@H2
        if v==2:
            W1=torch.diag(self.LinkSet[0,2]).float()
            H1=self.H[:,self.latdim*0:self.latdim*1]
            W2=torch.diag(torch.where((self.LinkSet[1,2]-self.LinkSet[0,2])>0,1,0)).float()
            H2=self.H[:,self.latdim*1:self.latdim*(2)]
            W3=torch.diag(torch.where((self.LinkSet[1,2]+self.LinkSet[0,2])>0,0,1)).float()
            H3=self.H[:,self.latdim*v:self.latdim*(v+1)]
            Hv=W1@H1+W2@H2+W3@H3
        return Hv

    def forward(self,X,view_num,mode="Pretrain"):
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        for i in range(view_num):
            Hv=self.getHv(i)
            recX[i]=self.Decoder[i](Hv)
            # loss+=torch.norm(recX[i]-X[i].float())
            loss+=F.mse_loss(recX[i],X[i].float())
            Dist=torch.cdist(Hv,Hv)
            loss+=torch.sum(Dist*self.W[i,i])*0.5
            for j in range(view_num):
                if i!=j:
                    W=self.W[i,j]
                    W=W/(torch.sum(W,1)+0.0001)
                    recX[j]=self.Decoder[j](Hv)
                    loss+=torch.norm(recX[j]-torch.mm(W,X[j].float()),p='fro')*0.5

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
    
    def getH(self):
        Hv=dict()
        for v in range(self.view_num):
            Hv[v]=self.getHv(v)
        return Hv
    
    def getLH(self):
        return self.comL
 
class SelfExpression(nn.Module):
    def __init__(self, nSmp,nAnchor,AIDX):
        super(SelfExpression, self).__init__()
        self.AIDX=AIDX
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(nSmp, nAnchor, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        AIDX=self.AIDX
    
        y = torch.matmul(self.Coefficient, x)
        return y

class MNet_AESelf_HW(nn.Module):
    # 用亲和度矩阵构造局部流形
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.view_num=view_num
        self.latdim=latdim
        self.extdim=100
        self.W=W
        self.comAnchorIDX,self.ncomAnchor=self.getAnchor(LinkSet=LinkSet)
        self.AnH=nn.Parameter(torch.rand(nSmp,latdim),requires_grad=True)
        self.Opt.append(torch.optim.Adam(self.parameters(),lr=lr))

        # init encoder decoder
        self.Decoder=[]
        self.Encoder=[] 
        self.SFEXP=[]
        for v in range(view_num):
            sfexp = SelfExpression(nSmp,self.ncomAnchor)
            self.SFEXP.append(sfexp)
            self.Opt.append(torch.optim.Adam(sfexp.parameters(),lr=lr))
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
            # encoder
            e=nn.Sequential(
                nn.Linear(vfeadim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.latdim),
            )
            self.Encoder.append(e)
            self.Opt.append(torch.optim.Adam(e.parameters(),lr=lr))
            # decoder
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(), 
                nn.Linear(vfeadim2,vfeadim)
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
           
        self.L=self.conL(W)
        
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        comAnchor=torch.zeros(LinkSet[0,1].shape)
        for i in range(self.view_num):
            comAIDXi=torch.zeros(LinkSet[0,1].shape)
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX
                    comAnchor+=Anchor_Point
                    comAIDXi+=Anchor_Point
            val,idx=torch.sort(comAIDXi,descending=True)
            comAIDXi=idx[0:int(torch.sum(comAIDXi))]
            self.AnchorIDX[i]=comAIDXi

        comAnchor=torch.where(comAnchor>0,1,0)
        val,idx=torch.sort(comAnchor,descending=True)
        comAIDX=idx[0:int(torch.sum(comAnchor))]

        return comAIDX,int(torch.sum(comAnchor))

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g
        return L
    def forward(self,X,view_num,mode="Pretrain"):
        Hv=dict()
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        comAIDX=self.comAnchorIDX
        nSmp=X[0].shape[0]
        
        for v in range(view_num):
            Hv[v]=self.Encoder[v](X[v].float())
            recX[v]=self.Decoder[v](Hv[v])
            loss+=F.mse_loss(recX[v],X[v].float(),reduction="sum")
            AIDXv=self.AnchorIDX[v]
            loss+=0.5*F.mse_loss(Hv[v][AIDXv,:],self.AnH[AIDXv,:])
            SRH=self.SFEXP[v](self.AnH[comAIDX,:])
            loss+=0.5*F.mse_loss(Hv[v],SRH)   
            # loss+=0.5*torch.sum(torch.pow(self.self.SFEXP[v].Coefficient[AIDXv,:], 2))
            loss+=0.5*torch.sum(torch.pow(self.SFEXP[v].Coefficient, 2))
            # loss+=0.5*torch.norm(self.SFEXP[v].Coefficient,1)
            loss+=F.mse_loss(self.Decoder[v](SRH),X[v].float(),reduction="sum")
          
                    

        # for i in range(view_num):
        #     # loss+=torch.sum((torch.cdist(Hv[i],Hv[i])**2)*self.W[i,i,i])
        #     HLH=torch.mm(Hv[i].t(),torch.mm(self.L[i],Hv[i]))
        #     # hw 0.5
        #     loss+=torch.trace(HLH)*0.5
        #     for j in range(view_num):
        #         W=self.W[i,j]
        #         W=W/torch.sum(W,1)
        #         loss+=torch.norm(torch.mm(W,Hv[i])-Hv[j])

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        

    def getH(self):
        Hv=dict()
        for v in range(self.view_num):
            Hv[v]=self.SFEXP[v].Coefficient
        return Hv
    
    def getLH(self):
        return self.comL


def cosineSimilarty(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)
    W=torch.mm(A,B.t())/1.0000001
    return W
class MNet_AESelf_2view(nn.Module):
    # 锚点语义表示：sum
    # 学习子空间的锚点图，利用原始空间的锚点图，构造自适应锚点图
    # 暂时不行，先搁置
    def __init__(self,view_num, views_feadim, nSmp,latdim,W,LinkSet,lr=0.005):
        super().__init__()
        self.Opt=[]   
        self.view_num=view_num
        self.latdim=latdim
        self.extdim=200
        self.W=W
        self.comAnchorIDX,self.ncomAnchor=self.getAnchor(LinkSet=LinkSet)
        # init encoder decoder
        self.Decoder=[]
        self.Encoder=[] 

        self.SFEXP=[]
        for v in range(view_num):
            sfexp = SelfExpression(nSmp,self.ncomAnchor,self.comAnchorIDX)
            self.SFEXP.append(sfexp)
            self.Opt.append(torch.optim.Adam(sfexp.parameters(),lr=lr))
          
            vfeadim=views_feadim[v]
            vfeadim2=int(vfeadim*0.8)
            # encoder
            e=nn.Sequential(
                nn.Linear(vfeadim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,latdim)
            )
            self.Encoder.append(e)
            self.Opt.append(torch.optim.Adam(e.parameters(),lr=lr))
            # decoder
            d=nn.Sequential(
                nn.Linear(self.latdim,self.extdim),
                nn.ReLU(),
                nn.Linear(self.extdim,vfeadim2),
                nn.ReLU(), 
                nn.Linear(vfeadim2,vfeadim)
            )
            self.Decoder.append(d)
            self.Opt.append(torch.optim.Adam(d.parameters(),lr=lr))
           
        self.L=self.conL(W)
        
   
    def getAnchor(self,LinkSet):
        self.AnchorIDX=dict()
        comAnchor=torch.zeros(LinkSet[0,1].shape)
        for i in range(self.view_num):
            comAIDXi=torch.zeros(LinkSet[0,1].shape)
            for j in range(self.view_num):
                if i!=j:
                    Anchor_Point=LinkSet[i,j]
                    val,idx=torch.sort(Anchor_Point,descending=True)
                    AIDX=idx[0:int(torch.sum(Anchor_Point))]
                    # 这里AIDX是得到ij视图的锚点索引list
                    self.AnchorIDX[i,j]=AIDX
                    comAnchor+=Anchor_Point
                    comAIDXi+=Anchor_Point
            val,idx=torch.sort(comAIDXi,descending=True)
            comAIDXi=idx[0:int(torch.sum(comAIDXi))]
            self.AnchorIDX[i]=comAIDXi

        comAnchor=torch.where(comAnchor>0,1,0)
        val,idx=torch.sort(comAnchor,descending=True)
        comAIDX=idx[0:int(torch.sum(comAnchor))]

        return comAIDX,int(torch.sum(comAnchor))

    def conL(self,W):
        L=dict()
        for v in range(self.view_num):
            g=W[v,v]
            D=torch.diag(torch.sum(g,dim=0))
            L[v]=D-g
        return L
    def forward(self,X,view_num,mode="Pretrain"):
        Hv=dict()
        recX=dict()
        if mode=="Pretrain":
            loss=0
        loss=0
        comAIDX=self.comAnchorIDX
        nSmp=X[0].shape[0]
        
        for v in range(view_num):
            Hv[v]=self.Encoder[v](X[v].float())
            recX[v]=self.Decoder[v](Hv[v])
            loss+=F.mse_loss(recX[v],X[v].float(),reduction="sum")
        
        AnH=(Hv[0][self.comAnchorIDX,:]+Hv[1][self.comAnchorIDX,:])/2
        Dist=torch.cdist(Hv[0],Hv[1])
        loss+=torch.sum(Dist@cosineSimilarty(self.SFEXP[0].Coefficient,self.SFEXP[1].Coefficient))
        for v in range(view_num):
            SRH=self.SFEXP[v](AnH)
            loss+=0.5*F.mse_loss(Hv[v],SRH)   
            # loss+=0.5*torch.sum(torch.pow(self.self.SFEXP[v].Coefficient[AIDXv,:], 2))
            # loss+=0.5*torch.sum(torch.pow(self.SFEXP[v].Coefficient, 2))
            loss+=0.5*torch.norm(self.SFEXP[v].Coefficient,1)
            loss+=F.mse_loss(self.Decoder[v](SRH),X[v].float(),reduction="sum")
          
                    

        # for i in range(view_num):
        #     # loss+=torch.sum((torch.cdist(Hv[i],Hv[i])**2)*self.W[i,i,i])
        #     HLH=torch.mm(Hv[i].t(),torch.mm(self.L[i],Hv[i]))
        #     # hw 0.5
        #     loss+=torch.trace(HLH)*0.5
        #     for j in range(view_num):
        #         W=self.W[i,j]
        #         W=W/torch.sum(W,1)
        #         loss+=torch.norm(torch.mm(W,Hv[i])-Hv[j])

        return loss


    def update_params(self):
        for opt in self.Opt:
            opt.step()
        for opt in self.Opt:
            opt.zero_grad()
        

    def getH(self):
        Hv=dict()
        for v in range(self.view_num):
            Hv[v]=self.SFEXP[v].Coefficient
        return Hv
    
    def getLH(self):
        return self.comL



def IsnanMat(mat):
    # 实用小函数，判断矩阵是否有nan值
    return torch.any(torch.isnan(mat))