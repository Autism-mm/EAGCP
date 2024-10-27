import argparse
from config import *
import time
import logging
import torch, gc
import numpy as np
import matplotlib.pyplot as plt
from data_loader import *
from alignment import *
from GCN_model import *
from Clustering import *
from contrastive_graph import congraphpair,condatapairs
parser = argparse.ArgumentParser(description='HC-inout in PyTorch')
parser.add_argument('--data', default='14', type=int,
                    help='choice of dataset, 0-HW,1-3Sources,2BBC,3-Scene15, 4-Caltech101,5-ORL_mtv,6-Caltech_7,7-Reuters,'
                         '8-20newsgroups,9-100leaves,10-BBC4,11-MSRCv1,12-BDGP,13-HandWritten,14-yale_mtv，15-Wikipedia-test,16-Movies,17-,18-ALOI')
parser.add_argument('-bs', '--batch-size', default='64', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='200', type=int, help='number of epochs to run')
parser.add_argument('-ap', '--aligned-prop', default='0.5', type=float,
                    help='originally aligned proportions in the partially view-aligned data')
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx to use.')
parser.add_argument('-lr', '--learn-rate', default='0.0002', type=float, help='learning rate of adam')

parser.add_argument('-nn', '--neighbor-num', default='6', type=int, help='Number of neighbors')
parser.add_argument('-s', '--start-fine', default=False, type=bool, help='flag to start use robust loss or not')
parser.add_argument('-m', '--margin', default='6', type=int, help='initial margin')
parser.add_argument('-aw', '--aw_enhanced', default='20', type=int, help='enhanced_graph weight')
class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_distX,pair_distY, P, margin, h0, h3,i,args):
        # 这个函数的计算是否之后模型输入的特征相关，还是已经无关了
        if i !=0:
            pair_distX, pair_distY = pairdis(pair_distX.data, pair_distY.data, h0, h3)

        dist_sqX = pair_distX ** 2
        dist_sqY=pair_distY**2
        P = P.to(torch.float32)

        N = P.shape[1]
        # N2=int(N/2)
        # print(P[:,3:])
        loss1 = P[:,:3] * dist_sqX[:,:3] + P[:,:3] * dist_sqY[:,:3]
        loss2=(1-P[:,3:]) * torch.pow(torch.clamp(margin - pair_distX[:,3:], min=0.0), 2)+(1-P[:,3:]) * torch.pow(torch.clamp(margin - pair_distY[:,3:], min=0.0), 2)
        # loss1 = P[:,:5] * dist_sqX[:,:5] + P[:,:5] * dist_sqY[:,:5]
        # loss2=(1-P[:,5:]) * torch.pow(torch.clamp(margin - pair_distX[:,5:], min=0.0), 2)+(1-P[:,5:]) * torch.pow(torch.clamp(margin - pair_distY[:,5:], min=0.0), 2)
        loss = (torch.sum(loss1)+torch.sum(loss2))/P.shape[0]
        # print(P.shape[0],'P')
        # print(int(loss))
        return loss
def train(align_out,kn,model, criterion,criterion1, optimizer,pair_distX, pair_distY,tensor_matrixlabel,numalign,i, args):

    if i % 10 == 0:
        logging.info("=======> Train epoch: {}/{}".format(i, args.epochs))
    model.train()

    time0 = time.time()
    loss_value = 0

        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
    for v in range(2):
        align_out[v],kn[v] =align_out[v].to(args.gpu), kn[v].to(args.gpu)
    h0, h1,h3,h4 = model(kn[0],align_out[0],kn[1],align_out[1])

    loss1 = criterion(align_out[0], h1)
    loss2= criterion(align_out[1], h4)
    loss3=criterion(h0,h3)
    #--------------------------
    coX, coY = cosineSimilartydis(h1, h1), cosineSimilartydis(h3, h3)
    hadX, hadY = find_k_nearst_AnchorCos(coX, 10), find_k_nearst_AnchorCos(coY,10)
    #----------------------------------------------
    loss4=torch.norm(hadX,p=2)+torch.norm(hadY,p=2)
    # print(loss4)
    loss5 = loss1 + loss2+loss3
    loss5 = loss5 / (align_out[0].shape[0] * 2)
    # loss7=criterion(hadX,hadY)
    # print(torch.sum(pair_distX)/84,pair_distY)
    # print(np.shape(pair_distX),'kkk')

    # print(args.margin)
    loss6 = criterion1(pair_distX, pair_distY,tensor_matrixlabel, args.margin,h0, h3,i, args)
    # loss6_detached = loss6.detach().requires_grad_(True)
    loss=loss5+loss6
    # loss6 = kn[0].mean()
    # print(loss6,"model-loss")

    loss_value += loss.item()
    # print(int(loss),'loss')
    if i!=0:
        optimizer.zero_grad()
        # loss1 = loss.detach_().requires_grad_(True)
        # loss1.backward()
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - time0
    return h0,h1,h3,h4,pair_distX, pair_distY,epoch_time,loss
def main():
    args = parser.parse_args()
    for j in np.arange(5, 41, 0.5):
        # print(j,'j')
        args.aw_enhanced = j
        ViewW = dict()
        AW=dict()
        align_out=[]
        kn=[]
        args = parser.parse_args()
        data_name = ['HandWritten', '3Sources', 'BBCsports', 'Scene15', 'Caltech101', 'ORL_mtv', 'Caltech101_7', 'Reuters',
                     '20NewsGroups','100leaves','BBC4','MSRCv1','BDGP','HandWritten','yale_mtv','Wikipedia-test','Movies','Prokaryotic','ALOI']
        NetSeed = random.randint(1, 1000)
        # NetSeed = random.seed()
        np.random.seed(NetSeed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(NetSeed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(NetSeed)  # 为当前GPU设置随机种子
        # print(data_name[args.data])
        (map_X,map_Y,unmapach_X,unmapach_Y,S_cosX,S_cosY,map_label, unmaplabel_X, unmaplabel_Y, map_idx, viewnum,all_data,
         all_label_X,all_label_Y,label_num,cluster_num) = load_data(data_name[args.data], args.aligned_prop,args.aw_enhanced)
        #unmapach_X和S_cosX都余弦距离表示的
        # ViewW[0],ViewW[1]= cosineSimilarty(S_cosX.T, S_cosX.T),cosineSimilarty(S_cosY.T, S_cosY.T)

        #align_outX, align_outY, alignS_cosX, alignS_cosY都是余弦距离表示
        S_unmap, align_outX, align_outY, alignS_cosX, alignS_cosY, acc = mapins(unmapach_X, unmapach_Y, S_cosX.T, S_cosY.T,
                                                                      unmaplabel_X, unmaplabel_Y, args.gpu)

    # alignS_cosX, alignS_cosY=torch.tensor(alignS_cosX),torch.tensor(alignS_cosY)
    align_outX, align_outY = torch.tensor(align_outX), torch.tensor(align_outY)
    # cmapX, cmapY, cunmapX, cunmapY = (cosineSimilartydis(map_X, map_X), cosineSimilartydis(map_Y, map_Y),alignS_cosX,alignS_cosY)
    cmapX, cmapY, cunmapX, cunmapY = (
    cosineSimilartydis(map_X, map_X), cosineSimilartydis(map_Y, map_Y), align_outX, align_outY)
    # -------------------------------------------------------------------------------------------------------------------------------------------
    numalign=map_X.shape[0]

    con_graph,condataX,condataY = congraphresult(cmapX, cmapY, cunmapX, cunmapY,numalign)#con_graph是余弦相似度


    # hk = np.size(np.unique(unmaplabel_X))
    hk=10
    # print(hk,'hk')
    combined_arrayposX,combined_arrayposY,combined_arraynegX,combined_arraynegY = congraphpair(con_graph,numalign,hk)

    # print(np.shape(combined_arrayposX),np.shape(combined_arraynegX))
    dataposX,datanegX,dataposY,datanegY=condatapairs(condataX, condataY, combined_arrayposX, combined_arrayposY, combined_arraynegX,
                 combined_arraynegY)
    # print(np.shape(align_outX))
    #------------------------------------------------------------------------------------------------
    datapairX=torch.zeros(dataposX.shape[0], datanegX.shape[1]+3,dataposX.shape[2])
    datapairY = torch.zeros(dataposY.shape[0], datanegY.shape[1]+3, dataposY.shape[2])
    datapairX[:, :3, :] = dataposX
    datapairX[:, 3:, :] = datanegX
    datapairY[:, :3, :] = dataposY
    datapairY[:, 3:, :] = datanegY
    expanded_tensorX, expanded_tensorY = align_outX.clone(), align_outY.clone()
    # expanded_tensorX, expanded_tensorY =torch.tensor(expanded_tensorX),torch.tensor(expanded_tensorY)
    expanded_tensorX, expanded_tensorY =expanded_tensorX.unsqueeze(1),expanded_tensorY.unsqueeze(1)
    cosine_simX = torch.nn.functional.cosine_similarity(expanded_tensorX, datapairX, dim=2)
    cosine_simY = torch.nn.functional.cosine_similarity(expanded_tensorY, datapairY, dim=2)
    disX=torch.norm(expanded_tensorX-datapairX, dim=2)
    disY = torch.norm(expanded_tensorY - datapairY, dim=2)

    # print(cosine_simY[0:10])
    tensor_matrixlabel = torch.zeros(cosine_simX.shape[0], datanegY.shape[1]+3)
    tensor_matrixlabel[:, :3] = 1
    #-------------------------------------------------------------------------------------
    reconX = np.concatenate((cmapX, cunmapX))
    reconY = np.concatenate((cmapY, cunmapY))
    reconX, reconY = torch.from_numpy(reconX), torch.from_numpy(reconY)
    reconX, reconY = reconX.to(args.gpu), reconY.to(args.gpu)
    #------------------------------------------------------------------------------------------
    align_outX, align_outY=np.array(align_outX, dtype=np.float32),np.array(align_outY, dtype=np.float32)
    align_outX, align_outY=torch.from_numpy(align_outX),torch.from_numpy(align_outY)
    align_outX, align_outY=align_outX.to(args.gpu), align_outY.to(args.gpu)
    ViewW[0], ViewW[1] = cosineSimilartydis(align_outX, align_outX), cosineSimilartydis(align_outY, align_outY)
    # ViewW[0], ViewW[1] = cosineSimilarty(map_X, map_X), cosineSimilarty(map_Y, map_Y)
    adjacency_matrixX, adjacency_matrixY = find_k_nearst_AnchorCos(ViewW[0],args.neighbor_num), find_k_nearst_AnchorCos(ViewW[1],args.neighbor_num)  # k近邻
    # adjacency_matrixX,adjacency_matrixY=np.where(S_cosX>0.9,1,0),np.where(S_cosY>0.9,1,0)#大于阈值设为1
# View0,View1 = cosineSimilarty(align_out0, align_out0),cosineSimilarty(align_out1, align_out1)
# kn0,kn1=find_k_nearst_AnchorCos(View0,label_num),find_k_nearst_AnchorCos(View1,label_num)

    # align_out.append(map_X)
    # align_out.append(map_Y)
    align_out.append(align_outX)
    align_out.append(align_outY)
    kn.append(adjacency_matrixX)
    kn.append(adjacency_matrixY)

    input_features=S_cosX.shape[0]
    output_features=cluster_num

    model=GCNNet(input_features,output_features)
    model.cuda()
    criterion = nn.MSELoss().to(args.gpu)
    criterion1 = NoiseRobustLoss().to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    CAR_list = []
    acc_list, nmi_list, ari_list,f_scorelist,f_scorelist2,presion_list,presion_list2,reccall_list,purity_list = [], [], [],[],[],[],[],[],[]
    train_time = 0
    # train
    pair_distX, pair_distY = 1 - cosine_simX, 1 - cosine_simY
    pred_label=np.array(unmaplabel_X,dtype=np.float32)
    args.margin = (torch.sum(pair_distX) + torch.sum(pair_distY)) / align_out[0].shape[0] * 3
    for i in range(0, args.epochs + 1):
        # if i == 0:
        #     with torch.no_grad():
        #         v0,v1,p0,p1,paX, paY,epoch_time,loss = train(align_out,kn,model, criterion, criterion1,optimizer, pair_distX, pair_distY,tensor_matrixlabel,numalign,i, args)
        #         pair_distX, pair_distY=paX, paY

        v0, v1,p0,p1, paX, paY,epoch_time,loss = train(align_out, kn, model, criterion, criterion1,optimizer, pair_distX, pair_distY,tensor_matrixlabel,numalign,i, args)
        pair_distX, pair_distY = paX, paY
        #test
        v0, v1= v0.to('cpu'),v1.to('cpu')#有问题
#--------------------------------------------------------------------------------------------
    # max_values, _ = torch.max(p0, dim=1, keepdim=True)
    #
    # # 将每行的最大值设置为1，其余设置为0
    # p0[p0 != max_values] = 0
    # p0[p0 == max_values] = 1
    # _, indices = torch.max(p0, dim=1)
    # # 将结果转换为numpy数组
    # indices=indices.to('cpu')
    # max_indices_np = indices.numpy()
    # print(max_indices_np)
#--------------------------------------------------------------------------------------
        data = []
        data.append(v0.detach().numpy())
        data.append(v1.detach().numpy())

        # data=0.5*v0.detach().numpy()+0.5*v1.detach().numpy()
        y_pred, ret,accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity = Clustering(data, pred_label)
        if i % 1 == 0:
            print(accuracy,nmi,ari,f_score,f_score2,precision,precision2,recall,purity)
            logging.info(
                "CAR={}, kmeans: acc={}, nmi={}, ari={}".format(round(acc, 4), ret['ACC'],
                                                                ret['NMI'], ret['ARI'],ret['f_score'],ret['f_score_weighted'],
                                                                ret['precision'],ret['precision_weighted'],ret['recall'],ret['purity']))

        acc_list.append(ret['ACC'])
        nmi_list.append(ret['NMI'])
        ari_list.append(ret['ARI'])
        f_scorelist.append(ret['f_score'])
        f_scorelist2.append(ret['f_score_weighted'])
        presion_list.append(ret['precision'])
        presion_list2.append(ret['precision_weighted'])
        reccall_list.append(ret['recall'])
        purity_list.append(ret['purity'])
        if i==0:
            GT=0
            # ariva=ari
            accva=accuracy
        else:
            # ariopt=ariva
            accopt=accva
            accva=accuracy
            # ariva=ari
            GT=100*(accopt-accva)/accopt
        if i>40 and GT>400:
            break
    print(max(acc_list))
    print(max(nmi_list))
    print(max(ari_list))
    print(max(f_scorelist))
    print(max(f_scorelist2))
    print(max(presion_list))
    print(max(presion_list2))
    print(max(reccall_list))
    print(max(purity_list))
    print(acc)

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(0,3):
        main()
