import mat73
import numpy as np
import scipy.io as sio
import torch
import random
from alignment import *
from find_anchor import *
from draw_pic import *
from contrastive_graph import congraphresult
def load_data(dataset, test_prop,a):
    all_data = []
    all_label=[]
    label = []
    if dataset=='Caltech101_7':
        path = './datasets/' + dataset + '.mat'  # 路径
        mat = mat73.loadmat(path)  # 加载mat文件
    else:
        mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'HandWritten':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == '3Sources':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'ALOI':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'BBCsports':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'ORL_mtv':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'Caltech101_7':
        data = mat['data'][3:5]
        data[0], data[1] = np.squeeze(data[0]), np.squeeze(data[1])
        data[0],data[1] = np.array(data[0]),np.array(data[1])
        label = np.squeeze(mat['labels'])
    elif dataset == 'Reuters':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == '20NewsGroups':
        data = mat['data'][0][1:3]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == '100leaves':
        mat['data'][0][0],mat['data'][0][1]=mat['data'][0][0].T,mat['data'][0][1].T
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == 'BBC4':
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
        # print(label)
    elif dataset == 'MSRCv1':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == 'BDGP':
        mat['X'][0][0],mat['X'][0][1]=mat['X'][0][0].T,mat['X'][0][1].T
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'HandWritten':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'yale_mtv':
        mat['X'][0][0], mat['X'][0][1] = mat['X'][0][0].T, mat['X'][0][1].T
        data = mat['X'][0][0:2]
        # print((data))
        label = np.squeeze(mat['gt'])
    elif dataset == 'Wikipedia-test':
        data = mat['X'][0:2][0:2]
        data=np.squeeze(data.T)
        # print(data)
        label = np.squeeze(mat['y'])
    elif dataset == 'Movies':
        data = mat['X'][0:2][0:2]
        data=np.squeeze(data.T)
        # print(data)
        label = np.squeeze(mat['y'])
    elif dataset == 'Prokaryotic':
        value1=mat['X'][0][0]
        value2=mat['X'][2][0]
        data = [value1, value2]
        # print(data)
        label = np.squeeze(mat['y'])
    l=[]
    for i in range(0,len(data)):
        l.append(data[i].shape[1])

    divide_seed = random.randint(1, 1000)

    map_idx, unmap_idx = TT_split(len(label), test_prop, divide_seed)
    map_label, unmap_label = label[map_idx], label[unmap_idx]
    # for v in range(len(data)):
    #     AWi = find_k_nearst_AnchorCos(ViewW[v][:,map_idx], len(label))
    map_X, map_Y, unmap_X, unmap_Y = data[0][map_idx], data[1][map_idx], data[0][unmap_idx], data[1][unmap_idx]
    shuffle_idx = random.sample(range(len(unmap_Y)), len(unmap_Y))
    unmap_Y = unmap_Y[shuffle_idx]

    unmaplabel_X, unmaplabel_Y = unmap_label, unmap_label[shuffle_idx]
    all_data.append(np.concatenate((map_X, unmap_X)))
    all_data.append(np.concatenate((map_Y, unmap_Y)))
    all_label_X = np.concatenate((map_label, unmaplabel_X))
    all_label_Y = np.concatenate((map_label, unmaplabel_Y))
    map_X, map_Y, unmap_X, unmap_Y = torch.tensor(map_X,dtype=torch.float64), torch.tensor(map_Y,dtype=torch.float64), torch.tensor(unmap_X,dtype=torch.float64), torch.tensor(
        unmap_Y,dtype=torch.float64)

    S_cosX = cosineSimilartydis(map_X, unmap_X)#51*118 align*unalign
    S_cosY = cosineSimilartydis(map_Y, unmap_Y)
    S_X= find_nanchor(map_X, unmap_X,a)
    S_Y = find_nanchor(map_Y, unmap_Y,a)#51*118

    # draw_grey(S_Y,S_Y.shape[1],S_Y.shape[0])
    #对齐*不对齐
    unmapach_X, unmapach_Y = S_X.T, S_Y.T #118*51

    # unmapach_X,unmapach_Y=unmap_X,unmap_Y
    viewnum=len(data)
    label_num=len(label)
    cluster_num=len(np.unique(label))
    return map_X,map_Y,unmapach_X,unmapach_Y,S_cosX,S_cosY,map_label, unmaplabel_X, unmaplabel_Y, map_idx, viewnum,all_data, all_label_X,all_label_Y,label_num,cluster_num

def Anchor_S(map_X,unmap_X):
    '''Anchor representation'''
    #不开根号比较好
    #直接就是每个维度减，绝对值相加。
    z = map_X.unsqueeze(1) - unmap_X.unsqueeze(0)
    z = torch.abs(z)
    pair_dist = torch.sum(z, 2, False)
    return pair_dist


def loader(train_bs, test_prop, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param test_prop: known aligned proportions for training MvCLN
    :param data_idx: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, \
    divide_seed = load_data(dataset, test_prop)
    # train_pair_dataset = GetDataset(train_pairs, train_pair_labels, train_pair_real_labels)
    # all_dataset = GetAllDataset(all_data, all_label, all_label_X, all_label_Y)
    #
    # train_pair_loader = DataLoader(
    #     train_pair_dataset,
    #     batch_size=train_bs,
    #     shuffle=True,
    #     drop_last=True
    # )
    # all_loader = DataLoader(
    #     all_dataset,
    #     batch_size=1024,
    #     shuffle=True
    # )
    # return train_pair_loader, all_loader, divide_seed


def normalize(x):
    x = (x - np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0) - np.min(x, axis=0)),
                                                                    (x.shape[0], 1))
    return x


def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil( test_prop * n_all).astype(int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor((1-test_prop) * n_all).astype(int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx
