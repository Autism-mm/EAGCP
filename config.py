import numpy as np
import torch
from model import *
def pairdis(pair_disX, pair_disY,hf0,hf1):
    resultcos = cosineSimilartyn(hf0,hf1)


    con_graphX = resultcos
    con_graphY = resultcos.T


    _, max_values_range1_index = torch.max(con_graphX, dim=1)
    _, min_values_range1_index = torch.min(con_graphX, dim=1)
    _, max_values_range2_index = torch.max(con_graphY, dim=1)
    _, min_values_range2_index = torch.min(con_graphY, dim=1)
    # print(max_values_range1_index)
    posX,negX,posY,negY=hf1[max_values_range1_index,:],hf1[min_values_range1_index,:],hf0[max_values_range2_index,:],hf0[min_values_range2_index,:]

    disXp,disXn=torch.nn.functional.cosine_similarity(hf0,posX),torch.nn.functional.cosine_similarity(hf0,negX)
    disYp, disYn = torch.nn.functional.cosine_similarity(hf1, posY), torch.nn.functional.cosine_similarity(hf1, negY)
    # print(disYp,disYn,disXn,disXp)
    pair_disX[:,1], pair_disX[:,4]=disXp,disXn
    pair_disY[:, 1], pair_disY[:, 4]=disYp, disYn


    return pair_disX, pair_disY

def pairdis_only_negative(pair_disX, pair_disY, hf0, hf1):
    """
    只动态更新负对（最不相似的对），正对保持不变。
    pair_disX, pair_disY: 原有pair距离矩阵（会被部分更新）
    hf0, hf1: 当前epoch模型输出的两个模态的嵌入（特征）
    """
    # 计算相似度矩阵
    resultcos = cosineSimilartyn(hf0, hf1)
    con_graphX = resultcos
    con_graphY = resultcos.T

    # 找最不相似的对（负对）
    _, min_values_range1_index = torch.min(con_graphX, dim=1)
    _, min_values_range2_index = torch.min(con_graphY, dim=1)
    negX = hf1[min_values_range1_index, :]
    negY = hf0[min_values_range2_index, :]
    disXn = torch.nn.functional.cosine_similarity(hf0, negX)
    disYn = torch.nn.functional.cosine_similarity(hf1, negY)

    # 只更新负对（第4列），正对（第1列）保持不变
    pair_disX_new = pair_disX.clone()
    pair_disY_new = pair_disY.clone()
    pair_disX_new[:, 4] = disXn
    pair_disY_new[:, 4] = disYn

    return pair_disX_new, pair_disY_new