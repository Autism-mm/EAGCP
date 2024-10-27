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