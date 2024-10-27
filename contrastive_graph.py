import numpy as np
import torch

from alignment import *
def congraphresult(cmapX,cmapY,cunmapX,cunmapY,numalign):

    N=cmapX.shape[0]
    print(N,'n')
    unit_tensor = torch.eye(N)
    condataX = np.concatenate((cmapX, cunmapX, cmapY))

    condataY = np.concatenate((cmapY, cunmapY, cmapX))
    condataX, condataY = torch.tensor(condataX, dtype=torch.float64), torch.tensor(condataY, dtype=torch.float64)

    result = cosineSimilartyn(condataX, condataY)
    result[:N, :N] = unit_tensor
    result[-N:,-N:]=unit_tensor

    numcon_graph = result.shape[0]

    i, j = int(numalign), int(numcon_graph - numalign)
    x=result[i:j,i:j]
    x=x.T


    return result, condataX, condataY

def congraphresultpast(cmapX,cmapY,cunmapX,cunmapY):
    '''直接相乘会造成原本对齐的不是最相似的，因为加上锚点用相似度度量，用了两次相似度度量，误差会比较大'''
    # cunmapX,cunmapY=cunmapX.T,cunmapY.T
    condataX=np.concatenate((cmapX, cunmapX, cmapY))

    condataY = np.concatenate((cmapY, cunmapY, cmapX))
    condataX,condataY=torch.tensor(condataX, dtype=torch.float64),torch.tensor(condataY,dtype=torch.float64)
    result=cosineSimilarty(condataX,condataY)
    # print(condataY[0],'condata')
    return result,condataX,condataY

def congraphpair(con_gph,numalign,hk):
    numcon_graph=con_gph.shape[0]

    i,j=int(numalign),int(numcon_graph-numalign)
    con_graphX=con_gph[i:j,:]
    con_graphT=con_gph.T
    con_graphY=con_graphT[i:j,:]
    # print(np.argmin(con_graphY[0]))
    # print(min(con_graphY[0]),'sssd')

    print(type(con_graphX),'con_gra')
    max_values_range1_index = torch.topk(con_graphX[:, 0:i], k=1,dim=1)[1]
    max_values_range2_index = torch.topk(con_graphX[:, i:j], k=1, dim=1)[1]
    max_values_range3_index = torch.topk(con_graphX[:, j:numcon_graph], k=1, dim=1)[1]
    max_values_range1_index,max_values_range2_index,max_values_range3_index=max_values_range1_index+0,max_values_range2_index+i,max_values_range3_index+j
    combined_arrayposX = torch.cat((max_values_range1_index, max_values_range2_index, max_values_range3_index), dim=1)

    min_values_range1_index = torch.topk(con_graphX[:, 0:i], k=1, dim=1, largest=False)[1]
    min_values_range2_index = torch.topk(con_graphX[:, i:j], k=hk, dim=1, largest=False)[1]
    min_values_range3_index = torch.topk(con_graphX[:, j:numcon_graph], k=1, dim=1, largest=False)[1]
    min_values_range1_index, min_values_range2_index, min_values_range3_index = min_values_range1_index + 0, min_values_range2_index + i, min_values_range3_index + j
    # 合并三个数组
    combined_arraynegX = torch.cat((min_values_range1_index, min_values_range2_index, min_values_range3_index),dim=1)



    max_values_range4_index = torch.topk(con_graphY[:, 0:i], k=1,dim=1)[1]
    max_values_range5_index = torch.topk(con_graphY[:, i:j], k=1, dim=1)[1]
    max_values_range6_index = torch.topk(con_graphY[:, j:numcon_graph], k=1,dim=1)[1]
    max_values_range4_index, max_values_range5_index, max_values_range6_index = max_values_range4_index + 0, max_values_range5_index + i, max_values_range6_index + j
    combined_arrayposY = torch.cat((max_values_range4_index,max_values_range5_index,max_values_range6_index), dim=1)


    min_values_range4_index = torch.topk(con_graphY[:, 0:i], k=1, dim=1, largest=False)[1]
    min_values_range5_index = torch.topk(con_graphY[:, i:j], k=hk, dim=1, largest=False)[1]
    min_values_range6_index = torch.topk(con_graphY[:, j:numcon_graph], k=1, dim=1, largest=False)[1]
    min_values_range4_index, min_values_range5_index, min_values_range6_index = min_values_range4_index + 0, min_values_range5_index + i, min_values_range6_index + j
    # 合并三个数组
    combined_arraynegY = torch.cat((min_values_range4_index,min_values_range5_index,min_values_range6_index), dim=1)
    # for i in range(1,10):
    #     print(con_graphY[i,combined_arrayposY[i]])
    #     print(con_graphY[i,combined_arraynegY[i]])

    return combined_arrayposX,combined_arrayposY,combined_arraynegX,combined_arraynegY

def condatapairs(condataX,condataY,combined_arrayposX,combined_arrayposY,combined_arraynegX,combined_arraynegY):

    # dataposX = condataY[combined_arrayposX,:]
    # dataposX2=condataX[combined_arrayposX,:]
    # dataposX2=dataposX2[:, [0, 2], :]
    # dataposX=torch.cat((dataposX, dataposX2), dim=1)
    #
    # datanegX = condataY[combined_arraynegX,:]
    # datanegX2=condataX[combined_arraynegX,:]
    # datanegX2=datanegX2[:, [0, 2], :]
    # datanegX=torch.cat((datanegX, datanegX2), dim=1)
    #
    # dataposY = condataX[combined_arrayposY,:]
    # dataposY2=condataY[combined_arrayposY,:]
    # dataposY2=dataposY2[:, [0, 2], :]
    # dataposY=torch.cat((dataposY, dataposY2), dim=1)
    #
    # datanegY = condataX[combined_arraynegY,:]
    # datanegY2=condataY[combined_arraynegY,:]
    # datanegY2=datanegY2[:, [0, 2], :]
    # datanegY=torch.cat((datanegY, datanegY2), dim=1)
    dataposX = condataY[combined_arrayposX, :]
    datanegX = condataY[combined_arraynegX, :]
    dataposY = condataX[combined_arrayposY, :]
    datanegY = condataX[combined_arraynegY, :]
    return dataposX,datanegX,dataposY,datanegY
