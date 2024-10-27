import torch
def Smask(S):

    SW=1-S.clone()
    SW[SW==1]=0
    print(SW)
    all_columns_lt_1 = torch.all(SW < 0.5, dim=0)
    SW[:, all_columns_lt_1] = 0
    S[:, all_columns_lt_1]=0
    count_ones = torch.sum(all_columns_lt_1 == True).item()
    print(count_ones/S.shape[1])
    return S