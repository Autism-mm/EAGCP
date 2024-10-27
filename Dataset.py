from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
class GetAllDataset(Dataset):
    def __init__(self, adjacency,data ):
        self.adjacency=adjacency
        self.data = data

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][:, index]).float(), \
                           torch.from_numpy(self.data[1][:, index]).float(), \

        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        return fea0, fea1, label, class_labels0, class_labels1

    def __len__(self):
        return len(self.labels)