from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class Wrap_Dataset(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.y = torch.Tensor(y).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.y = torch.Tensor(y).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y) 
    

class Wrap_Dataset2(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, X_usr, duration ,use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.X_usr = torch.LongTensor(X_usr).cuda()
            self.y = torch.Tensor(y).cuda()
            self.duration = torch.Tensor(duration).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.X_usr = torch.LongTensor(X_usr).cpu()
            self.y = torch.Tensor(y).cpu()
            self.duration = torch.Tensor(duration).cpu()

        # if use_cuda:
        #     self.X = torch.tensor(X).cuda()
        #     self.X_usr = torch.tensor(X_usr).cuda()
        #     self.y = torch.tensor(y).cuda()
        #     self.duration = torch.tensor(duration).cuda()
        # else:
        #     self.X = torch.tensor(X).cpu()
        #     self.X_usr = torch.tensor(X_usr).cpu()
        #     self.y = torch.tensor(y).cpu()
        #     self.duration = torch.tensor(duration).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index], self.X_usr[index], self.duration[index]

    def __len__(self):
        return len(self.y) 


class Wrap_Dataset3(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, weight ,use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.y = torch.Tensor(y).cuda()
            self.weight = torch.Tensor(weight).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.y = torch.Tensor(y).cpu()
            self.weight = torch.Tensor(weight).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index], self.weight[index]

    def __len__(self):
        return len(self.y) 