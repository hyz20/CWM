from turtle import forward
import torch
import numpy as np
import time
from torch import nn
from torch.nn import MarginRankingLoss
from torch.nn import functional as F


class CWMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rel_score, c_inv, wt, duration, sigma):
        rel_true_less = torch.exp(-c_inv/(wt+1))[wt<duration]
        rel_score_less = rel_score[wt<duration]
        rel_true_over = torch.exp(-c_inv/(wt+1))[wt>=duration]
        rel_score_over = rel_score[wt>=duration]
        logsigmoid = nn.LogSigmoid()
        def inv_sigmoid(y):
            d = 1e-6
            y = torch.clamp(y, d, 1 - d)
            x = torch.log(y / (1 - y))
            return x
        loss_less = (torch.square(rel_score_less- inv_sigmoid(rel_true_less))/(2*(sigma**2))).mean()
        loss_over = -logsigmoid((rel_score_over - inv_sigmoid(rel_true_over))/sigma).mean()
        return loss_less + loss_over 

