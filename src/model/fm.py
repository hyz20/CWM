import torch
import numpy as np
from torch.nn import ReLU
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class My_FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)
        # return torch.sqrt(torch.square(x.squeeze(1)))

class Usr_FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        # x = self.linear(x)
        # return torch.sigmoid(x.squeeze(1))
        # return x.squeeze(1)
        # if np.random.rand() > 0.996:
        #     with torch.no_grad():
        #         print(x.squeeze(1).min())
        #         print(x.squeeze(1).max())
        #         print(torch.sqrt(torch.square(x.squeeze(1))).min())
        #     # print(usr_score)
        #     # print(usr_true)

        # return torch.clamp(torch.sqrt(torch.square(x.squeeze(1))),0,200)
        return torch.sqrt(torch.square(x.squeeze(1)))
        # return x.squeeze(1)
        # f_relu = ReLU()
        # return f_relu(x.squeeze(1))
        # return 20*torch.tanh(x.squeeze(1)) + 40
        # return 15*torch.sigmoid(x.squeeze(1)) 


class Share_FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

        self.linear_c = FeaturesLinear(field_dims)
        self.fm_c = FactorizationMachine(reduce_sum=True)


    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_m = self.linear(x) + self.fm(self.embedding(x))
        x_c = self.linear_c(x) + self.fm_c(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x_m.squeeze(1), x_c.squeeze(1)
        # return torch.sqrt(torch.square(x.squeeze(1)))
        