import torch
from torch import nn
import numpy as np

class Linear(torch.nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()

        self.in_features=in_features
        self.out_features=out_features

        self.weight=nn.Parameter(torch.empty(self.in_features,self.out_features,dtype=torch.float32))
        self.bies=nn.Parameter(torch.empty(self.out_features,dtype=torch.float32))

    def reset_params(self):
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bies)

    def forward(self,x):
        return x@self.weight+self.bies


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 1/(1+torch.exp(-x))


class LogisticModel(torch.nn.Module):
    def __init__(self,input_shape: int = 14):
        super().__init__()

        self.sigmoid=Sigmoid()
        self.linear=Linear(input_shape,1)
        self.linear.reset_params()

    def forward(self, x):
        x=self.linear(x)
        y=self.sigmoid(x)
        return y

