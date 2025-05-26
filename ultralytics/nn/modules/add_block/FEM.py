import warnings
import numpy
import torch.nn as nn
import torch
from ultralytics.nn.modules import *
from ultralytics.nn.modules.block import *
from ultralytics.nn.modules.Attention import *
from ultralytics.nn.modules.add_block import *
all = ['FEM']

        
class FEM(nn.Module):
    def __init__(self,c1,c2,k=5):
        super(FEM, self).__init__()
        c_= c1 // 2
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c_*6,c2,1,1)
        self.m = nn.MaxPool2d(kernel_size=k,stride=1,padding=k//2)
        self.a = nn.AvgPool2d(kernel_size=k,stride=1,padding=k//2)
        self.am = nn.AdaptiveMaxPool2d(1)
        self.aa = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y11 = self.m(x)+self.a(x)
            y21 = self.m(y1) + self.a(y1)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x,y11,y21,self.m(y2)+self.a(y2),x+self.am(x).expand_as(x),x+self.aa(x).expand_as(x)),1))

