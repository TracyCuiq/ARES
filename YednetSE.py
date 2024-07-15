
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# 30 SRM filtes
from srm_filter_kernel import all_normalized_hpf_list
# Global covariance pooling
from MPNCOV import *  # MPNCOV


# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output

# Pre-processing Module
class HPF(nn.Module):
    def __init__(self, in_channels=1, out_channels=30):
        super(HPF, self).__init__()
        # Load 30 SRM Filters
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)
        all_hpf_array_5x5 = np.array(all_hpf_list_5x5)
        hpf_weight = nn.Parameter(torch.tensor(all_hpf_array_5x5, dtype=torch.float32).view(30, 1, 5, 5), requires_grad=True)
        #hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)
        return output

class HPF_W(nn.Module):
    def __init__(self, in_channels=1, out_channels=30):
        super(HPF_W, self).__init__()
        # Load 30 SRM Filters
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)
        all_hpf_array_5x5 = np.array(all_hpf_list_5x5)
        hpf_weight = nn.Parameter(torch.tensor(all_hpf_array_5x5, dtype=torch.float32).view(30, 1, 5, 5), requires_grad=True)
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):
        output = self.hpf(input)
        #output = self.tlu(output)
        return output

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            #nn.ReLU(inplace=True)
        )
        #nn.init.xavier_uniform_(self.fc.weight, gain=1.0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class YedNetSE(nn.Module):
    def __init__(self):
        super(YedNetSE, self).__init__()

        self.se_1 = SELayer(32*2, )
        #self.se_2 = SELayer(32, )
        #self.se_3 = SELayer(64, )
        #self.se_4 = SELayer(128, )
        #self.se_5 = SELayer(256, )
        self.hpf_w1 = HPF_W()
        self.hpf1 = HPF()

        self.avgp = nn.AvgPool2d(8, stride=8)

        #self.group1 = HPF()

        self.group2_1 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group2_2 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #############
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        

        self.fc1 = nn.Linear(int((256+30)* ((256+30) + 1) / 2), 2)

    def forward(self, input):
        output = input
        h1 = self.hpf_w1(output)
        h2 = self.hpf1(output)

        #output = self.group1(output)#B,30,256,256
        h2_l = self.avgp(h2)
        #print(h2_l.shape)
        h1 = self.group2_1(h1)
        h2 = self.group2_2(h2)
        h = torch.cat((h1, h2), dim=1)
        output = self.se_1(h)

        #output = self.group2(output)
        #output = self.se_2(output)
        #output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)
        #output = self.group6(output)

        #print(output.shape)
        output = torch.cat((output, h2_l), dim=1)
        #print(output.shape)

        # Global covariance pooling
        output = CovpoolLayer(output)
        #print(output.shape)
        output = SqrtmLayer(output, 5)
        output = TriuvecLayer(output)
        
        output = output.view(output.size(0), -1)
        output = self.fc1(output)

        return output

# Initialization
def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)

