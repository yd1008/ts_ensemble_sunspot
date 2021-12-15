#!/bin/bash python
import torch 
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 256, num_layers = 1, dropout = 0.1,bidirectional = False):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size = hidden_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, input_size)

        self.init_state()

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.linear1(x)
        x = self.relu(x)
        x, h = self.gru(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x.permute(1,0,2)

    def init_state(self):
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)