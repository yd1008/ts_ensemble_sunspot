#!/bin/bash python
import torch 
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 256, num_layers = 1, dropout = 0.1,bidirectional = False):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, input_size)


        self.init_state()

    def forward(self, x):
        x = x.permute(1,0,2)
        x = self.linear1(x)
        x = self.relu(x)
        x, (h,c) = self.lstm(x)
        #print(f'After lstm layer, the x is {x[-1,-1,:]}, h is {h[-1,-1,:]}, c is {c[-1,-1,-1]}')
        x = self.dropout(x)
        x = self.linear2(x)


        #print(f'After output layer, the x is {x[-1,-1,:]} of shape {x.shape}, x_test is {x_test[-1,-1,:]} of shape {x_test.shape}')

        return x.permute(1,0,2)

    def init_state(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)