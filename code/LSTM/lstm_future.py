#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
from lstm import LSTM
from utils import *


class early_stopping():
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_loss = None
        self.counter = 0
        self.best_model = None
    
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss-self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model
            torch.save(model, 'best_inf.pth')
            print(f'Saving best model')
        else:
            self.counter += 1 
            if self.counter == self.patience:
                self.early_stop = True
                print('Early stopping')
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----')
    
  
    


if __name__ == "__main__":
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)

    root_dir = '/scratch/yd1008/sunspot_informer/LSTM/tune_results'
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    print('pytorch version: ', torch.__version__)
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2

    window_size = 192
    batch_size = 128
    train_loader, val_loader, test_loader, scaler = get_data_loaders(train_proportion, test_proportion, val_proportion,\
        window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 1, pin_memory = False, test_mode = True)

    model = torch.load('best_lstm.pth')

### Predict
    model.eval()
    future_rollout = torch.Tensor(0)   
    future_result = torch.Tensor(0)  
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i == 0:
                data_in = data
                future_rollout = targets
            else:
                data_in = future_rollout[:,-window_size:,:]
            data_in = data_in.to(device)
            output = model(data_in)
            future_rollout = torch.cat([future_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            future_result = torch.cat((future_result, output[:,-1,:].view(-1).detach().cpu()), 0)


        for _ in range(240): ### number of forecast steps
  
            data_in = future_rollout[:,-window_size:,:]

            data_in = data_in.to(device)
            output = model(data_in)

            future_rollout = torch.cat([future_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            future_result = torch.cat((future_result, output[:,-1,:].view(-1).detach().cpu()), 0)
    
            
    ### Plot prediction
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(future_result,label='future_forecast')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + '/figs/lstm_future_pred.png')
    plt.close(fig)

### Check MSE, MAE
    future_result = future_result.numpy()
    future_result = scaler.inverse_transform(future_result)


    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(future_result,label='future_forecast')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + '/figs/lstm_future_inversed_pred.png')
    plt.close(fig)
### Save model result


    future_result_df = pd.DataFrame(future_result)
    future_result_df.to_csv(root_dir + '/lstm_future_prediction.csv')

