#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from informer import Informer
from utils import *


def process_one_batch(batch_x, batch_y):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]]).float().to(device)
        dec_inp = torch.cat([batch_y[:,:(window_size-1),:], dec_inp], dim=1).float().to(device)
        outputs = model(batch_x, dec_inp)

        return outputs, batch_y




if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--num_future_preds", type=int)        
    parser.add_argument("--pre_trained_file_name")
    args = parser.parse_args()

    num_future_preds = args.num_future_preds
    pre_trained_file_name = args.pre_trained_file_name

    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)

    root_dir = '' #specify where results will be saved
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    print('pytorch version: ', torch.__version__)
    train_proportion = 0.7
    test_proportion = 0
    val_proportion = 0.3


    window_size = 192
    batch_size = 128
    train_val_loader, train_loader, val_loader, test_loader,scaler= get_data_loaders(train_proportion, test_proportion, val_proportion,\
        window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 1, pin_memory = True, test_mode = True)

    model = torch.load(pre_trained_file_name)
    
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
            if i == len(test_loader.dataset)-1:
                print(f'this line should be run once')
                enc_in = data
                dec_in = targets
                future_rollout = targets
                enc_in, dec_in = enc_in.to(device), dec_in.to(device)
                output = model(enc_in, dec_in)
                future_rollout = torch.cat([future_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
                future_result = torch.cat((future_result, output[:,-1,:].view(-1).detach().cpu()), 0)

        for _ in range(num_future_preds): ### number of forecast steps
  
            enc_in = future_rollout[:,-window_size:,:]
            dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
            dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()

            enc_in, dec_in = enc_in.to(device), dec_in.to(device)
            output = model(enc_in, dec_in)

            future_rollout = torch.cat([future_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            future_result = torch.cat((future_result, output[:,-1,:].view(-1).detach().cpu()), 0)
    
            
    ### Plot prediction
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(future_result,label='future_forecast')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + '/figs/informer_future_pred.png')
    plt.close(fig)

### Check MSE, MAE
    future_result = future_result.numpy()
    future_result = scaler.inverse_transform(future_result)


    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(future_result,label='future_forecast')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + '/figs/informer_future_inversed_pred.png')
    plt.close(fig)
### Save model result
    

    future_result_df = pd.DataFrame(future_result)
    future_result_df.to_csv(root_dir + '/informer_future_prediction.csv')

