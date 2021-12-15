#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from transformer import Tranformer
from utils import *


def process_one_batch(batch_x, batch_y):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]]).float().to(device)
        dec_inp = torch.cat([batch_y[:,:(window_size-1),:], dec_inp], dim=1).float().to(device)
        # encoder - decoder
        outputs = model(batch_x, dec_inp)

        return outputs, batch_y



# def evaluate(model,data_loader,criterion):
#     model.eval()
#     total_loss = 0.
#     rmse = 0.
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#     with torch.no_grad():
#         for (data,targets) in data_loader:
#             #data, targets = data.to(device), targets.to(device)
#             output, targets = process_one_batch(data,targets)
#             total_loss += criterion(output, targets).detach().cpu().numpy()
#     return total_loss

def evaluate(model,data_loader,criterion):
    model.eval()    
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    total_loss = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(data_loader):
            # if i == 0:
            #     enc_in = data
            #     dec_in = targets
            #     test_rollout = targets
            # else:
            #     enc_in = test_rollout[:,-window_size:,:]
            #     dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
            #     dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
            #     #dec_in = test_rollout[:,-window_size:,:]
            # enc_in, dec_in, targets = enc_in.to(device), dec_in.to(device), targets.to(device)
            # output = model(enc_in, dec_in)
            #print(f'i = {i}, Enc_in: {enc_in}, Dec_in: {dec_in}')
            enc_in, dec_in, targets = data.to(device), targets.to(device), targets.to(device)
            output, _ = process_one_batch(enc_in,dec_in)
            #print(f'Output: {output}, targets: {targets}')
            total_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()
            test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            #test_rollout = torch.cat([test_rollout,targets[:,-1:,:].detach().cpu()],dim = 1)

    return total_loss, test_rollout[:,-10:,:]

def predict_model(model, test_loader, window_size, epoch, plot=True):
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(test_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                test_rollout = targets
            else:
                enc_in = test_rollout[:,-window_size:,:]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
                #dec_in = enc_in[:,:(window_size-1),:]
            enc_in, dec_in, targets = enc_in.to(device), dec_in.to(device), targets.to(device)
            output = model(enc_in, dec_in)

            #output, _ = process_one_batch(enc_in,dec_in)
            #print(f'output: {output}, targets: {targets}, test_rollout: {test_rollout}')
            test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            #test_rollout = torch.cat([test_rollout,targets[:,-1:,:].detach().cpu()],dim = 1)
            #print('Test rollout shape: ',test_rollout.shape)
            #print(f'Targets:{targets[-1][-1]}, dec_in: {dec_in[-1][-1]}, output: {output[-1][-1]}')
            #print(f'targets: {targets}, test_rollout: {test_rollout}')
            test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            truth = torch.cat((truth, targets[:,-1,:].view(-1).detach().cpu()), 0)
            
    if plot==True:
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        ax.plot(test_result,label='forecast')
        ax.plot(truth,label = 'truth')
        ax.plot(test_result-truth,ls='--',label='residual')
        #ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        fig.savefig(root_dir + f'/figs/transformer_epoch{epoch}_pred.png')
        plt.close(fig)

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
        else:
            self.counter += 1 
            if self.counter == self.patience:
                self.early_stop = True
                print('Early stopping')
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----')
    
  
    


if __name__ == "__main__":
    print(f'Pytorch version {torch.__version__}')
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)

    root_dir = '/scratch/yd1008/sunspot_informer/transformer/tune_results/'
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    #best_config = {'feature_size': 512, 'num_enc_layers': 4, 'num_dec_layers': 2, 'num_head': 4, 'd_ff': 1024, 'dropout': 0.1, 'window_size': 144}
    best_config = {'feature_size': 512, 'num_enc_layers': 3, 'num_dec_layers': 2, 'num_head': 4, 'd_ff': 512, 'dropout': 0.2, 'window_size': 192, 'lr': 5e-6, 'batch_size': 8}
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    feature_size = best_config['feature_size']
    num_enc_layers = best_config['num_enc_layers']
    num_dec_layers = best_config['num_dec_layers']
    d_ff = best_config['d_ff']
    num_head = best_config['num_head']
    dropout = best_config['dropout']
    lr = best_config['lr']
    window_size = best_config['window_size']
    batch_size = best_config['batch_size']

    model = Tranformer(feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
        d_ff = d_ff, dropout=dropout,num_head=num_head)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    print('Using device: ',device)
    model.to(device)
      
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.95)
    writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
        
    train_loader, val_loader, test_loader,scaler = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 8, pin_memory = True, test_mode = True)

    epochs = 800
    train_losses = []
    test_losses = []
    tolerance = 10
    best_test_loss = float('inf')
    Early_Stopping = early_stopping(patience=20)
    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.
        
        for i,(data, targets) in enumerate(train_loader):

   
            data, targets = data.to(device), targets.to(device)
            #print(f'data shape: {data.shape}, targets shape: {targets.shape}')
            optimizer.zero_grad()
            output, truth = process_one_batch(data,targets)
            #print(f'output shape: {output.shape}')
            #output = model(data,targets)
            #print('Targets : ',targets[:,-2:,:])
            #print('Output : ',output[:,-2:,:])
            loss = criterion(output[:,-1,:], targets[:,-1,:])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if (epoch%10 == 0):
            print(f'Saving prediction for epoch {epoch}')
            predict_model(model, test_loader, window_size, epoch, plot=True)    
        train_losses.append(total_loss*batch_size/len(train_loader.dataset))
        test_loss, debug_output = evaluate(model, test_loader, criterion)
        #print(f'Debug output: {debug_output}')
        test_losses.append(test_loss/len(test_loader.dataset))
        if epoch==1: ###DEBUG
            print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set')
        print(f'Epoch: {epoch}, train_loss: {total_loss*batch_size/len(train_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}')
        Early_Stopping(model, test_loss/len(test_loader))
        if Early_Stopping.early_stop:
            break
        writer.add_scalar('train_loss',total_loss,epoch)
        writer.add_scalar('val_loss',test_loss,epoch)
        if epoch%1 == 0:
            scheduler.step() 
### Plot losses        
    model = Early_Stopping.best_model
    xs = np.arange(len(train_losses))
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,train_losses)
    fig.savefig(root_dir + 'figs/transformer_train_loss.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,test_losses)
    fig.savefig(root_dir + 'figs/transformer_test_loss.png')
    plt.close(fig)
### Predict
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(test_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                test_rollout = targets
            else:
                enc_in = test_rollout[:,-window_size:,:]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
            enc_in, dec_in, targets = enc_in.to(device), dec_in.to(device), targets.to(device)
            output = model(enc_in, dec_in)

            test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            truth = torch.cat((truth, targets[:,-1,:].view(-1).detach().cpu()), 0)

    val_rollout = torch.Tensor(0)   
    val_result = torch.Tensor(0)  
    val_truth = torch.Tensor(0)

    with torch.no_grad():
        for i, (data,targets) in enumerate(val_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                val_rollout = targets
            else:
                enc_in = val_rollout[:,-window_size:,:]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
            enc_in, dec_in, targets = enc_in.to(device), dec_in.to(device), targets.to(device)
            output = model(enc_in, dec_in)

            val_rollout = torch.cat([val_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            val_result = torch.cat((val_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            val_truth = torch.cat((val_truth, targets[:,-1,:].view(-1).detach().cpu()), 0)


    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(test_result,label='forecast')
    ax.plot(truth,label = 'truth')
    ax.plot(test_result-truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/transformer_pred.png')
    plt.close(fig)

### Check MSE, MAE
    test_result = test_result.numpy()
    test_result = scaler.inverse_transform(test_result)
    truth = truth.numpy()
    truth = scaler.inverse_transform(truth)
    RMSE = mean_squared_error(truth, test_result)**0.5
    MAE = mean_absolute_error(truth, test_result)
    RMSE_first_window = mean_squared_error(truth[:window_size+1], test_result[:window_size+1])**0.5
    MAE_first_window = mean_absolute_error(truth[:window_size+1], test_result[:window_size+1])
    RMSE_after_first_window = mean_squared_error(truth[window_size:], test_result[window_size:])**0.5
    MAE_after_first_window = mean_absolute_error(truth[window_size:], test_result[window_size:])
    print(f'RMSE: {RMSE}, MAE: {MAE} \n RMSE_first_window: {RMSE_first_window}, MAE_first_window: {MAE_first_window} \n RMSE_after_first_window: {RMSE_after_first_window}, MAE_after_first_window: {MAE_after_first_window}')
    
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(test_result,label='forecast')
    ax.plot(truth,label = 'truth')
    ax.plot(test_result-truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/transformer_inverse_prediction.png')

    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(val_result,label='forecast')
    ax.plot(val_truth,label = 'truth')
    ax.plot(val_result-val_truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/transformer_val_inverse_prediction.png')

### Save model result
    val_result = val_result.numpy()
    val_result = scaler.inverse_transform(val_result)

    val_result_df = pd.DataFrame(val_result)
    val_result_df.to_csv(root_dir + '/transformer_val_prediction.csv')

    test_result_df = pd.DataFrame(test_result,columns=['predictions'])
    test_result_df.to_csv(root_dir + 'transformer_prediction.csv')
