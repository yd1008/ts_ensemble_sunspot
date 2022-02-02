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
import argparse
import random
from gru import GRU
from utils import *



def evaluate(model,data_loader,criterion):
    model.eval()    
    total_loss = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()
    return total_loss

def predict_model(model, test_loader, window_size, epoch, plot=True):
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    predict_loss = 0.
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(test_loader):
            if i == 0:
                data_in = data
                test_rollout = targets
            else:
                data_in = test_rollout[:,-window_size:,:]
            data_in, targets = data_in.to(device), targets.to(device)
            output = model(data_in)
            predict_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()

            test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
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
        fig.savefig(root_dir + f'/figs/gru_epoch{epoch}_pred.png')
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
            torch.save(model, 'best_gru.pth')
        else:
            self.counter += 1 
            if self.counter == self.patience:
                self.early_stop = True
                print('Early stopping')
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----')
    
  
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requires_training", default=False, action="store_true")
    parser.add_argument("--use_pre_trained", default=False, action="store_true")
    parser.add_argument("--use_nasa_test_range", default=False, action="store_true")
    parser.add_argument("--pre_trained_file_name")
    args = parser.parse_args()

    requires_training = args.requires_training
    use_pre_trained = args.use_pre_trained
    pre_trained_file_name = args.pre_trained_file_name
    use_nasa_test_range = args.use_nasa_test_range 

    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)


    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    root_dir = '' #specify where results will be saved
    #1999
    # best_config = {'hidden_size': 512, 'num_layers': 1, 'dropout': 0.1, 'window_size': 192, 'lr': 0.001, 'batch_size': 128, 'optim_step': 2, 'lr_decay': 0.8}
    # train_proportion = 0.6
    # test_proportion = 0.2
    # val_proportion = 0.2
    #future
    best_config = {'hidden_size': 1024, 'num_layers': 1, 'dropout': 0.2, 'window_size': 192, 'lr': 1e-05, 'batch_size': 128, 'optim_step': 20, 'lr_decay': 0.9}
    train_proportion = 0.7
    test_proportion = 0
    val_proportion = 0.3

    input_size = 1#best_config['input_size']
    hidden_size = best_config['hidden_size']
    num_layers = best_config['num_layers']
    optim_step = best_config['optim_step']
    lr_decay = best_config['lr_decay']
    dropout = best_config['dropout']
    lr = best_config['lr']
    window_size = best_config['window_size']
    batch_size = best_config['batch_size']

    train_val_loader, train_loader, val_loader, test_loader,scaler = get_data_loaders(train_proportion, test_proportion, val_proportion,\
        window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 1, pin_memory = False, test_mode = True, use_nasa_test_range = use_nasa_test_range)

    if requires_training == True:
        if use_pre_trained == True:
            model = torch.load(pre_trained_file_name)
        else: 
            model = GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = False)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        print('Using device: ',device)
        model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, optim_step, gamma=lr_decay)
        # writer = tensorboard.SummaryWriter('') #specify where tensorboard results will be saved
        
        epochs = 200
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        Early_Stopping = early_stopping(patience=20)

        for epoch in range(1, epochs + 1):
            model.train() 
            total_loss = 0.
            for (data, targets) in train_val_loader:

                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, targets)
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
            if (epoch%10 == 0):
                print(f'Saving prediction for epoch {epoch}')
                predict_model(model, test_loader, window_size, epoch, plot=True)    
            train_losses.append(total_loss*batch_size)
            test_loss = evaluate(model, test_loader, criterion)
            test_losses.append(test_loss/len(test_loader.dataset))
            if epoch==1: ###DEBUG
                print(f'Total of {len(train_val_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set')
            print(f'Epoch: {epoch}, train_loss: {total_loss*batch_size/len(train_val_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}')
            Early_Stopping(model, test_loss/len(test_loader))
            if Early_Stopping.early_stop:
                break
            # writer.add_scalar('train_loss',total_loss,epoch)
            # writer.add_scalar('val_loss',test_loss,epoch)
            scheduler.step() 
    ## Plot losses        
        
        xs = np.arange(len(train_losses))
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        ax.plot(xs,train_losses)
        fig.savefig(root_dir + 'figs/gru_train_loss.png')
        plt.close(fig)
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        ax.plot(xs,test_losses)
        fig.savefig(root_dir + 'figs/gru_test_loss.png')
        plt.close(fig)

### Predict
    if not requires_training: 
        model = torch.load(pre_trained_file_name)
    else:
        model = torch.load('best_gru.pth')
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    predict_loss = 0.
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(test_loader):
            if i == 0:
                data_in = data
                test_rollout = targets
            else:
                data_in = test_rollout[:,-window_size:,:]
                
            data_in, targets = data_in.to(device), targets.to(device)
            output = model(data_in)

            test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            truth = torch.cat((truth, targets[:,-1,:].view(-1).detach().cpu()), 0)

    val_rollout = torch.Tensor(0)   
    val_result = torch.Tensor(0)  
    val_truth = torch.Tensor(0)

    with torch.no_grad():
        for i, (data,targets) in enumerate(val_loader):
            if i == 0:
                data_in = data
                val_rollout = targets
            else:
                data_in = val_rollout[:,-window_size:,:]
            data_in, targets = data.to(device), targets.to(device)#data_in.to(device), targets.to(device)
            output = model(data_in)

            val_rollout = torch.cat([val_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            val_result = torch.cat((val_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            val_truth = torch.cat((val_truth, targets[:,-1,:].view(-1).detach().cpu()), 0)

    train_rollout = torch.Tensor(0)   
    train_result = torch.Tensor(0)  
    train_truth = torch.Tensor(0)

    with torch.no_grad():
        for i, (data,targets) in enumerate(train_loader):
            if i == 0:
                data_in = data
                train_rollout = targets
            else:
                data_in = train_rollout[:,-window_size:,:]
            data_in, targets = data.to(device), targets.to(device)#data_in.to(device), targets.to(device)
            output = model(data_in)

            train_rollout = torch.cat([train_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            train_result = torch.cat((train_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            train_truth = torch.cat((train_truth, targets[:,-1,:].view(-1).detach().cpu()), 0)
            

    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(test_result,label='forecast')
    ax.plot(truth,label = 'truth')
    ax.plot(test_result-truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/gru_pred.png')

    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(val_result,label='forecast')
    ax.plot(val_truth,label = 'truth')
    ax.plot(val_result-val_truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/gru_val_inverse_prediction.png')

    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(train_result,label='forecast')
    ax.plot(train_truth,label = 'truth')
    ax.plot(train_result-train_truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/gru_train_inverse_prediction.png')
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
    fig.savefig(root_dir + 'figs/gru_inverse_prediction.png')
### Save model result
    val_result = val_result.numpy()
    val_result = scaler.inverse_transform(val_result)
    val_result_df = pd.DataFrame(val_result)
    val_result_df.to_csv(root_dir + '/gru_val_prediction.csv')

    train_result = train_result.numpy()
    train_result = scaler.inverse_transform(train_result)
    train_result_df = pd.DataFrame(train_result)
    train_result_df.to_csv(root_dir + '/gru_train_prediction.csv')

    test_result_df = pd.DataFrame(test_result,columns=['predictions'])
    test_result_df.to_csv(root_dir + 'gru_prediction.csv')
