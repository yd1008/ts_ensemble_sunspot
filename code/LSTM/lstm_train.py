#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import time
import random
import os
import tensorboard
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune import CLIReporter
from lstm import LSTM
from utils import *


def evaluate(model,data_loader,criterion,batch_size):
    model.eval()
    total_loss = 0.
    rmse = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    for (data,targets) in data_loader:

        data, targets = data.to(device), targets.to(device)
        output = model(data)
        
        total_loss += criterion(output, targets).detach().cpu().numpy()
    return total_loss

def train(config, checkpoint_dir):
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)

    train_proportion = 0.7
    test_proportion = 0
    val_proportion = 0.3
    input_size = 1#config['input_size']
    window_size = config['window_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    dropout = config['dropout']
    lr = config['lr']
    lr_decay = config['lr_decay']
    optim_step = config['optim_step']
    batch_size = config['batch_size']

    print(f'Current configs are: {config}')

    model = LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = False)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    model.to(device)
    epochs = 200        
    criterion = nn.MSELoss() ######MAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, optim_step, gamma=lr_decay)

        
    train_loader,val_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 1, pin_memory = True, use_nasa_test_range = 'non_nasa_no_test')#non_nasa_no_test

    for epoch in range(1, epochs + 1):
        assert device == "cuda:0"

        model.train() 
        total_loss = 0.
        for (data, targets) in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output= model(data)
            loss = criterion(output, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        val_loss = evaluate(model, val_loader, criterion, batch_size)
        print(f'Epoch: {epoch}, train_loss: {total_loss}, val_loss: {val_loss}')
        tune.report(val_loss=val_loss)
        scheduler.step() 




if __name__ == "__main__":

    ray.init(ignore_reinit_error=False)
    config = {
        'hidden_size':tune.choice([128,216,512,1024]),
        'num_layers':tune.choice([1,2,3,4]),
        'dropout':tune.choice([0.1,0.2]),
        'lr':tune.choice([1e-3,5e-4,1e-4,5e-5,1e-5]),
        'window_size':tune.choice([192]),
        'batch_size':tune.choice([8,16,32,64,128]),
        'optim_step': tune.choice([2,5,10,15,20]), 
        'lr_decay': tune.choice([0.95,0.9,0.85,0.8,0.75,0.7]),
    }

    sched = ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=2,
            )
    analysis = tune.run(train, config=config, num_samples=1000, metric='val_loss', mode='min', scheduler=sched, resources_per_trial=tune.PlacementGroupFactory([{"CPU": 6, "GPU": 0.5}]),max_concurrent_trials = 8, queue_trials = True, max_failures=200, local_dir="") #specify where tune results will be saved
    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail)
    ray.shutdown()