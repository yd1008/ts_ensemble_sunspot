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
from gru import GRU
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
    # torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)

    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    input_size = 1#config['input_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    dropout = config['dropout']
    lr = config['lr']
    window_size = config['window_size']
    batch_size = config['batch_size']

    print(f'Current configs are: {config}')

    model = GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = False)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    model.to(device)
    epochs = 200        
    criterion = nn.MSELoss() ######MAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95, last_epoch = -1 )

        
    train_loader,val_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 0, pin_memory = False)

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


class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated
    

if __name__ == "__main__":
    

    print('Current ray version is: ',ray.__version__)
# Limit the number of rows.
    reporter = CLIReporter(max_progress_rows=10)
    ray.init(ignore_reinit_error=False)
    config = {
        'hidden_size':tune.grid_search([216,512,1024]),
        'num_layers':tune.grid_search([1,2,3,4]),
        'dropout':tune.grid_search([0.1,0.2]),
        'window_size': tune.grid_search([192]),
        'lr': tune.grid_search([1e-3,5e-4,1e-4,5e-5,1e-5]),
        'batch_size': tune.grid_search([8,16,32,64,128]),
    }
    #reporter = CLIReporter(max_progress_rows=10)
    reporter.add_metric_column("val_loss")
    sched = ASHAScheduler(
            max_t=100,
            grace_period=30,
            reduction_factor=2,
            )
    # analysis = tune.run(tune.with_parameters(train), config=config, metric='val_loss', mode='min',\
    #      scheduler=sched, resources_per_trial={"gpu": 1/2, "cpu": 24}, max_concurrent_trials=2, max_failures=100,queue_trials = True, local_dir="/scratch/yd1008/ray_results")#, progress_reporter=reporter)
    analysis = tune.run(train, config=config, metric='val_loss', mode='min', scheduler=sched, resources_per_trial={"gpu": 1/2, "cpu": 24}, max_concurrent_trials=2, local_dir="/scratch/yd1008/ray_results")
    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail)
    ray.shutdown()