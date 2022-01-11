#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import time
import random
import os
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from transformer import Tranformer
from utils import *

def process_one_batch(model, batch_x, batch_y, device, window_size):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]]).float().to(device)
        dec_inp = torch.cat([batch_y[:,:(window_size-1),:], dec_inp], dim=1).float().to(device)
        outputs = model(batch_x, dec_inp)

        return outputs, batch_y

def evaluate(model,data_loader,criterion,window_size):
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
            enc_in, dec_in = data.clone().to(device), targets.clone().to(device)
            targets = targets.to(device)
            dec_in = torch.cat([dec_in[:,:(window_size-1),:], torch.zeros([dec_in.shape[0], 1, dec_in.shape[-1]]).to(device)], dim=1).float().to(device)
            output = model(enc_in, dec_in)

            total_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()

    return total_loss

def train(config, checkpoint_dir):
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)

    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2

    batch_size = config['batch_size']
    lr = config['lr']
    window_size = config['window_size']

    feature_size = config['feature_size']
    num_enc_layers = config['num_enc_layers']
    num_dec_layers = config['num_dec_layers']
    d_ff = config['d_ff']
    num_head = config['num_head']
    dropout = config['dropout']
    
    

    model = Tranformer(feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
        d_ff = d_ff, dropout=dropout,num_head=num_head)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    epochs = 150
    criterion = nn.MSELoss() ######MAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.95)
    #writer = tensorboard.SummaryWriter('./test_logs')
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
        
    train_loader,val_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 2, pin_memory = True)

    assert device == "cuda:0"
    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.

        for (data, targets) in train_loader:
            enc_in, dec_in = data.clone().to(device), targets.clone().to(device)
            targets = targets.to(device)
            #enc_in, dec_in, targets = data.to(device),targets.to(device),targets.to(device)
            optimizer.zero_grad()
            dec_zeros = torch.zeros([dec_in.shape[0], 1, dec_in.shape[-1]]).float().to(device)
            dec_in = torch.cat([dec_in[:,:(window_size-1),:], dec_zeros], dim=1).float().to(device)
            output = model(enc_in, dec_in)
            #output = process_one_batch(data,targets,device,window_size)
            loss = criterion(output[:,-1,:], targets[:,-1,:])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        val_loss = evaluate(model, val_loader, criterion, window_size)
        print(f'Epoch: {epoch}, train_loss: {total_loss}, val_loss: {val_loss}')
        #writer.add_scalar('train_loss',total_loss,epoch)
        #writer.add_scalar('val_loss',val_loss,epoch)
        tune.report(val_loss=val_loss/len(val_loader.dataset))
        scheduler.step() 


if __name__ == "__main__":
    config = {
        'feature_size':tune.choice([216,512,1024]),
        'num_enc_layers':tune.choice([2,3,4,5]),
        'num_dec_layers':tune.choice([2,3,4,5]),
        'num_head':tune.choice([2,4,8]),
        'd_ff':tune.choice([216,512,1024]),
        'dropout':tune.choice([0.1,0.2]),
        'lr':tune.grid_search([1e-3,5e-4,1e-4,5e-5,1e-5]),
        'window_size':tune.choice([192]),
        'batch_size':tune.grid_search([16,32,64,128,256]),
        'optim_step': tune.choice([2,5,10,15,20]), 
        'lr_decay': tune.choice([0.95,0.9,0.85,0.8,0.75,0.7]),
}
    ray.init(ignore_reinit_error=False, include_dashboard=True, dashboard_host= '0.0.0.0')
    sched = ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=2)
    analysis = tune.run(tune.with_parameters(train), config=config, num_samples=1000, metric='val_loss', mode='min',\
         scheduler=sched, resources_per_trial={"cpu": 12,"gpu": 1/2},max_concurrent_trials=4, max_failures=1000, local_dir="/scratch/yd1008/ray_results",)

    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail)
    ray.shutdown()