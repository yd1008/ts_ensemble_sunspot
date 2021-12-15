#!/bin/bash python
import sklearn.metrics
import os
import pandas as pd
import numpy as np
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb

import ray
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

from utils_ensemble import *

def get_data(model_names, dirs, truth_dir):
    predictions = []
    truth = pd.read_csv(truth_dir,index_col=0).iloc[:,0].values#.reshape(-1,1)
    for model_name,dir in zip(model_names,dirs):
        file = pd.read_csv(dir)
        try:
          preds = file['predictions'].values.reshape(-1,1)
        except:
          preds = file['0'].values.reshape(-1,1)
        #predictions[model_name+'_pred'] = preds
        predictions.append(preds)
    #combine_df = pd.DataFrame(predictions)
    return np.concatenate(predictions,axis=1), truth

def train(config: dict):
    # This is a simple training function to be passed into Tune
    # Load dataset
    data_dir = '/scratch/yd1008/sunspot_informer/ensemble/forecast_results/'
    #model_names = ['transformer','lstm','gru','informer']
    model_names = ['lstm','gru','informer']
    #dirs = ['transformer_val_prediction.csv','lstm_val_prediction.csv','gru_val_prediction.csv','informer_val_prediction.csv']
    dirs = ['lstm_val_prediction.csv','gru_val_prediction.csv','informer_val_prediction.csv']
    dirs = [data_dir+dir for dir in dirs]
    data, labels = get_data(model_names,dirs,data_dir+'sunspot_val_truth.csv')
    # Split into train and test set
    total_len = len(data)
    window_len = 224
    train_len = int(window_len*0.7)
    #data[:train_len],data[train_len:window_len],labels[:train_len],labels[train_len:window_len]
    train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size=total_len-train_len,shuffle=False)
    #train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3,shuffle=False)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    val_set = xgb.DMatrix(val_x, label=val_y)
    #test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier, using the Tune callback
    xgb.train(
        config,
        train_set,
        evals=[(val_set, "eval")],
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])
    # RMSE = analysis.best_result["eval-rmse"]
    # tune.report(rmse=RMSE)

def get_best_model_checkpoint(analysis):
    best_bst = xgb.Booster()
    best_bst.load_model(os.path.join(analysis.best_checkpoint, "model.xgb"))
    RMSE = analysis.best_result["eval-rmse"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model rmse: {RMSE}")
    return best_bst





if __name__ == "__main__":
    # model_names = ['transformer','lstm','gru']
    # dirs = ['transformer_prediction.csv','lstm_prediction.csv','gru_prediction.csv']
    # x,y = get_data(model_names,dirs,'sunspot_truth.csv')
    config = {
        # You can mix constants with search space objects.
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": tune.randint(4, 50),
        "min_child_weight": tune.randint(5,100),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 5e-1),
        # "gamma": tune.randint(0,100),
        # "max_delta_step": tune.uniform(1,10),
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=40,  # 10 training iterations
        grace_period=5,
        reduction_factor=2)

    analysis = tune.run(
        train,
        metric="eval-rmse",
        mode="min",
        # You can add "gpu": 0.1 to allocate GPUs
        resources_per_trial={"gpu": 0.2},
        config=config,
        num_samples=300,
        scheduler=scheduler)
    
    #best_trail = analysis.get_best_config(mode='min')
    #print('The best configs are: ',best_trail)

    best_bst = get_best_model_checkpoint(analysis)

    data_dir = '/scratch/yd1008/sunspot_informer/ensemble/forecast_results/'
    # model_names = ['transformer','lstm','gru','informer']
    # dirs = ['transformer_prediction.csv','lstm_prediction.csv','gru_prediction.csv','informer_prediction.csv']
    model_names = ['lstm','gru','informer']
    dirs = ['lstm_prediction.csv','gru_prediction.csv','informer_prediction.csv']
    dirs = [data_dir+dir for dir in dirs]
    data, labels = get_data(model_names,dirs,data_dir+'sunspot_truth.csv')
    total_len = len(data)
    #window_len = 224
    #train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=total_len-window_len,shuffle=False)
    #train_x, train_y = data, labels
    test_set = xgb.DMatrix(data, label=labels)
    preds = best_bst.predict(test_set)
    RMSE = mean_squared_error(labels,preds)**0.5
    MAE = mean_absolute_error(labels,preds)
    print(f'RMSE: {RMSE}, MAE: {MAE}')

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('xgb_preds.csv')