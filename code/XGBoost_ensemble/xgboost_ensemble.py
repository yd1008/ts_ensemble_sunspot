#!/bin/bash python
import sklearn.metrics
import os
import numpy as np
import pandas as pd
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb

import ray
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback


def get_data(model_names, dirs, truth_dir):
    predictions = []
    truth = pd.read_csv(truth_dir,index_col=0).iloc[:,0].values#.reshape(-1,1)
    for model_name,dir in zip(model_names,dirs):
        file = pd.read_csv(dir)
        try:
          preds = file['predictions'].values.reshape(-1,1)
        except:
          preds = file['0'].values.reshape(-1,1)
        predictions.append(preds)
    return np.concatenate(predictions,axis=1), truth

def train(config: dict):
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    val_set = xgb.DMatrix(val_x, label=val_y)
    xgb.train(
        config,
        train_set,
        num_boost_round = 20,
        evals=[(val_set, "eval")],
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])


def get_best_model_checkpoint(analysis):
    best_bst = xgb.Booster()
    best_bst.load_model(os.path.join(analysis.best_checkpoint, "model.xgb"))
    RMSE = analysis.best_result["eval-rmse"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model rmse: {RMSE}")
    return best_bst





if __name__ == "__main__":


    data_dir = '/scratch/yd1008/sunspot_informer/ensemble/forecast_results/'
    model_names = ['lstm','gru','informer','transformer']
    dirs = ['lstm_prediction.csv','gru_prediction.csv','informer_prediction.csv','transformer_prediction.csv']
    dirs = [data_dir+dir for dir in dirs]
    data, labels = get_data(model_names,dirs,data_dir+'sunspot_truth.csv')
    # Split into train and test set
    total_len = len(data)
    train_len = int(total_len*0.5)
    val_len = int(total_len*0.2)
    test_len = total_len - train_len - val_len
    # train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size=total_len-train_len,shuffle=False)
    # val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=test_len,shuffle=False)

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=test_len,shuffle=False)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_len,shuffle=False)

    config = {
        "objective": "reg:squarederror", #
        "eval_metric": "rmse",
        "max_depth": tune.randint(2, 20),
        "min_child_weight": tune.randint(1,20),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 5e-1),
        "gamma": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "seed": 1008,
    }
    scheduler = ASHAScheduler(
        max_t=100,  
        grace_period=5,
        reduction_factor=2)

    analysis = tune.run(
        train,
        metric="eval-rmse",
        mode="min",
        # You can add "gpu": 0.1 to allocate GPUs
        resources_per_trial={"gpu": 0.2},
        config=config,
        num_samples=1000,
        scheduler=scheduler)

    best_bst_dl = get_best_model_checkpoint(analysis)

    test_set = xgb.DMatrix(test_x, label=test_y)
    preds = best_bst_dl.predict(test_set)
    RMSE = mean_squared_error(test_y,preds)**0.5
    MAE = mean_absolute_error(test_y,preds)
    print(f'RMSE: {RMSE}, MAE: {MAE}')

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('xgb_dl_preds.csv')


### REPEAT PROCESS FOR CLASSICAL METHODS
    model_names = ['arima','prophet','es']
    dirs = ['arima_prediction.csv','prophet_prediction.csv','es_prediction.csv']
    dirs = [data_dir+dir for dir in dirs]
    data, labels = get_data(model_names,dirs,data_dir+'sunspot_truth.csv')
    # Split into train and test set
    total_len = len(data)
    train_len = int(total_len*0.5)
    val_len = int(total_len*0.2)
    test_len = total_len - train_len - val_len

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=test_len,shuffle=False)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_len,shuffle=False)


    config = {
        "objective": "reg:squarederror", #
        "eval_metric": "rmse",
        "max_depth": tune.randint(2, 20),
        "min_child_weight": tune.randint(1,20),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 5e-1),
        "gamma": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "seed": 1008,
    }
    scheduler = ASHAScheduler(
        max_t=100,  
        grace_period=5,
        reduction_factor=2)

    analysis = tune.run(
        train,
        metric="eval-rmse",
        mode="min",
        resources_per_trial={"gpu": 0.2},
        config=config,
        num_samples=1000,
        scheduler=scheduler)
    
    #best_trail = analysis.get_best_config(mode='min')
    #print('The best configs are: ',best_trail)

    best_bst = get_best_model_checkpoint(analysis)

    test_set = xgb.DMatrix(test_x, label=test_y)
    preds = best_bst.predict(test_set)
    RMSE = mean_squared_error(test_y,preds)**0.5
    MAE = mean_absolute_error(test_y,preds)
    print(f'RMSE: {RMSE}, MAE: {MAE}')

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('xgb_cl_preds.csv')

### Ensemble all models

    data_dir = '/scratch/yd1008/sunspot_informer/ensemble/forecast_results/'
    model_names = ['lstm','gru','informer','transformer','arima_prediction.csv','prophet_prediction.csv','es_prediction.csv']
    dirs = ['lstm_prediction.csv','gru_prediction.csv','informer_prediction.csv','transformer_prediction.csv','arima_prediction.csv','prophet_prediction.csv','es_prediction.csv']
    dirs = [data_dir+dir for dir in dirs]
    data, labels = get_data(model_names,dirs,data_dir+'sunspot_truth.csv')
    # Split into train and test set
    total_len = len(data)
    train_len = int(total_len*0.5)
    val_len = int(total_len*0.2)
    test_len = total_len - train_len - val_len
    # train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size=total_len-train_len,shuffle=False)
    # val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=test_len,shuffle=False)

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=test_len,shuffle=False)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_len,shuffle=False)

    config = {
        "objective": "reg:squarederror", #
        "eval_metric": "rmse",
        "max_depth": tune.randint(2, 20),
        "min_child_weight": tune.randint(1,20),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 5e-1),
        "gamma": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "seed": 1008,
    }
    scheduler = ASHAScheduler(
        max_t=100,  
        grace_period=5,
        reduction_factor=2)

    analysis = tune.run(
        train,
        metric="eval-rmse",
        mode="min",
        # You can add "gpu": 0.1 to allocate GPUs
        resources_per_trial={"gpu": 0.2},
        config=config,
        num_samples=1000,
        scheduler=scheduler)

    best_bst_dl = get_best_model_checkpoint(analysis)

    test_set = xgb.DMatrix(test_x, label=test_y)
    preds = best_bst_dl.predict(test_set)
    RMSE = mean_squared_error(test_y,preds)**0.5
    MAE = mean_absolute_error(test_y,preds)
    print(f'RMSE: {RMSE}, MAE: {MAE}')

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('xgb_all_preds.csv')

### Now apply the learned dl model on ensembling future forecasts
    model_names = ['lstm','gru','informer','transformer']
    dirs = ['lstm_future_prediction.csv','gru_future_prediction.csv','informer_future_prediction.csv','transformer_future_prediction.csv']
    dirs = [data_dir+dir for dir in dirs]
    data_future, _ = get_data(model_names,dirs,data_dir+'sunspot_truth.csv')
    # Split into train and test set
    total_len_future = len(data_future)
    forecast_len_future = total_len_future-total_len
    future_preds = data_future[-forecast_len_future:,:]

    future_set = xgb.DMatrix(future_preds)
    ensembled_future = best_bst_dl.predict(future_set)

    future_preds_df = pd.DataFrame(ensembled_future)
    future_preds_df.to_csv('xgb_dl_future_preds.csv')

#Best Config for Classical model ensemble: {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': 2, 'min_child_weight': 17, 'subsample': 0.8465963829651589, 'eta': 0.022196362873346344, 'gamma': 0.7624626803533553, 'colsample_bytree': 0.6949972004384192, 'seed': 1008}
#Best Config for Deep Learning model ensemble: {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': 9, 'min_child_weight': 6, 'subsample': 0.5003656722001157, 'eta': 0.045109592193823975, 'gamma': 0.7805612947804967, 'colsample_bytree': 0.6757012060741239, 'seed': 1008}
#Best Config for all model ensemble: {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': 11, 'min_child_weight': 19, 'subsample': 0.5277500988636198, 'eta': 0.03357901686398789, 'gamma': 0.61608833081076, 'colsample_bytree': 0.9015460901454133, 'seed': 1008}