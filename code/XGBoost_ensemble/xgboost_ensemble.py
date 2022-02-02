#!/bin/bash python
import sklearn.metrics
import os
import numpy as np
import pandas as pd
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
import argparse
import ray
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback


def get_data(model_names, dirs, truth_dir):
    predictions = []
    for model_name,dir in zip(model_names,dirs):
        file = pd.read_csv(dir)
        try:
          preds = file['predictions'].values.reshape(-1,1)
        except:
          preds = file['0'].values.reshape(-1,1)
        predictions.append(preds)
    if truth_dir != '':
        truth = pd.read_csv(truth_dir,index_col=0).iloc[:,0].values#.reshape(-1,1)
        return np.concatenate(predictions,axis=1), truth
    elif truth_dir == '':
        return np.concatenate(predictions,axis=1)

def train(config: dict):
    # Build input matrices for XGBoost
    val_set = xgb.DMatrix(val_x, label=val_y)
    train_set = xgb.DMatrix(train_x, label=train_y)
    n_estimators = config["n_estimators"]
    del config["n_estimators"]
    xgb.train(
        config,
        train_set,
        num_boost_round = n_estimators,
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
    parser = argparse.ArgumentParser()  
    parser.add_argument("--requires_training", default = False, action='store_true')        
    parser.add_argument("--test_pre_trained_file_name")
    parser.add_argument("--future_pre_trained_file_name")
    args = parser.parse_args()

    test_pre_trained_file_name = args.test_pre_trained_file_name
    future_pre_trained_file_name = args.future_pre_trained_file_name
    requires_training = args.requires_training

    data_dir = '' #specify where individual models' predictions are stored
    model_names = ['gru','lstm','informer','transformer']
    train_dirs = ['../GRU/gru_train_prediction_1999.csv','../LSTM/lstm_train_prediction_1999.csv','../Informer/informer_train_prediction_1999.csv','../Transformer/transformer_train_prediction_1999.csv']
    # val_dirs = ['../GRU/gru_val_prediction_1999.csv','../LSTM/lstm_val_prediction_1999.csv','../Informer/informer_val_prediction_1999.csv','../Transformer/transformer_val_prediction_1999.csv']
    # test_dirs = ['../GRU/gru_prediction_1999.csv','../LSTM/lstm_prediction_1999.csv','../Informer/informer_prediction_1999.csv','../Transformer/transformer_prediction_1999.csv']
    train_dirs = [data_dir+dir for dir in train_dirs]
    # val_dirs = [data_dir+dir for dir in val_dirs]
    # test_dirs = [data_dir+dir for dir in test_dirs]

    # train_x, train_y = get_data(model_names,train_dirs,data_dir+'../Informer/sunspot_train_truth_1999.csv')
    # val_x, val_y = get_data(model_names,val_dirs,data_dir+'../Informer/sunspot_val_truth_1999.csv')
    # train_val_x = np.concatenate([train_x,val_x],axis=0)
    # train_val_y = np.concatenate([train_y,val_y],axis=0)
    # train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=len(val_x),shuffle=True, random_state=1008)
    test_x, test_y = get_data(model_names,test_dirs,data_dir+'../Informer/sunspot_truth_1999.csv')
    
    if requires_training:
        config = {
            "objective": "reg:squarederror", #
            "eval_metric": "rmse",
            "n_estimators": tune.randint(1,31),
            "max_depth": tune.randint(1, 21),
            "min_child_weight": tune.randint(1,41),
            "subsample": tune.uniform(0.1, 1.0),
            "eta": tune.loguniform(1e-6, 9e-1),
            "gamma": tune.uniform(0, 1.0),
            'reg_lambda': tune.choice([0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            'reg_alpha':tune.choice([1e-5, 1e-2, 0.1, 1, 100]),
            "colsample_bytree": tune.uniform(0.1, 1.0),
            "seed": tune.choice([1008]),
        }
        scheduler = ASHAScheduler(
            max_t=20,  
            grace_period=10,
            reduction_factor=2)

        analysis = tune.run(
            train,
            metric="eval-rmse",
            mode="min",
            resources_per_trial={"gpu": 0.05},
            config=config,
            num_samples=100,
            scheduler=scheduler)
        best_bst_dl = get_best_model_checkpoint(analysis)
    else:
        best_bst_dl = xgb.Booster()   
        best_bst_dl.load_model(test_pre_trained_file_name)
    test_set = xgb.DMatrix(test_x, label=test_y)
    preds = best_bst_dl.predict(test_set)
    RMSE = mean_squared_error(test_y,preds)**0.5
    MAE = mean_absolute_error(test_y,preds)
    print(f'RMSE: {RMSE}, MAE: {MAE}')

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('xgb_dl_preds.csv')
    #best_bst_dl.save_model("model_dl.pth")

### REPEAT PROCESS FOR FUTURE ENSEMBLE
    model_names = ['gru','lstm','informer','transformer']
    train_dirs = ['../GRU/gru_prediction_future.csv','../LSTM/lstm_future_prediction_future.csv','../Informer/informer_future_prediction_future.csv','../Transformer/transformer_future_prediction_future.csv']
    # val_dirs = ['../GRU/gru_val_prediction_future.csv','../LSTM/lstm_val_prediction_future.csv','../Informer/informer_val_prediction_future.csv','../Transformer/transformer_val_prediction_future.csv']
    # test_dirs = ['../GRU/gru_prediction_future.csv','../LSTM/lstm_prediction_future.csv','../Informer/informer_prediction_future.csv','../Transformer/transformer_prediction_future.csv']
    train_dirs = [data_dir+dir for dir in train_dirs]
    # val_dirs = [data_dir+dir for dir in val_dirs]
    # test_dirs = [data_dir+dir for dir in test_dirs]

    # train_x, train_y = get_data(model_names,train_dirs,data_dir+'../Informer/sunspot_train_truth_1999.csv')
    # val_x, val_y = get_data(model_names,val_dirs,data_dir+'../Informer/sunspot_val_truth_1999.csv')
    # train_val_x = np.concatenate([train_x,val_x],axis=0)
    # train_val_y = np.concatenate([train_y,val_y],axis=0)
    # train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=len(val_x),shuffle=True, random_state=1008)
    test_x, test_y = get_data(model_names,test_dirs,data_dir+'../Informer/sunspot_truth_1999.csv')
    
    if requires_training:
        config = {
            "objective": "reg:squarederror", #
            "eval_metric": "rmse",
            "n_estimators": tune.randint(1,31),
            "max_depth": tune.randint(1, 21),
            "min_child_weight": tune.randint(1,41),
            "subsample": tune.uniform(0.1, 1.0),
            "eta": tune.loguniform(1e-6, 9e-1),
            "gamma": tune.uniform(0, 1.0),
            'reg_lambda': tune.choice([0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            'reg_alpha':tune.choice([1e-5, 1e-2, 0.1, 1, 100]),
            "colsample_bytree": tune.uniform(0.1, 1.0),
            "seed": tune.choice([1008]),
        }
        scheduler = ASHAScheduler(
            max_t=20,  
            grace_period=10,
            reduction_factor=2)

        analysis = tune.run(
            train,
            metric="eval-rmse",
            mode="min",
            resources_per_trial={"gpu": 0.05},
            config=config,
            num_samples=100,
            scheduler=scheduler)
        best_bst_dl = get_best_model_checkpoint(analysis)
    else:
        best_bst_dl = xgb.Booster()   
        best_bst_dl.load_model(future_pre_trained_file_name)
    test_set = xgb.DMatrix(test_x, label=test_y)
    preds = best_bst_dl.predict(test_set)
    RMSE = mean_squared_error(test_y,preds)**0.5
    MAE = mean_absolute_error(test_y,preds)
    print(f'RMSE: {RMSE}, MAE: {MAE}')

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('xgb_dl_preds.csv')
    #best_bst_dl.save_model("model_dl.pth")

