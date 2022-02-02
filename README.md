# ts_ensemble_sunspot

**Requirements**
 - Python 3.8
 - xgboost==1.5.1
 - ray==1.9.0 
 - ray[tune]==1.9.0 
 - torch==1.9.0
 - numpy==1.20.3 
 - pandas==1.1.4
 - matplotlib==3.4.2 
 - seaborn==0.11.2

Run the following command to install the required dependencies:
```python
pip install -r requirements.txt
```
Follow the commands below to reproduce results in this study:

 1. Generate predictions on the test portion with provided pre-trained models:
 ```python
 python informer_result.py --use_pre_trained --use_nasa_test_range --pre_trained_file_name ../../train_models/best_informer.pth
 python transformer_result.py --use_pre_trained --use_nasa_test_range --pre_trained_file_name ../../train_models/best_transformer.pth
 python lstm_result.py --use_pre_trained --use_nasa_test_range --pre_trained_file_name ../../train_models/best_lstm.pth
 python gru_result.py --use_pre_trained --use_nasa_test_range --pre_trained_file_name ../../train_models/best_gru.pth
 ```
 2. Generate predictions on the future portion with provided pre-trained models:
 ```python
 python informer_future.py --use_pre_trained --pre_trained_file_name ../../train_models/best_informer_future.pth
 python transformer_future.py --use_pre_trained --pre_trained_file_name ../../train_models/best_transformer_future.pth
 python lstm_future.py --use_pre_trained --pre_trained_file_name ../../train_models/best_lstm_future.pth
 python gru_future.py --use_pre_trained --pre_trained_file_name ../../train_models/best_gru_future.pth
 ```
 3. Combine predictions on both test and future portions with pre-trained XGBoost models:
  ```python
 python xgboost_ensemble.py --test_pre_trained_file_name ../../train_models/xgboost_dl.pth --future_pre_trained_file_name ../../train_models/xgboost_future.pth
 ```


