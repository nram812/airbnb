import pandas as pd
import numpy as np
def create_training_dataset(categorical, regression, price, index, regression_columns, norm_reg = True):
    """
    categorical data preprocessed
    index: training, test and validation dataframe
    regression data preprocessed
    regression columns used
    normalize the data
    """
    X = pd.concat([categorical, regression, price], axis =1)
    X_train = X[index['dset'] == 'train'].dropna()
    # compute the training and testing dataset mean
    X_train_regression_mean = X_train[regression_columns].mean()
    X_train_regression_std = X_train[regression_columns].std()

    
    X_test = X[index['dset'] == 'test'].dropna()
    # remove nan values in the training data - note this means the amount of training
    # /testing data is reduced.

    
    X_val = X[index['dset'] == 'validation'].dropna()
    # normalize relative to the output
    if norm_reg:
        # note the testing and validation are normalized relative to the training dataset mean and std
        X_val[regression_columns] = ( X_val[regression_columns] - X_train_regression_mean)/ X_train_regression_std
        X_test[regression_columns] = (X_test[regression_columns] - X_train_regression_mean)/ X_train_regression_std
        X_train[regression_columns] = (X_train[regression_columns] - X_train_regression_mean)/ X_train_regression_std
    y_train = price[index['dset'] == 'train']
    y_test = price[index['dset'] == 'test']
    y_val = price[index['dset'] == 'validation']
    # I wouldn't normally write code like this, but had to get this done!
    return X_train.drop(["price","log_price"], axis =1),\
            X_test.drop(["price","log_price"], axis =1), X_val.drop(["price","log_price"], axis =1),\
            X_train["log_price"], X_test["log_price"], X_val["log_price"]
    
def eval_metrics(x, y, model):
    """
    quick and dirty validation metrics for the model. 
    """
    y_pred_df = pd.DataFrame(y.copy())
    y = y.values.flatten()
    
    y_hat = model.predict(x).flatten()
    assert y_hat.size == y.size
    metrics = {}
    metrics['log_corrcoef'] = np.corrcoef(y_hat , y)[0,1]
    metrics['corrcoef'] = np.corrcoef(np.exp(y_hat ), np.exp(y))[0,1]
    metrics['rmse'] = np.sqrt(np.nanmean(abs(np.exp(y_hat ) - np.exp(y))**2))
    metrics['log_rmse'] = np.sqrt(np.nanmean(abs(y_hat  - y)**2))
    df = pd.DataFrame(metrics.items())
    df.index = df[0]
    y_pred_df.loc[:,'predictions'] = y_hat.ravel()
    return df[1], y_pred_df