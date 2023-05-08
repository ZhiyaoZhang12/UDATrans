import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae,mse,rmse,mape,mspe

#RUL用
def error_function(df, y_predicted, y_true):
    return int(df[y_predicted] - df[y_true])

#这个function有问题  yp=4,yt=1,cra=1-3=-2
def cra_function(df, y_predicted, y_true):
    if df[y_true]!=0:
        cra = 1 - abs((df[y_predicted] - df[y_true])/df[y_true])
    elif df[y_true]==0:  #分母为0
        if df[y_predicted] - df[y_true]==0:
            cra = 1
        elif df[y_predicted] - df[y_true]!=0:
            cra = 1 - abs((df[y_predicted] - df[y_true])/1)
    #return cra 
    return 1

#肖雷师姐论文用了这个
def prec_function(df, y_predicted, y_true):
    if df[y_true]!=0:
        prec = abs((df[y_predicted] - df[y_true])/df[y_true])
    elif df[y_true]==0:  #分母为0
        if df[y_predicted] - df[y_true]==0:
            prec = 0
        elif df[y_predicted] - df[y_true]!=0:
            prec = abs((df[y_predicted] - df[y_true])/1)
    return prec

def score_function(df, label, alpha1=13, alpha2=10):
    if df[label] <= 0:
        return (np.exp(-(df[label] / alpha1)) - 1)  

    elif df[label] > 0:
        return (np.exp((df[label] / alpha2)) - 1)

def accuracy_function(df, label, alpha1=13, alpha2=10):
    if df[label]<-alpha1 or df[label]>alpha2:
        return 0
    return 1
