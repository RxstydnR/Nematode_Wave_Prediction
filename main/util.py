import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal

def timestep_predict(x_test):
    
    for i in range(len(x_test)):        
        state_value = encoder_model.predict(x_test[i])
        y_decoder = np.zeros((1, 1, 1)) # 出力の値
        predicted = [] # 変換結果

        for i in range(N_SEQUENCE):
            y, h, c = decoder_model.predict([y_decoder] + state_value)  # 前の出力と状態を渡す
            y = y[0][0][0]
            predicted.append(y)
            y_decoder[0][0][0] = y  # 次に渡す値
            state_value = [h, c] # 次に渡す状態

    return predicted



########## トレンド除去 ##########
def DE_remove_trend(X,freq):
    """ 時系列データの基本成分の分解(decompose)により、トレンドを除去したデータを返す.
    
        Note: 
            データ毎に周期パラメータが必要であることが問題
    """
    X_DE=[]
    for i in range(X.shape[0]):
        x = X[i]
        sdr = sm.tsa.seasonal_decompose(x,period=freq)
        x_DE = sdr.observed - sdr.trend
        X_DE.append(x_DE)
    return np.array(X_DE)

def Linear_remove_trend(X):
    """ Scipy(手法は最小二乗法??)によりトレンドを除去したデータを返す.
    """
    X_detrend=[]
    for i in range(X.shape[0]):
        x = X[i]
        x_detrend = signal.detrend(x)
        X_detrend.append(x_detrend)
    return np.array(X_detrend)

########## 平滑化（ノイズ除去） ##########

def MA(X, wsize, center=True):
    """ Mean Average (移動平均)
    """
    X_df = pd.DataFrame(X)
    X_df = X_df.rolling(wsize,center=center,axis=1).mean()
    X_df = X_df.interpolate(method='spline', order=1, axis=1, limit_direction='both')
    return X_df.values

def SSA(X, wsize, fac):
    """ Singular Spectrum Analysis (特異スペクトル解析)
    """
    X_SSA=[]
    for i in range(X.shape[0]):
        x = X[i]
        tra = np.array([x[j:j+wsize] for j in range(len(x)-wsize+1)])
        u,s,v = np.linalg.svd(tra)
        t = np.dot(np.dot(u[:,:fac], np.diag(s[:fac])), v[:fac])
        x_SSA = np.concatenate([t[0], t[1:,-1]])
        X_SSA.append(x_SSA)
    return np.array(X_SSA)



def show_history(history, SAVE_PATH):
    """ Save training history
    """
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"],label="Loss")
    plt.plot(history.history["val_loss"],label="Val Loss")
    plt.legend()
    plt.title("Training History")
    plt.savefig(os.path.join(SAVE_PATH,f"history.jpg"))
    plt.clf()
    plt.close()
    

def show_results(X_test,Y_test,Y_pred,SAVE_PATH):
    """ Save test prediction
    """

    width=2
    height=int(np.ceil(len(X_test)/width))

    plt.figure(figsize=(width*7,height*3))
    for i in range(len(X_test)):
        plt.subplot(height,width,i+1)
        plt.plot(X_test[i], label="x:GC")
        plt.plot(Y_test[i], label="y:pa")
        plt.plot(Y_pred[i], label="pred:pa")
        plt.legend()    
    plt.title("Test Prediction Results")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH,f"results.jpg"))
    plt.clf()
    plt.close()