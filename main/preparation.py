import glob
import numpy as np
import pandas as pd

from data import get_data, get_data_name, make_start_0
from data import create_timestep_sequences,create_sequences
from data import get_image_data, crop_image, resize_image
from data_augmentation import data_augmentation,data_augmentation_xy
from util import DE_remove_trend, Linear_remove_trend
from util import MA,SSA
from util import MA,SSA

def data_prepare(train_date, test_date):

    df = pd.read_csv("../data/Nematode_dataset_1.csv",index_col=0)
    df = df.append(df.iloc[-1]).reset_index(drop=True) # データ数を1400に揃える

    df_X = df.filter(regex='GC_p$',axis=1) # GC
    df_Y = df.filter(regex='pa_p$',axis=1) # pa

    df_X_train=pd.DataFrame()
    df_Y_train=pd.DataFrame()
    for date in train_date:
        df_X_train = pd.concat([df_X_train, df_X.filter(like=date+"_",axis=1)], axis=1)
        df_Y_train = pd.concat([df_Y_train, df_Y.filter(like=date+"_",axis=1)], axis=1)

    df_X_test=pd.DataFrame()
    df_Y_test=pd.DataFrame()
    for date in test_date:
        df_X_test = pd.concat([df_X_test, df_X.filter(like=date+"_",axis=1)], axis=1)
        df_Y_test = pd.concat([df_Y_test, df_Y.filter(like=date+"_",axis=1)], axis=1)

    X_train = df_X_train.T.values
    X_test  = df_X_test.T.values
    Y_train = df_Y_train.T.values
    Y_test  = df_Y_test.T.values

    X_train = MA(X_train, 20, center=False)
    X_test  = MA(X_test, 20, center=False)
    Y_train = MA(Y_train, 20, center=False)
    Y_test  = MA(Y_test, 20, center=False)

    # 先頭の余分な部分を削る
    # X = X[:,99:]
    # Y = Y[:,99:]

    # 2波形の相関係数を計算
    # corr_cs=[] 
    # for i in range(X.shape[0]):
    #     corr_c = np.corrcoef(X[i],Y[i])[0,1]
    #     corr_cs.append(corr_c)
    # corr_cs = np.array(corr_cs)

    # 相関係数 0.5 以上を採用
    # X = X[corr_cs>=0.5]
    # Y = Y[corr_cs>=0.5]

    # 波形の先頭を0に揃える
    X_train = make_start_0(X_train)
    X_test = make_start_0(X_test)
    Y_train = make_start_0(Y_train)
    Y_test = make_start_0(Y_test)

    return X_train,X_test,Y_train,Y_test


def image_data_prepare(train_date, test_date):
    
    paths = sorted(glob.glob("/data/Users/katafuchi/RA/Nematode/2021*"))
    
    train_path=[]
    test_path=[]
    for path in paths:
        name = os.path.basename(path) 
        date = name[5:9]+name[42:]
        if date in train_date:
            train_path.append(path)
        elif date in test_date:
            test_path.append(path)

    X_train=[]
    for path in train_path:
        x = get_image_data(sorted(glob.glob(f"{path}/*.tif")))
        x = crop_image(x, kind="GC", TB=True)
        x = x[1:1400]
        x = np.vstack([x,np.expand_dims(x[-1],axis=0)])
        X_train.append(x)
    X_train = np.array(X_train)
    
    X_test=[]
    for path in test_path:
        x = get_image_data(sorted(glob.glob(f"{path}/*.tif")))
        x = crop_image(x, kind="GC", TB=True)
        x = x[1:1400]
        x = np.vstack([x,np.expand_dims(x[-1],axis=0)])
        X_test.append(x)
    X_test = np.array(X_test)

    return X_train,X_test
