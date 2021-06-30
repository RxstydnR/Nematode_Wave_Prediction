import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from data import create_timestep_sequences
from data_augmentation import data_augmentation_xy
from util import show_history, show_results
from util import MA,SSA
from preparation import data_prepare, image_data_prepare

from model import LSTM_AE,GRU_AE, Time_AE
from model import LSTM_TS,GRU_TS, RNN_TS
from model import WaveModel,Utime


def many2many(
    train_date, 
    test_date,
    SAVE_PATH,
    input_movie,
    model_name,
    augment,
    aug_times,
    epochs,
    batchsize,
    ):
    
    print("Getting data ...",flush=True)    

    # get wave data
    X_train,X_test,Y_train,Y_test = data_prepare(train_date, test_date)

    # prepare image data
    if input_movie:
        X_train_img, X_test_img = image_data_prepare(train_date, test_date, size=(64,64))
        X_train_img = X_train_img[...,np.newaxis].transpose(0,2,3,1,4)
        X_test_img = X_test_img[...,np.newaxis].transpose(0,2,3,1,4)
        X_train_img = X_train_img.astype("float32")/255.
        X_test_img  = X_test_img.astype("float32")/255.

    # Data Augmentation (Warning!!: If use an image sequence, don't apply data augmentation to only wave data.)
    if augment==True:
        
        # Reshape for data augmentation
        X_train = X_train.reshape((-1,X_train.shape[1],1))
        X_test = X_test.reshape((-1,X_test.shape[1],1))
        Y_train = Y_train.reshape((-1,Y_train.shape[1],1))
        Y_test = Y_test.reshape((-1,Y_test.shape[1],1))

        X_train_aug, Y_train_aug = [],[]
        for _ in range(aug_times):
            X_train_, Y_train_ = data_augmentation_xy(X_train,Y_train)
            X_train_aug.append(X_train_)
            Y_train_aug.append(Y_train_)
        X_train = np.concatenate(X_train_aug,axis=0)
        Y_train = np.concatenate(Y_train_aug,axis=0)

    print("Check data shape...")
    if input_movie:
        print(f"Train : Wave {X_train.shape}, Img {X_train_img.shape}")
        print(f"Train : Wave {Y_train.shape}")
        print(f"Test : Wave {X_test.shape}, Img {X_test_img.shape}")
        print(f"Test : Wave {Y_test.shape}")
    else:
        print(f"Train : Wave {X_train.shape}")
        print(f"Train : Wave {Y_train.shape}")
        print(f"Test : Wave {X_test.shape}")
        print(f"Test : Wave {Y_test.shape}")
        
    
    print("Creating a model...",flush=True)
    N_SEQUENCE = X_train.shape[1]
    # Make model
    if model_name=="Time_AE":
        model = Time_AE(N_SEQUENCE=N_SEQUENCE, N_FEATURE=1).build_model()
    elif model_name=="LSTM_AE":
        model = LSTM_AE(N_SEQUENCE=N_SEQUENCE, N_FEATURE=1).build_model()
    elif model_name=="GRU_AE":
        model = GRU_AE(N_SEQUENCE=N_SEQUENCE, N_FEATURE=1).build_model()
    elif model_name=="Utime":
        model = Utime(N_SEQUENCE=N_SEQUENCE, N_FEATURE=1, movie_branch=input_movie, movie_input_shape=X_train_img.shape[1:]).build_model()
        
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # Training
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=20)

    if input_movie:
        X_train_inputs = [X_train,X_train_img]
        X_test_inputs = [X_test,X_test_img]
    else:
        X_train_inputs = X_train
        X_test_inputs = X_test
        
    print("Training ...",flush=True)
    history = model.fit(
                X_train_inputs, 
                Y_train, 
                epochs=epochs, 
                batch_size=batchsize, 
                shuffle=True, 
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=2
            )
    Y_pred = model.predict(X_test_inputs)

    X_test = X_test.reshape((-1,X_test.shape[1]))
    Y_test = Y_test.reshape((-1,Y_test.shape[1]))
    Y_pred = Y_pred.reshape((-1,Y_pred.shape[1]))
    
    print("Saving results ...",flush=True)
    show_history(history,SAVE_PATH)
    show_results(X_test,Y_test,Y_pred,SAVE_PATH)
    

def many2one(
    train_date, 
    test_date,
    SAVE_PATH,
    n_sequence,
    input_image,
    model_name,
    image_model,
    pretrained,
    epochs,
    batchsize,
    val_len, # # 各データセットの後半100データをvalidationに使用
    ):

    # get wave data
    X_train,X_test,Y_train,Y_test = data_prepare(train_date, test_date)
    X_test_org = X_test.copy()
    Y_test_org = Y_test.copy()

    # for validation data
    n_train = len(train_date)
    n_test = len(test_date)
    n_data = X_train.shape[-1] - n_sequence

    # make sequence data for training
    X_train,Y_train = create_timestep_sequences(X_train,Y_train,N_SEQUENCE=n_sequence, N_FEATURE=1)
    X_test, Y_test  = create_timestep_sequences(X_test, Y_test ,N_SEQUENCE=n_sequence, N_FEATURE=1)

    # Reshape for training
    X_train = X_train.reshape((-1,X_train.shape[1],1))
    X_test = X_test.reshape((-1,X_test.shape[1],1))
    Y_train = Y_train.reshape((-1,Y_train.shape[1],1))
    Y_test = Y_test.reshape((-1,Y_test.shape[1],1))

    # prepare image data
    if input_image:
        X_train_img, X_test_img = image_data_prepare(train_date, test_date)
        X_train_img = X_train_img[:,n_sequence:].reshape((-1,256,256,1)).astype("float32")/255.
        X_test_img  = X_test_img[:,n_sequence:].reshape((-1,256,256,1)).astype("float32")/255.

    # make validation data
    X_train_,Y_train_ = [],[]
    X_val,Y_val = [],[]
    X_train_img_, X_val_img = [],[]

    for i in range(1,n_train+1):
        # train
        x_train = X_train[(i-1)*n_data:i*n_data-val_len]
        y_train = Y_train[(i-1)*n_data:i*n_data-val_len]
        X_train_.append(x_train)
        Y_train_.append(y_train)
            
        # validation
        x_val = X_train[i*n_data-val_len:i*n_data]
        y_val = Y_train[i*n_data-val_len:i*n_data]
        X_val.append(x_val)
        Y_val.append(y_val)

        if input_image:
            # train
            x_train_img = X_train_img[(i-1)*n_data:i*n_data-val_len]
            X_train_img_.append(x_train_img)
            # validation
            x_val_img = X_train_img[i*n_data-val_len:i*n_data]
            X_val_img.append(x_val_img)

    X_train = np.array(X_train_).reshape((-1,n_sequence,1))
    Y_train = np.array(Y_train_).reshape((-1,1))
    X_val = np.array(X_val).reshape((-1,n_sequence,1))
    Y_val = np.array(Y_val).reshape((-1,1))

    if input_image:
        X_train_img = np.array(X_train_img_).reshape((-1,256,256,1))
        X_val_img = np.array(X_val_img).reshape((-1,256,256,1))

    # Make model
    if input_image:
        model = WaveModel(N_SEQUENCE=n_sequence, N_FEATURE=1, image_model=image_model, pretrained=pretrained).build_model()
        X_train_inputs = [X_train,X_train_img]
        X_test_inputs = [X_test, X_test_img]
    else:
        if model_name=="LSTM_TS":
            model = LSTM_TS(N_SEQUENCE=n_sequence, N_FEATURE=1).build_model()
        elif model_name=="GRU_TS":
            model = GRU_TS(N_SEQUENCE=n_sequence, N_FEATURE=1).build_model()
        elif model_name=="RNN_TS":
            model = RNN_TS(N_SEQUENCE=n_sequence, N_FEATURE=1).build_model()
        X_train_inputs = X_train
        X_test_inputs = X_test 
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # Training
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=30)
    history = model.fit(
                X_train_inputs,
                Y_train, 
                epochs=epochs, 
                batch_size=batchsize, 
                shuffle=False,
                validation_data=([X_val,X_val_img],Y_val),
                callbacks=[early_stopping],
                verbose=2
            )
    show_history(history, SAVE_PATH)

    # Prediction
    Y_pred = model.predict(X_test_inputs)

    # Reshape for plot
    Y_pred = Y_pred.reshape((n_test,-1))
    pad = np.zeros((n_test,n_sequence))
    pad[:] = np.nan
    Y_pred = np.concatenate([pad,Y_pred],axis=1)

    # save prediction
    show_results(X_test_org,Y_test_org,Y_pred,SAVE_PATH)
    np.save(os.path.join(SAVE_PATH,"pred"),Y_pred)