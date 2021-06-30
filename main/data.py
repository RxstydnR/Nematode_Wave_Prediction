import glob
import numpy as np
import pandas as pd
import tifffile
import cv2
import os

# Wave
def get_data(data_dir, kind="original"):
    
    print(f" \"{kind}\" data is selected.")

    # get data paths
    if kind == "original":
        data_GC_paths = sorted(glob.glob(f"{data_dir}/*/quantification-GC-original.npy"))
        data_pa_paths = sorted(glob.glob(f"{data_dir}/*/quantification-pa-original.npy"))
    elif kind == "preprocess":
        data_GC_paths = sorted(glob.glob(f"{data_dir}/*/quantification-GC_preprocess*.npy"))
        data_pa_paths = sorted(glob.glob(f"{data_dir}/*/quantification-pa_preprocess*.npy"))
    else:
        ValueError(f"{kind} is invalid. Choose original or preprocess.")

    assert len(data_GC_paths)==len(data_pa_paths),"The number of files does not match."
    
    def get_from_npy(paths):
        data=[]
        for path in paths:
            d = np.load(path,allow_pickle=True)
            d = d[:1399] # データ数を揃えるため
            data.append(d)
        return np.array(data)
    
    # get data array
    data_GC = get_from_npy(data_GC_paths)
    data_pa = get_from_npy(data_pa_paths)
    assert len(data_GC)==len(data_pa),"The number of files does not match."
    
    def check_wave_shape(name,data):
        wave_shapes = set([data[i].shape[0] for i in range(len(data))])
        assert len(wave_shapes)==1, f"{name} contains at least one or more wave data of different lengths."
    
    # check wave shape are the same
    check_wave_shape("data_GC",data_GC)
    check_wave_shape("data_pa",data_pa)
    
    return data_GC, data_pa


def get_data_name(data_dir):
    data_names = [os.path.basename(name) for name in sorted(glob.glob(f"{data_dir}/*"))]
    return np.array(data_names)


def create_timestep_sequences(X, Y, N_SEQUENCE, N_FEATURE):
    """ Generated training sequences for use in the model.

        Ex) When ... N_SEQUENCE=200, N_FEATURE=1

            Before X shape = (5, 1300)
            Before Y shape = (5, 1300)

            After X shape = (5500, 200, 1)
            After Y shape = (5500, 1)

    """
    X_sequences = []
    Y_sequences = []
    for i in range(len(X)):
        for j in range(len(X[i]) - N_SEQUENCE):
            X_sequences.append(X[i,j:(j+N_SEQUENCE)])
            Y_sequences.append(Y[i,j+N_SEQUENCE])

    X_sequences = np.array(X_sequences).reshape((-1,N_SEQUENCE,N_FEATURE,))
    Y_sequences = np.array(Y_sequences).reshape((-1,N_FEATURE,))
    
    print(f"Before X shape = {X.shape}")
    print(f"Before Y shape = {Y.shape}")
    print(f"After X shape = {X_sequences.shape}")
    print(f"After Y shape = {Y_sequences.shape}")
    
    return X_sequences,Y_sequences


def create_sequences(X,N_SEQUENCE,N_FEATURE):
    """ 
    """
    print(f"Before X shape = {X.shape}")
    X_seq = X.reshape((-1,N_SEQUENCE,N_FEATURE,))
    print(f"After X shape = {X_seq.shape}")
    
    return X_seq


def make_start_0(X):
    X_0s = []
    for i in range(X.shape[0]):
        X_0 = X[i]-X[i,0]
        X_0s.append(X_0)
    return np.array(X_0s)


# Image
def crop_image(X,kind,TB=True):
    
    assert X.ndim >= 3, "Must be multiple images array."
    assert (kind=="GC") or (kind=="pa"), f"kind {kind} is invalid."

    top, bottom = 78, 334
    border = 256

    # crop from top to bottom
    if TB==True: 
        X = X[:,top:bottom,:]

    # extract Nematode area (right(256:)? or left(:256)?)
    if kind == "pa":
        X = X[:,:,:border]
    elif kind == "GC":
        X = X[:,:,border:]    
    
    return X


def to_uint8(img):
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img


def get_image_data(Imgs_path, flip=True):
    Imgs=[]
    for path in Imgs_path:
        # TIFF to JPG
        img = tifffile.imread(path) 
        img = to_uint8(img)

        if flip:
            img = img[:,::-1] # flip

        if img.ndim>=3:
           img = img[:,:,0] 
        
        Imgs.append(img)
    return np.array(Imgs)


def resize_image(X,size=(256,256,1)):
    X_resized=[]
    for x in X:
        x = cv2.resize(x, size)
        X_resized.append(x)
    return np.array(X_resized)

