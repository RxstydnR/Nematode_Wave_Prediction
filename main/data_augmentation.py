import numpy as np

def data_augmentation_xy(x,y):
    """ Apply data augmentation to both Input wave X and output wave Y.
        This is an exteded version of 'Time Series Augmentation' (https://github.com/uchidalab/time_series_augmentation).

    Args:
        x (array): input wave
        y (array): output wave

    Note:    
        - Shape of x and y must be (number of data, wave length, number of features).
        - And the number of data of x and y must be the same.
        

    Returns:
        ret_x (array): Data-augmented x
        ret_y (array): Data-augmented y
    """

    def scaling(x, y, sigma=0.1):
        # https://arxiv.org/pdf/1706.00527.pdf
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
        return np.multiply(x, factor[:,np.newaxis,:]),np.multiply(y, factor[:,np.newaxis,:])

    def magnitude_warp(x, y, sigma=0.2, knot=4):
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
        warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T

        ret_x = np.zeros_like(x)
        ret_y = np.zeros_like(y)

        for i, pat in enumerate(x):
            warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
            ret_x[i] = pat * warper

        for i, pat in enumerate(y):
            warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
            ret_y[i] = pat * warper

        return ret_x, ret_y

    def time_warp(x, y, sigma=0.2, knot=4):
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
        warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T

        ret_x = np.zeros_like(x)
        ret_y = np.zeros_like(y)

        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (x.shape[1]-1)/time_warp[-1]
                ret_x[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T

        for i, pat in enumerate(y):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (x.shape[1]-1)/time_warp[-1]
                ret_y[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T

        return ret_x, ret_y

    def window_slice(x, y, reduce_ratio=0.9):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x
        starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
        ends = (target_len + starts).astype(int)

        ret_x = np.zeros_like(x)
        ret_y = np.zeros_like(y)

        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                ret_x[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T

        for i, pat in enumerate(y):
            for dim in range(x.shape[2]):
                ret_y[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T

        return ret_x, ret_y

    def window_warp(x, y, window_ratio=0.1, scales=[0.5, 2.]):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        warp_scales = np.random.choice(scales, x.shape[0])
        warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret_x = np.zeros_like(x)
        ret_y = np.zeros_like(y)

        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[:window_starts[i],dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
                end_seg = pat[window_ends[i]:,dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))                
                ret_x[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T

        for i, pat in enumerate(y):
            for dim in range(x.shape[2]):
                start_seg = pat[:window_starts[i],dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
                end_seg = pat[window_ends[i]:,dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))                
                ret_y[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T

        return ret_x, ret_y

    

    # Scaling
    if np.random.random() < 1:
        x, y = scaling(x, y, sigma=0.1)

    aug_select_n = np.random.randint(4)

    # Magnitude_warp
    if aug_select_n==0: 
        x, y = magnitude_warp(x, y, sigma=0.2, knot=4)

    # Time warp
    elif aug_select_n==1: 
        x, y = time_warp(x, y, sigma=0.2, knot=4)

    # Window slice
    elif aug_select_n==2:
        x, y = window_slice(x, y, reduce_ratio=0.9)

    # Window_warp
    elif aug_select_n==3:
        x, y = window_warp(x, y, window_ratio=0.1, scales=[0.5, 2.])

    return x,y


