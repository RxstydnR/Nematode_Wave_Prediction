""" Time series data translation by many-in-one-out model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" 

import argparse
from pprint import pprint
from preparation import many2many,many2one

def main():

    # Many2Many
    if opt.pred_type=='Many2Many':
        many2many(
            train_date=opt.DATA_TRAIN_LIST, 
            test_date=opt.DATA_TEST_LIST,
            SAVE_PATH=opt.SAVE_PATH,
            input_movie=opt.input_movie,
            model_name=opt.model_name,
            augment=opt.augment,
            aug_times=opt.aug_times,
            epochs=opt.epochs,
            batchsize=opt.batchsize,
            )

    # Many2One
    elif opt.pred_type=='Many2One':

        many2one(
            train_date=opt.DATA_TRAIN_LIST, 
            test_date=opt.DATA_TEST_LIST,
            SAVE_PATH=opt.SAVE_PATH,
            n_sequence=opt.n_sequence,
            input_image=opt.input_image,
            model_name=opt.model_name,
            image_model=opt.image_model,
            pretrained=opt.image_model_pretrained,
            epochs=opt.epochs,
            batchsize=opt.batchsize,
            val_len=opt.val_len,
            )


if __name__ == '__main__':

    """ Running Example

        python main_wave_image.py \
            --DATA_TRAIN_LIST 0303-3 0323-1 0323-2 0323-3 \
            --DATA_TEST_LIST 0324-1 0324-2 0324-3 \
            --SAVE_PATH /data/Users/katafuchi/RA/Nematode/Result_wave_image \
            --pred_type Many2Many \
            --augment \
            --batchsize 64 \
            --epochs 300
    """
    model_choices=[
        "LSTM_TS", "RNN_TS", "GRU_TS",           # Many2One
        "LSTM_AE", "GRU_AE", "Time_AE", "Utime", # Many2Many
        "WaveModel", # Many2One with images
        ]

    parser = argparse.ArgumentParser(description='Wave Prediction.') 
    # Dataset
    parser.add_argument('--DATA_TRAIN_LIST', type=str, nargs='+', required=True, help='multiple dataset dates for training data.')
    parser.add_argument('--DATA_TEST_LIST', type=str, nargs='+', required=True, help='multiple dataset dates for test data.')
    parser.add_argument('--SAVE_PATH', type=str, required=True, help='path to save directory') 
    # model type
    parser.add_argument('--pred_type', choices=['Many2Many', 'Many2One'], required=True, help='Time series model type') 
    # Many2One
    parser.add_argument('--n_sequence', type=int, required=False, default=200, help='Length of data sequence.') 
    parser.add_argument('--val_len',    type=int, required=False, default=100, help='Length of validation data.') 
    # Many2Many
    parser.add_argument('--augment', action='store_true', required=False, help='Whether to apply data augmentation.') 
    parser.add_argument('--aug_times', type=int, required=False, default=50, help='how many times the training wave data is increased.') 
    # model
    parser.add_argument('--model', choices=model_choices, required=True, help='name of model') 
    # Many2One + image
    parser.add_argument('--input_image', action='store_true', required=False, help='Whether to use image feature extraction.') 
    parser.add_argument('--image_model', choices=["ResNet18","ResNet50"], default="ResNet18", required=False, help='Name of an image feature extractor') 
    parser.add_argument('--image_model_pretrained', action='store_true', help='Whether to use pretrained image feature extractor.')
    # Many2Many + movie
    parser.add_argument('--input_movie', action='store_true', required=False, help='Whether to use movie feature extraction.') 
        # model is resnet18 only
        # parser.add_argument('--movie_model', choices=["ResNet18"], default="ResNet18", required=False, help='path to save directory') 
        # parser.add_argument('--movie_model_pretrained', action='store_true', required=False, help='path to save directory')
    # training parameters
    parser.add_argument('--batchsize', type=int, required=False, default=64, help='training batch size') 
    parser.add_argument('--epochs', type=int, required=False, default=100, help='training epochs') 
    opt = parser.parse_args() 
    pprint(vars(opt))

    # make save folder
    os.makedirs(opt.SAVE_PATH, exist_ok=True)

    # check the dataset
    assert len(opt.DATA_TRAIN_LIST)>0,"Set at least one training dataset."
    assert len(opt.DATA_TEST_LIST)>0,"Set at least one test dataset."
    assert len(set(opt.DATA_TRAIN_LIST)&set(opt.DATA_TEST_LIST))==0,"Cannot set the same data for training and test."

    assert not (opt.input_movie==opt.augment==True), "Cannot apply data augmentation to image data sequence."
    assert not ((opt.model_name=="Time_AE") & (opt.input_movie==True)), "Cannot apply data augmentation to image data sequence."
    
    main()
    