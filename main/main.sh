python main_wave_image.py \
    --DATA_TRAIN_LIST 0323-1 0323-2 0323-3 0323-4 0323-5 0323-6 0323-7 0323-8 0323-9 0323-10 0323-11 0323-12 0323-13 0323-14 0323-15 0323-16 0323-17 0323-18\
    --DATA_TEST_LIST  0324-1 0324-2 0324-3 0324-4 0324-5 0324-6 0324-7 0324-8 0324-9 0324-10 0324-11 0324-12 0324-13 0324-14 0324-15 0324-16 0324-17 0324-18 \
    --SAVE_PATH /data/Users/katafuchi/RA/Nematode/Result_wave_image \
    --pred_type Many2Many \
    --model Utime \
    --augment \
    --batchsize 64 \
    --epochs 300