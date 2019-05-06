# https://github.com/tqbl/dcase2018_task2/blob/master/task2/main.py
import argparse
import sys

# import pandas as pd
# from tqdm import tqdm
#
# import config as cfg
# import file_io as io
# import utils

import training
import data_synthesis

# python3 main.py train --scale --input ../../input_data/ --epochs 100 --batch 64

# Non-scaled:
# 3976/3976 [==============================] - 84s 21ms/step - loss: 4.6387 - acc: 0.0490 - val_loss: 5.9569 - val_acc: 0.0161
# Epoch 2/5
# 3976/3976 [==============================] - 75s 19ms/step - loss: 4.2357 - acc: 0.0732 - val_loss: 5.1581 - val_acc: 0.0634
# Epoch 3/5
# 3976/3976 [==============================] - 75s 19ms/step - loss: 3.9095 - acc: 0.1315 - val_loss: 6.2956 - val_acc: 0.0463
# Epoch 4/5
# 3976/3976 [==============================] - 75s 19ms/step - loss: 3.4736 - acc: 0.2394 - val_loss: 6.7605 - val_acc: 0.1298
# Epoch 5/5
# 3976/3976 [==============================] - 75s 19ms/step - loss: 3.0297 - acc: 0.3161 - val_loss: 5.0649 - val_acc: 0.1791

# Scaled:
# Epoch 1/45
# 2019-05-06 09:16:11.963971: I tensorflow/stream_executor/dso_loader.cc:153] successfully opened CUDA library libcublas.so.10.0 locally
# 3976/3976 [==============================] - 91s 23ms/step - loss: 4.6918 - acc: 0.0397 - val_loss: 7.9476 - val_acc: 0.0221
# Epoch 2/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 4.3235 - acc: 0.0760 - val_loss: 6.1382 - val_acc: 0.0423
# Epoch 3/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 4.0815 - acc: 0.0943 - val_loss: 7.1156 - val_acc: 0.0563
# Epoch 4/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 3.6667 - acc: 0.1715 - val_loss: 6.4663 - val_acc: 0.1237
# Epoch 5/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 3.2398 - acc: 0.2689 - val_loss: 7.6223 - val_acc: 0.1429
# ...
# Epoch 43/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 0.2689 - acc: 0.9193 - val_loss: 2.1712 - val_acc: 0.6076
# Epoch 44/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 0.2556 - acc: 0.9243 - val_loss: 2.1302 - val_acc: 0.5976
# Epoch 45/45
# 3976/3976 [==============================] - 76s 19ms/step - loss: 0.3484 - acc: 0.9095 - val_loss: 2.6337 - val_acc: 0.5352

# Looks like current model caps out at ~0.60 val acc

"""
TODO:
Assume this model gets us to decent validation accuracy. What else do we need to do before running final tests?:
    Test other models than vgg13?
        Committed!
    
    Choose how to deal with uneven sound file lengths.
    
    Decide what to do with unverified data set. 
        Currently using trusted dataset of size ~3900. I think we will be left with too few test samples?
        
    Generate mixed training sets with mixtures of size n = 1, 2, 3, 4, ...
        Should this be done on vm to save hdd space, upload/dl issues, ect?
        This depends on how we want to deal with uneven sound file lengths.
    
    Define function which tells us not only accuracy of model, but confusion matrices and accuracy per label count.
    
    Define function to load trained models and graph their metrics per epoch.
    
    Generate test set which is partitioned from training/val (CURRENTLY USING ALL DATA FOR TRAIN/VAL!).
    
    Define function to load trained model and evaluate it on test set.
    
    Think about ensembling at prediction stage. Could show model every chunk from audio file and sum prediction vectors.
"""


def main():
    # Parse which mode of use the user is specifying.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # Add sub-parser for training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model',
                              choices=['vgg13'],
                              default='vgg13')
    parser_train.add_argument('--input',
                              default='input/')
    parser_train.add_argument('--output', default='temp')
    # parser_train.add_argument('--scale', action='store_true')
    parser_train.add_argument('--epochs', type=int, default=1)
    parser_train.add_argument('--batch', type=int, default=None)
    parser_train.add_argument('--val_split', type=float, default=0.15)
    parser_train.add_argument('--extra_chunks', action='store_true')

    # Add sub-parser for generate_data
    parser_generate_data = subparsers.add_parser('generate_data')
    parser_generate_data.add_argument('--wavs_dir', default=None)
    parser_generate_data.add_argument('--output', default=None)
    parser_generate_data.add_argument('--chunk_size', type=int, default=128)
    parser_generate_data.add_argument('--test_frac', type=float, default=0.2)

    args = parser.parse_args()

    if args.mode == 'train':
        print('Input path: ' + args.input)
        training.train(args.model, args.input, args.output, args.epochs, args.batch, args.val_split, args.extra_chunks)
    elif args.mode == 'generate_data':
        data_synthesis.generate_data(args.wavs_dir, args.output, args.chunk_size, args.test_frac)
    else:
        print("Incorrect command line arguments")


if __name__ == '__main__':
    sys.exit(main())
