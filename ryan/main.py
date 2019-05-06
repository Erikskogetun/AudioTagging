# https://github.com/tqbl/dcase2018_task2/blob/master/task2/main.py
import argparse
import glob
import os
import sys

import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# import config as cfg
# import file_io as io
# import utils

from sklearn.preprocessing import StandardScaler

import convnet_models

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

"""
TODO:
Assume this model gets us to decent validation accuracy. What else do we need to do before running final tests?:
    Test other models than vgg13?
    
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
    parser_train.add_argument('--scale', action='store_true')
    parser_train.add_argument('--epochs', type=int, default=1)
    parser_train.add_argument('--batch', type=int, default=None)

    args = parser.parse_args()

    if args.mode == 'train':
        print('Input path: ' + args.input)
        train(args.model, args.input, args.output, args.epochs, args.scale, args.batch)
    else:
        print("Incorrect command line arguments")


def train(model_name, input_path, output_file, epochs, scale_input, batch_size):
    """
    Train the neural network model. Saves trained model to output/[input_path].h5
    Args:
        model_name (str): The neural network architecture.
        input_path (str): The path to the input data files.
        output_file (str): The filename to save the fitted model to. (In output/ dir)
        epochs (int): The number of epochs to train the model for.
        scale_input(bool): True indicates that the input will be scaled before training.
        batch_size(int): The batch size to use while training.
    Note:
        For reproducibility, the random seed is set to a fixed value.
    """
    # Try to create reproducible results
    np.random.seed(1234)

    # Load training data
    targets = np.load(input_path + 'first_chunk_targets.npy')
    specs = np.load(input_path + 'first_chunk_specs.npy')

    # Scale input if specified
    if scale_input:
        print('Scaling input...', end='')
        x, y, z = specs.shape
        specs = specs.reshape((x * y, z))
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(specs)
        specs = scaler.transform(specs).reshape((x, y, z))
        print('Done!')

    # Reshape specs to conform to model expected dimensions.
    # TODO: investigate why this is needed. Last dim is 1? (1 Channels?)
    specs = np.reshape(specs, specs.shape + (1,))

    # Get model from convnet_models.py
    if model_name == 'vgg13':
        model = convnet_models.vgg13(specs.shape[1:], targets.shape[1])
    else:
        print('Specified model does not exist!: ' + model_name)
        return

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Fit the model
    model.fit(x=specs,
              y=targets,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2)

    # Save the model
    if not os.path.exists('output'):
        os.makedirs('output')
    model.save('output/' + output_file + '.h5')


if __name__ == '__main__':
    sys.exit(main())
