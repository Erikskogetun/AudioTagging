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

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # Add sub-parser for training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model',
                              choices=['vgg13'],
                              default='vgg13')
    parser_train.add_argument('--input',
                              default='input/')
    parser_train.add_argument('--scale', action='store_true')
    parser_train.add_argument('--epochs', type=int, default=1)
    parser_train.add_argument('--batch', type=int, default=None)

    args = parser.parse_args()

    if args.mode == 'train':
        print('Input path: ' + args.input)
        train(args.model, args.input, args.epochs, args.scale, args.batch)
    else:
        print("Incorrect command line arguments")

def train(model, input_path, epochs, scale_input, batch_size):
    """Train the neural network model.
    Args:
        model (str): The neural network architecture.
        fold (int): The fold to use for validation.
        use_class_weight (bool): Whether to use class-wise weights.
        noisy_sample_weight (float): Examples that are not verified are
            weighted according to this value.
    Note:
        For reproducibility, the random seed is set to a fixed value.
    """
    # Try to create reproducible results
    # TODO
    np.random.seed(1234)

    # Load training data
    targets = np.load(input_path + 'first_chunk_targets.npy')
    specs = np.load(input_path + 'first_chunk_specs.npy')

    if scale_input:
        print('Scaling input...', end='')

        x, y, z = specs.shape
        specs = specs.reshape((x * y, z))
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(specs)
        specs = scaler.transform(specs).reshape((x, y, z))
        print('Done!')

    specs = np.reshape(specs, specs.shape + (1,))

    # TODO: use model param
    model = convnet_models.vgg13(specs.shape[1:], targets.shape[1])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(x=specs,
    #           y=targets,
    #           epochs=epochs,
    #           batch_size=batch_size,
    #           validation_split=0.2)
    #
    # model.save('trained_model.h5')


if __name__ == '__main__':
    sys.exit(main())
