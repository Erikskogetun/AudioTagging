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

import convnet_models

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # Add sub-parser for training
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model',
                              choices=['vgg13'],
                              default='vgg13',
                              )
    parser_train.add_argument('--input',
                              default='input/')
    parser_train.add_argument('--scale', action='store_true')
    parser_train.add_argument('--epochs', type=int, default=1)
    # parser_train.add_argument('--fold', type=int, default=-1)
    # parser_train.add_argument('--class_weight', action='store_true')
    # parser_train.add_argument('--sample_weight', type=float)

    # Add sub-parser for inference
    # parser_predict = subparsers.add_parser('predict')
    # parser_predict.add_argument('dataset', choices=['training', 'test'])
    # parser_predict.add_argument('--fold', type=int, default=-1)

    # Add sub-parser for evaluation
    # parser_evaluate = subparsers.add_parser('evaluate')
    # parser_evaluate.add_argument('fold', type=int)

    args = parser.parse_args()

    if args.mode == 'train':
        if args.scale:
            print("scaling...")
        print('Input path: ' + args.input)
        train(args.model, args.input, args.epochs)
    else:
        print("Incorrect command line arguments")

def train(model, input_path, epochs):
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
    # np.random.seed(cfg.initial_seed)

    # Load training data
    targets = np.load(input_path + 'first_chunk_targets.npy')
    specs = np.load(input_path + 'first_chunk_specs.npy')

    specs = np.reshape(specs, specs.shape + (1,))

    # TODO: use model param
    model = convnet_models.vgg13(specs.shape[1:], targets.shape[1])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x=specs,
              y=targets,
              batch_size=256,
              epochs=epochs,
              validation_split=0.2)

    model.save('trained_model.h5')


if __name__ == '__main__':
    sys.exit(main())
