# https://github.com/tqbl/dcase2018_task2/blob/master/task2/main.py
import argparse
import sys

import training
import data_synthesis
import evaluation


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
    # parser_train.add_argument('--generate_mixes', action='store_true')
    parser_train.add_argument('--mix_order', type=int, default=None)

    # Add sub-parser for generate_data
    parser_generate_data = subparsers.add_parser('generate_data')
    parser_generate_data.add_argument('--input', default=None)
    parser_generate_data.add_argument('--output', default=None)
    parser_generate_data.add_argument('--chunk_size', type=int, default=128)
    parser_generate_data.add_argument('--test_frac', type=float, default=0.2)
    parser_generate_data.add_argument('--remove_silence', action='store_true')
    parser_generate_data.add_argument('--n_mels', type=int, default=64)
    parser_generate_data.add_argument('--generate_mixes', action='store_true')
    parser_generate_data.add_argument('--mix_order', type=int, default=2)
    parser_generate_data.add_argument('--debug_skip', action='store_true')

    # Add sub-parser for evaluation
    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('--model', type=str)
    parser_evaluate.add_argument('--test_set', type=str)
    parser_evaluate.add_argument('--argmax_zero_labels', action='store_true')
    # parser_evaluate.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == 'train':
        print('Input path: ' + args.input)
        training.train(args.model,
                       args.input,
                       args.output,
                       args.epochs,
                       args.batch,
                       args.val_split,
                       args.extra_chunks,
                       args.mix_order)
    elif args.mode == 'generate_data':
        data_synthesis.generate_data(args.input,
                                     args.output,
                                     args.chunk_size,
                                     args.test_frac,
                                     args.remove_silence,
                                     args.n_mels,
                                     args.generate_mixes,
                                     args.mix_order,
                                     args.debug_skip)
    elif args.mode == 'evaluate':
        evaluation.evaluate(args.model, args.test_set, args.argmax_zero_labels)
    else:
        print("Incorrect command line arguments")


if __name__ == '__main__':
    sys.exit(main())
