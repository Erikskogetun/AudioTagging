import os
import numpy as np
import keras.backend as K
import keras.callbacks
from keras.callbacks import ModelCheckpoint
from random import randint
# from sklearn.preprocessing import StandardScaler
import convnet_models


def train(model_name, input_path, output_file, epochs, batch_size, val_split, extra_chunks, mix_order):
    """
    Train the neural network model. Saves trained model to output/[input_path].h5
    Args:
        model_name (str): The neural network architecture.
        input_path (str): The path to the input data files.
        output_file (str): The filename to save the fitted model to. (In output/ dir)
        epochs (int): The number of epochs to train the model for.
        # scale_input (bool): True indicates that the input will be scaled before training.
        batch_size (int): The batch size to use while training.
        val_split (float): The fraction of training data to be used as validation data.
        extra_chunks (bool): True indicates to append extra_chunked_specs and extra_chunked_targets to training data.
        # generate_mixes (bool): True indicates to add specs together to make pairs of inputs.
        mix_order (int): None indicates to not load any mix files. Otherwise specifies max order of mix files to load.
    Note:
        For reproducibility, the random seed is set to a fixed value.
    """
    # Try to create reproducible results
    np.random.seed(1234)

    # Get lists of all target and spec files to use in training.
    target_files = [input_path + 'train_main_labels.npy']  # Ordered list of files containing target label vectors to train on.
    spec_files = [input_path + 'train_main_chunks.npy']  # Ordered list of files containing specs to train on.

    if extra_chunks:
        target_files.append(input_path + 'train_extra_labels.npy')
        spec_files.append(input_path + 'train_extra_chunks.npy')
    if mix_order:
        for i in range(2, mix_order + 1):
            spec_files.append(input_path + 'mixes_chunks_' + str(i) + '.npy')
            target_files.append(input_path + 'mixes_labels_' + str(i) + '.npy')
    print("Loading target files: " + str(target_files))
    print("Loading spec files: " + str(spec_files))

    assert len(target_files) == len(spec_files)

    # Load files
    targets = np.load(target_files[0])
    specs = np.load(spec_files[0])
    for i in range(1, len(target_files)):
        a = np.load(target_files[i])
        b = np.load(spec_files[i])
        targets = np.concatenate((targets, a))
        specs = np.concatenate((specs, b))

    assert len(targets) == len(specs)

    """
    Try just adding spectrograms as data mixing technique.
    """
    # if generate_mixes:
    #     orig_len = len(specs)
    #     for i in range(orig_len):
    #         if i % 25 == 0:
    #             print("\r", 'Generating ' + str(i) + ' of ' + str(orig_len), end="")
    #         j = randint(0, orig_len - 1)
    #         # If we happened to pick the same index, just skip this chunk.
    #         if i == j:
    #             continue
    #         spec_a = specs[i]
    #         spec_b = specs[j]
    #         spec_new = np.add(spec_a, spec_b)
    #         target_new = np.maximum(targets[i], targets[j])
    #         specs = np.concatenate((specs, spec_new.reshape((1,) + spec_new.shape)))
    #         targets = np.concatenate((targets, target_new.reshape((1,) + target_new.shape)))
    #
    #     print("Augmented data with mixes. From " + str(orig_len) + " samples to " + str(len(specs)) + " samples.")

    # Reshape specs to conform to model expected dimensions.
    # TODO: investigate why this is needed. Last dim is 1? (1 Channels?)
    specs = np.reshape(specs, specs.shape + (1,))

    # Get model from convnet_models.py
    if model_name == 'vgg13':
        model = convnet_models.vgg13(specs.shape[1:], targets.shape[1])
    else:
        print('Specified model does not exist!: ' + model_name)
        return

    # I think this is reporting 'true' accuracy, that is, if the two labels match exactly (after thresholding at 0.5)
    def full_multi_label_metric(y_true, y_pred):
        comp = K.equal(y_true, K.round(y_pred))
        return K.cast(K.all(comp, axis=-1), K.floatx())

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', full_multi_label_metric])

    checkpoint_filepath = 'train_output/' + output_file + '.best.h5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(x=specs,
              y=targets,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=val_split,
              callbacks=callbacks_list)

    # Save the model
    if not os.path.exists('train_output'):
        os.makedirs('train_output')
    model.save('train_output/' + output_file + '.h5')
