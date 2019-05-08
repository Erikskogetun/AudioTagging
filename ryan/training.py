import os
import numpy as np

import convnet_models

# from sklearn.preprocessing import StandardScaler


def train(model_name, input_path, output_file, epochs, batch_size, val_split, extra_chunks):
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
    Note:
        For reproducibility, the random seed is set to a fixed value.
    """
    # Try to create reproducible results
    np.random.seed(1234)

    # Load training data
    targets = np.load(input_path + 'train_main_labels.npy')
    specs = np.load(input_path + 'train_main_chunks.npy')
    assert len(targets) == len(specs)

    # If using extra chunks (more than one chunk per wav for longer wavs), load and append.
    if extra_chunks:
        extra_targets = np.load(input_path + 'train_extra_labels.npy')
        extra_specs = np.load(input_path + 'train_extra_chunks.npy')
        assert len(extra_targets) == len(extra_specs)

        specs = np.concatenate((specs, extra_specs))
        targets = np.concatenate((targets, extra_targets))
        assert len(specs) > len(extra_specs)

    """
    Scaling didn't appear to matter much and complicates evaluation and retraining. Removing for now.
    """
    # # Scale input if specified
    # if scale_input:
    #     print('Scaling input...', end='')
    #     x, y, z = specs.shape
    #     specs = specs.reshape((x * y, z))
    #     scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    #     scaler.fit(specs)
    #     specs = scaler.transform(specs).reshape((x, y, z))
    #     print('Done!')

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
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy', 'categorical_accuracy'])

    # Fit the model
    model.fit(x=specs,
              y=targets,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=val_split)

    # Save the model
    if not os.path.exists('train_output'):
        os.makedirs('train_output')
    model.save('train_output/' + output_file + '.h5')
