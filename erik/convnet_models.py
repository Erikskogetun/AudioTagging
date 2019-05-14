# https://github.com/tqbl/dcase2018_task2/blob/master/task2/convnet.py
# import keras.backend as K
# import keras
from keras.layers import BatchNormalization
# from keras.layers import Bidirectional
from keras.layers import Conv2D
from keras.layers import Dense
# from keras.layers import GRU
from keras.layers import Input
# from keras.layers import Lambda
from keras.layers import MaxPooling2D
# from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

def vgg13(input_shape, n_classes):
    """Create a VGG13-style model.
    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of classes for classification.
    Returns:
        A Keras model of the VGG13 architecture.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')

    x = _conv_block(input_tensor, n_filters=64)
    x = _conv_block(x, n_filters=128)
    x = _conv_block(x, n_filters=256)
    x = _conv_block(x, n_filters=512)
    x = _conv_block(x, n_filters=512)

    x = GlobalAveragePooling2D()(x)

    x = Dense(n_classes, activation='sigmoid')(x)
    return Model(input_tensor, x, name='vgg13')

def _conv_block(x, n_filters, kernel_size=(3, 3), pool_size=(2, 2), **kwargs):
    """Apply two batch-normalized convolutions followed by max pooling.
    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of convolution filters.
        kernel_size (int or tuple): Convolution kernel size.
        pool_size (int or tuple): Max pooling parameter.
        kwargs: Other keyword arguments.
    Returns:
        tensor: The output tensor.
    """
    x = _conv_bn(x, n_filters, kernel_size, **kwargs)
    x = _conv_bn(x, n_filters, kernel_size, **kwargs)
    return MaxPooling2D(pool_size=pool_size)(x)


def _conv_bn(x, n_filters, kernel_size=(3, 3), **kwargs):
    """Apply a convolution operation followed by batch normalization.
    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of convolution filters.
        kernel_size (int or tuple): Convolution kernel size.
        kwargs: Other keyword arguments.
    Returns:
        tensor: The output tensor.
    """
    x = Conv2D(n_filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               **kwargs)(x)
    return BatchNormalization(axis=-1)(x)