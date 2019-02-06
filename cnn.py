from keras.layers import Conv2D, MaxPool2D, Input, Flatten, Dense
from keras.layers.core import Activation
from keras.models import Model

import numpy as np


def conv_relu_pool(num_filter, kernel_size, pool_size):
    def conv_func(x):
        x = Conv2D(num_filter, kernel_size, strides=(1,1), padding='same')(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size)(x)
        return x
    return conv_func


def create_cnn(input_shape, hidden_dim, output_dim):
    # Encoder
    input = Input(shape=input_shape, name='image')
    
    x = conv_relu_pool(3, (3,3), (2,2))(input)
    x = conv_relu_pool(3, (3,3), (2,2))(x)
    x = conv_relu_pool(3, (3,3), (2,2))(x)
    x = Flatten()(x)
    x = Dense(hidden_dim, activation='relu')(x)
    output = Dense(output_dim, activation='sigmoid')(x)

    cnn_model = Model(input, output)
    return cnn_model


