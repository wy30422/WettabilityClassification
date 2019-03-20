from keras.layers import Input, Flatten, Dense, BatchNormalization
from keras.layers.core import Activation
from keras.models import Model

import numpy as np


def fc_block(hidden_size):
    def fc_block_(x):
        x = Dense(hidden_size)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    return fc_block_


def create_fc(input_shape, output_dim):
    # Encoder
    input = Input(shape=input_shape, name='image')
    
    x = Flatten()(input)
    
    x = fc_block(512)(x)
    x = fc_block(256)(x)
    x = fc_block(128)(x)
    x = fc_block(128)(x)

    output = Dense(output_dim, activation='softmax')(x)

    fc_model = Model(input, output)
    return fc_model


