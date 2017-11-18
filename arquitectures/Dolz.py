import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv3D, Cropping2D, Cropping3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_dolz_multi_model(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    num_modalities = gen_conf['dataset_info'][dataset]['num_modalities']
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    assert dimension in [2, 3]

    model = generate_dolz_multi_model(
        dimension, num_classes, input_shape, output_shape, activation)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def generate_dolz_multi_model(dimension, num_classes, input_shape, output_shape, activation) :
    init_input = Input(shape=input_shape)

    x = get_conv_core(dimension, init_input, 25)
    y = get_conv_core(dimension, x, 50)
    z = get_conv_core(dimension, y, 75)

    x_crop = get_cropping_layer(dimension, x, crop_size=(6, 6))
    y_crop = get_cropping_layer(dimension, y, crop_size=(3, 3))

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = get_conv_fc(dimension, concat, 400)
    fc = get_conv_fc(dimension, fc, 200)
    fc = get_conv_fc(dimension, fc, 150)

    pred = get_conv_fc(dimension, fc, num_classes)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[init_input], outputs=[pred])

def get_conv_core(dimension, input, num_filters) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size)(input)
        x = PReLU()(x)
        x = Conv2D(num_filters, kernel_size=kernel_size)(x)
        x = PReLU()(x)
        x = Conv2D(num_filters, kernel_size=kernel_size)(x)
        x = PReLU()(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size)(input)
        x = PReLU()(x)
        x = Conv3D(num_filters, kernel_size=kernel_size)(x)
        x = PReLU()(x)
        x = Conv3D(num_filters, kernel_size=kernel_size)(x)
        x = PReLU()(x)

    return x

def get_cropping_layer(dimension, input, crop_size=(6, 6)) :
    cropping_param = (crop_size, crop_size) if dimension == 2 else (crop_size, crop_size, crop_size)

    if dimension == 2 :
        return Cropping2D(cropping=cropping_param)(input)
    else :
        return Cropping3D(cropping=cropping_param)(input)

def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    return PReLU()(fc)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)