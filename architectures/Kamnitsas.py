import numpy as np

from keras import backend as K
from keras.layers import Activation, Input, AveragePooling2D, AveragePooling3D
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, Cropping3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_kamnitsas_model(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    num_modalities = gen_conf['dataset_info'][dataset]['modalities']
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    assert dimension in [2, 3]

    model = __generate_kamnitsas_model(
        dimension, num_classes, input_shape, output_shape, activation)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def __generate_kamnitsas_model(dimension, num_classes, input_shape, output_shape, activation) :   
    original_input = Input(shape=input_shape)

    normal_res_input = get_cropping_layer(dimension, original_input, crop_size=(8, 8))
    low_res_input = get_low_res_layer(dimension, original_input)

    normal_res = get_conv_core(dimension, normal_res_input, 30)
    normal_res = get_conv_core(dimension, normal_res, 40)
    normal_res = get_conv_core(dimension, normal_res, 40)
    normal_res = get_conv_core(dimension, normal_res, 50)

    low_res = get_conv_core(dimension, low_res_input, 30)
    low_res = get_conv_core(dimension, low_res, 40)
    low_res = get_conv_core(dimension, low_res, 40)
    low_res = get_conv_core(dimension, low_res, 50)
    low_res = get_deconv_layer(dimension, low_res, 50)

    concat = concatenate([normal_res, low_res], axis=1)

    fc = get_conv_fc(dimension, concat, 150)
    fc = get_conv_fc(dimension, fc, 150)

    pred = get_conv_fc(dimension, fc, num_classes)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[original_input], outputs=[pred])

def get_conv_core(dimension, input, num_filters) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size)(input)
        x = PReLU()(x)
        x = Conv2D(num_filters, kernel_size=kernel_size)(x)
        x = PReLU()(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size)(input)
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

def get_low_res_layer(dimension, input) :
    if dimension == 2 :
        return AveragePooling2D()(input)
    else :
        return AveragePooling3D()(input)

def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    return PReLU()(fc)

def get_deconv_layer(dimension, input, num_filters) :
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)
    strides = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)