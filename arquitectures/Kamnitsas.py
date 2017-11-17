from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv3D, Cropping2D, Cropping3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_kamnitsas_model(configuration) :
    activation = configuration['activation']
    dimension = configuration['dimension']
    num_classes = configuration['num_classes']
    num_modalities = configuration['num_modalities']
    output_shape = configuration['output_shape']
    patch_shape = configuration['patch_shape']

    loss = configuration['loss']
    metrics = configuration['metrics']
    optimizer = configuration['optimizer']

    input_shape = (num_modalities, ) + patch_shape

    assert dimension in [2, 3]

    model = generate_kamnitsas_model(
        dimension, num_classes, input_shape, output_shape, activation)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def generate_kamnitsas_model(dimension, num_classes, input_shape, output_shape, activation) :
    normal_res_input = Input(shape=input_shape[0])
    low_res_input = Input(shape=input_shape[1])

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

    return Model(inputs=[normal_res_input, low_res_input], outputs=[pred])

def get_conv_core(dimension, input, num_filters) :
    x = None
    kernel_size = (5, 5) if dimension == 2 else (5, 5, 5)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size)(input)
        x = PReLU()(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size)(input)
        x = PReLU()(x)

    return x

def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    return PReLU()(fc)

def get_deconv_layer(dimension, input, num_filters) :
    pool_size = (3, 3) if dimension == 2 else (3, 3, 3)
    strides = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2:
        return Conv2DTranspose(pool_size=pool_size, strides=strides)(input)
    else :
        return Conv3DTranspose(pool_size=pool_size, strides=strides)(input)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)