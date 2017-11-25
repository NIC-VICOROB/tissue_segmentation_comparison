import numpy as np

from keras.utils import np_utils

from .extraction import extract_patches
from .general_utils import pad_both_sides

def split_train_val(train_indexes, validation_split) :
    N = len(train_indexes)
    val_volumes = np.int32(np.ceil(N * validation_split))
    train_volumes = N - val_volumes

    return train_indexes[:train_volumes], train_indexes[train_volumes:]

def build_training_set(gen_conf, train_conf, input_data, labels) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    dimension = train_conf['dimension']
    extraction_step = train_conf['extraction_step']
    modalities = dataset_info['modalities']
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    label_selector = determine_label_selector(dimension, patch_shape, output_shape)
    print label_selector
    minimum_non_bg = bg_discard_percentage * np.prod(output_shape)

    data_patch_shape = (modalities, ) + patch_shape
    data_extraction_step = (modalities, ) + extraction_step
    output_patch_shape = (np.prod(output_shape), num_classes)

    x = np.zeros((0, ) + data_patch_shape)
    y = np.zeros((0, ) + output_patch_shape)
    for idx in range(len(input_data)) :
        y_length = len(y)

        pad_size = ()
        for dim in range(dimension) :
            pad_size += (patch_shape[dim] // 2, )

        label_vol = pad_both_sides(dimension, labels[idx, 0], pad_size)
        input_vol = pad_both_sides(dimension, input_data[idx], pad_size)

        label_patches = extract_patches(dimension, label_vol, patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)

        valid_idxs = np.where(np.sum(label_patches != 0, axis=sum_axis) >= minimum_non_bg)

        label_patches = label_patches[valid_idxs]

        N = len(label_patches)

        x = np.vstack((x, np.zeros((N, ) +  data_patch_shape )))
        y = np.vstack((y, np.zeros((N, ) + output_patch_shape )))

        for i in range(N) :
            tmp = np_utils.to_categorical(label_patches[i].flatten(), num_classes)
            y[i + y_length] = tmp

        del label_patches

        data_train = extract_patches(dimension, input_vol, data_patch_shape, data_extraction_step)
        x[y_length:] = data_train[valid_idxs]
        del data_train
    return x, y

def build_testing_set(gen_conf, train_conf, input_data) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension']
    extraction_step = train_conf['extraction_step_test']
    modalities = dataset_info['modalities']
    patch_shape = train_conf['patch_shape']

    data_patch_shape = (modalities, ) + patch_shape
    data_extraction_step = (modalities, ) + extraction_step

    return extract_patches(dimension, input_data, data_patch_shape, data_extraction_step)

def determine_label_selector(dimension, patch_shape, output_shape) :
    ndim = len(patch_shape)
    patch_shape_equal_output_shape = patch_shape == output_shape

    slice_none = slice(None)
    if not patch_shape_equal_output_shape :
        return [slice_none] + [slice(output_shape[i], patch_shape[i] - output_shape[i]) for i in range(ndim)]
    else :
        return [slice_none for i in range(ndim)]