import itertools

import numpy as np

from .general_utils import pad_both_sides

def reconstruct_volume(gen_conf, train_conf, patches) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension']
    expected_shape = dataset_info['dimensions']
    extraction_step = train_conf['extraction_step_test']
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    rec_volume = perform_voting(
        dimension, patches, output_shape, expected_shape, extraction_step, num_classes)

    return rec_volume

def perform_voting(dimension, patches, output_shape, expected_shape, extraction_step, num_classes) :
    vote_img = np.zeros(expected_shape + (num_classes, ))

    coordinates = generate_indexes(
        dimension, output_shape, extraction_step, expected_shape)

    if dimension == 2 : 
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        selection += [slice(None)]
        vote_img[selection] += patches[count]

    return np.argmax(vote_img[:, :, :, 1:], axis=3) + 1

def generate_indexes(dimension, output_shape, extraction_step, expected_shape) :
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    
    return itertools.product(*idxs)