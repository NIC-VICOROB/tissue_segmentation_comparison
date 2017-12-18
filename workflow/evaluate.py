import numpy as np

from sklearn.model_selection import LeaveOneOut

from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks, generate_output_filename
from utils.ioutils import read_dataset, save_volume, save_volume_MICCAI2012
from utils.reconstruction import reconstruct_volume
from utils.training_testing_utils import split_train_val, build_training_set, build_testing_set
from utils.general_utils import pad_both_sides

def run_evaluation_in_dataset(gen_conf, train_conf) :
    if train_conf['dataset'] == 'iSeg2017' :
        return evaluate_using_loo(gen_conf, train_conf)
    if train_conf['dataset'] == 'IBSR18' :
        return evaluate_using_loo(gen_conf, train_conf)
    if train_conf['dataset'] == 'MICCAI2012' :
        return evaluate_using_training_testing_split(gen_conf, train_conf)

def evaluate_using_training_testing_split(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_modalities = dataset_info['modalities']
    num_volumes = dataset_info['num_volumes']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    output_shape = train_conf['output_shape']

    input_data, labels = read_dataset(gen_conf, train_conf)

    model, mean, std = train_model(
        gen_conf, train_conf, input_data[:num_volumes[0]], labels[:num_volumes[0]], 1)

    testing_set = [1003, 1019, 1038, 1107, 1119, 1004, 1023, 1039, 1110, 1122, 1005,
        1024, 1101, 1113, 1125, 1018, 1025, 1104, 1116, 1128]

    test_indexes = range(num_volumes[0], num_volumes[0] + num_volumes[1])

    for idx, test_index in enumerate(test_indexes) :
        test_vol = normalise_volume(input_data[test_index], num_modalities, mean, std)

        if patch_shape != output_shape :
            pad_size = ()
            for dim in range(dimension) :
                pad_size += (output_shape[dim], )
            test_vol = pad_both_sides(dimension, test_vol, pad_size)

        x_test = build_testing_set(gen_conf, train_conf, test_vol)
        rec_vol = test_model(gen_conf, train_conf, x_test, model)

        save_volume_MICCAI2012(gen_conf, train_conf, rec_vol, testing_set[idx])

        del x_test


def evaluate_using_loo(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_modalities = dataset_info['modalities']
    num_volumes = dataset_info['num_volumes']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    output_shape = train_conf['output_shape']

    input_data, labels = read_dataset(gen_conf, train_conf)

    if dataset == 'IBSR18' :
        for case in range(num_volumes) :
            input = input_data[case, 0]
            input_data[case, 0] -= input[input != 0].mean()
            input_data[case, 0] /= input[input != 0].std()

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(range(num_volumes)):
        print train_index, test_index

        model, mean, std = train_model(
            gen_conf, train_conf, input_data[train_index], labels[train_index], test_index[0] + 1)

        test_vol = normalise_volume(input_data[test_index[0]], num_modalities, mean, std)

        if patch_shape != output_shape :
            pad_size = ()
            for dim in range(dimension) :
                pad_size += (output_shape[dim], )
            test_vol = pad_both_sides(dimension, test_vol, pad_size)

        x_test = build_testing_set(gen_conf, train_conf, test_vol)
        rec_vol = test_model(gen_conf, train_conf, x_test, model)

        save_volume(gen_conf, train_conf, rec_vol, test_index[0]+1)

        del x_test

    return True

def train_model(
    gen_conf, train_conf, input_data, labels, case_name) :
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities']
    output_shape = train_conf['output_shape']
    
    train_index, val_index = split_train_val(
        range(len(input_data)), train_conf['validation_split'])

    mean, std = compute_statistics(input_data, num_modalities)
    input_data = normalise_set(input_data, num_modalities, mean, std)

    if train_conf['num_epochs'] != 0 :
        x_train, y_train = build_training_set(
            gen_conf, train_conf, input_data[train_index], labels[train_index])
        x_val, y_val = build_training_set(
            gen_conf, train_conf, input_data[val_index], labels[val_index])

        if approach == "DolzMulti" :
            slicer = [slice(None), slice(x_train.shape[1]-1, x_train.shape[1])]
            slicer += [slice(9, -9) for i in range(dimension)]
            print slicer
            train_mask = (x_train[slicer] != 0).reshape((-1, y_train.shape[1]))
            val_mask = (x_val[slicer] != 0).reshape((-1, y_val.shape[1]))
        
        if approach == "Kamnitsas" :
            slicer = [slice(None), slice(x_train.shape[1]-1, x_train.shape[1])]
            slicer += [slice(16, -16) for i in range(dimension)]
            train_mask = (x_train[slicer] != 0).reshape((-1, y_train.shape[1]))
            val_mask = (x_val[slicer] != 0).reshape((-1, y_val.shape[1]))

        if approach == "Guerrero" or approach == "Cicek" :
            train_mask = (x_train[:, -1] != 0).reshape((-1, y_train.shape[1]))
            val_mask = (x_val[:, -1] != 0).reshape((-1, y_val.shape[1]))

        callbacks = generate_callbacks(
            gen_conf, train_conf, case_name)

        __train_model(
            gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name, callbacks, train_mask, val_mask)

    model = read_model(gen_conf, train_conf, case_name)

    return model, mean, std

def __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name, callbacks, train_mask=None, val_mask=None) :
    model = generate_model(gen_conf, train_conf)

    model.fit(
        x_train, y_train,
        epochs=train_conf['num_epochs'],
        validation_data=(x_val, y_val, val_mask),
        verbose=train_conf['verbose'],
        sample_weight=train_mask,
        callbacks=callbacks)

    return True

def read_model(gen_conf, train_conf, case_name) :
    model = generate_model(gen_conf, train_conf)

    model_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'h5')

    model.load_weights(model_filename)

    return model

def test_model(gen_conf, train_conf, x_test, model) :
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']

    pred = model.predict(x_test, verbose=1)
    pred = pred.reshape((len(pred), ) + output_shape + (num_classes, ))

    return reconstruct_volume(gen_conf, train_conf, pred)

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std

def normalise_set(input_data, num_modalities, mean, std) :
    print input_data.shape
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] -= mean[modality]
            input_data_tmp[vol_idx, modality] /= std[modality]
    return input_data_tmp

def normalise_volume(input_data, num_modalities, mean, std) :
    print input_data.shape
    input_data_tmp = np.copy(input_data)
    for modality in range(num_modalities) :
        input_data_tmp[modality] -= mean[modality]
        input_data_tmp[modality] /= std[modality]
    return input_data_tmp