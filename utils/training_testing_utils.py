import numpy as np

def split_train_val(train_indexes, validation_split) :
    N = len(train_indexes)
    val_volumes = np.int32(np.ceil(N * validation_split))
    train_volumes = N - val_volumes

    return train_indexes[:train_volumes], train_indexes[train_volumes:]

def build_training_set(train_conf, input_data, labels) :

    pass

def build_testing_set(train_conf, input_data) :
    pass