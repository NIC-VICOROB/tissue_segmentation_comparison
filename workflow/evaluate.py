from sklearn.model_selection import LeaveOneOut

from architectures.arch_creator import generate_model
from utils.ioutils import read_dataset, save_volume
from utils.training_testing_utils import split_train_val, build_training_set, build_testing_set
from utils.callbacks import generate_output_filename

def run_evaluation_in_dataset(gen_conf, train_conf) :
    if train_conf['dataset'] == 'iSeg2017' :
        return evaluate_using_loo(gen_conf, train_conf)
    if train_conf['dataset'] == 'IBSR18' :
        return evaluate_using_loo(gen_conf, train_conf)
    if train_conf['dataset'] == 'MICCAI2012' :
        pass

def evaluate_using_loo(gen_conf, train_conf) :
    num_volumes = gen_conf['num_volumes']
    input_data, labels = read_dataset(gen_conf, train_conf)

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(range(num_volumes)):
        print train_index, test_index

        model = train_model(
            gen_conf, train_conf, input_data[train_index], labelstrain_index], test_index[0] + 1)

        x_test = build_testing_set(train_conf, input_data[test_index])
        test_model(gen_conf, train_conf, x_test, model, test_index[0]+1)

        del x_test

    return True

def train_model(
    gen_conf, train_conf, input_data, labels, case_name) :
    train_index, val_index = split_train_val(
        range(len(input_data)), train_conf['validation_split'])

    x_train, y_train = build_training_set(
        train_conf, input_data[train_index], labels[train_index])
    x_val, y_val = build_training_set(
        train_conf, input_data[val_index], labels[val_index])

    callbacks = generate_callbacks(
        gen_conf, train_conf, case_name)

    model = __train_model(
        gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name, callbacks)

    return model

def __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name, callbacks) :
    model = generate_model(gen_conf, train_conf)

    model.fit(
        x_train, y_train,
        epochs=train_conf['num_epochs'],
        validation_data=(x_val, y_val),
        verbose=train_conf['verbose'],
        callbacks=callbacks)

    model_filename = generate_output_filename(
        gen_conf['model_path'], train_conf['dataset'], case_name, train_conf['approach'], 'h5')

    model.load_weights(model_filename)

    return model

def test_model(gen_conf, train_conf, x_test, model, case_idx) :
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']

    pred = segmenter.predict(x_test, verbose=1)
    pred = pred.reshape((len(pred), ) + output_shape + (num_classes, ))

    rec_vol = reconstruct_volume(gen_conf, train_conf, pred)

    save_volume(gen_conf, train_conf, rec_vol, case_idx)