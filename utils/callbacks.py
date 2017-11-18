from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from configuration import general_configuration, training_configuration

def generate_callbacks(case_name) :
    file_pattern = '{}/{}/{:02}-{}.{}'
    model_filename = file_pattern.format(
        general_configuration['model_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        'h5')

    csv_filename = file_pattern.format(
        general_configuration['log_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        'cvs')

    stopper = EarlyStopping(
        patience=training_configuration['patience'])

    checkpointer = ModelCheckpoint(
        filepath=model_filename,
        verbose=0,
        save_best_only=True,
        save_weights_only=True)

    csv_logger = CSVLogger(csv_filename, separator=';')

    return [stopper, checkpointer, csv_logger]