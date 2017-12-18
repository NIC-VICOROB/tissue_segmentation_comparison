from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def generate_output_filename(
    path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension) :
    file_pattern = '{}/{}/{:02}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)

def generate_callbacks(general_configuration, training_configuration, case_name) :
    model_filename = generate_output_filename(
        general_configuration['model_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        training_configuration['dimension'],
        str(training_configuration['patch_shape']),
        str(training_configuration['extraction_step']),
        'h5')

    csv_filename = generate_output_filename(
        general_configuration['log_path'],
        training_configuration['dataset'],
        case_name,
        training_configuration['approach'],
        training_configuration['dimension'],
        str(training_configuration['patch_shape']),
        str(training_configuration['extraction_step']),
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