import numpy as np
import nibabel as nib

from sklearn.model_selection import LeaveOneOut

from keras import backend as K
from keras.utils import np_utils

from configuration import general_configuration, training_configuration
from utils.callbacks import generate_callbacks

# Leave-one-out cross-validation on the training set
loo = LeaveOneOut()
for train_index, test_index in loo.split(range(nvols)):
    print train_index, test_index

    x_train, y_train = build_training_set()

    callbacks = generate_callbacks(test_index[0] + 1)

    ### Train the model
    model.fit(
        x_train, y_train,
        epochs=training_configuration['num_epochs'],
        validation_split=training_configuration['validation_split'],
        verbose=training_configuration['verbose'],
        callbacks=[checkpointer, csv_logger, stopper])

    del x_train
    del y_train

    model = model_creator.generate_model()
    model.load_weights(model_filename)

    del x_test