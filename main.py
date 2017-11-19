import numpy as np
import nibabel as nib



from keras import backend as K
from keras.utils import np_utils

from configuration import general_configuration, training_configuration
from utils.callbacks import generate_callbacks

