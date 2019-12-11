from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.keras.layers import Activation, Reshape, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import ELU, ThresholdedReLU
from tensorflow.keras import backend as K
from termcolor import cprint

from models.learning_indicators import keras_losses
from models.set_optimizers import optimizer_setup
from models.layer_initializers import KernelInit, BiasInit
from models.layer_regularizers import regularizers
from utils.data_postprocessing import Visualizer
from utils.base_support import mkdirs, Timer

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
import sys

OSNAME = os.name

if OSNAME == 'nt':
    NEWLINECODE = '\r\n'
else:
    NEWLINECODE = '\n'

class SuperResolution:
    def __init__(self):
        pass
