from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.advanced_activations import ELU, ThreshholdedReLU
from keras import backend as K
from termcolor import cprint
import tensorflow as tf
import numpy as np
import random
import os
import sys

class GANmodel:
    def __init__(self, imgtensors, imgnames, gpusave=False, summary=False,
            summaryout=False, auto_zdim=False, zdim=100, **kwargs):
        if gpusave:
            cfg = tf.ConfigProto(allow_soft_placement=True)
            cfg.gpu_options.allow_growth = True
            K.set_session(tf.Session(config=cfg))

        self.tensor = imgtensors
        self.names = imgnames

        if auto_zdim:
            try:
                self.zdim = int(np.prod(np.array(self.tensor.shape[1:])))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('Image tensor shape: not-4D.', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            self.zdim = zdim

    def __sample1_g_forward(self):
        reshape_unit = self.tensor.shape[0]//4

        inputs = Input(shape=(self.zdim,))

        x = Dense(1024)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(128*reshape_unit**2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        reshape = Reshape((reshape_unit,reshape_unit,128),
                input_shape=(128*reshape_unit**2,))(x)

        x = UpSampling2D((2,2))(reshape)
        x = Conv2D(64,(5,5),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2,2))(reshape)

        outputs = Conv2D(self.tensor.shape[-1],(5,5),padding='same',activation='tanh')(x)

        return Model(inputs, outputs, name='sample1-generator')

    def __sample1_d_forward(self):
        inputs = Input(shape=self.tensor.shape[1:])

        x = Conv2D(64,(5,5),strides=(2,2),padding='same')(inputs)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128,(5,5),strides=(2,2))(x)
        x = LeakyReLU(0.2)(x)

        fc = Flatten()(x)
        fc = Dense(256)(fc)
        fc = LeakyReLU(0.2)(fc)
        fc = Dropout(0.5)(fc)

        outputs = Dense(1,activation='sigmoid')(fc)

        return Model(inputs, outputs, name='sample1-discriminator')

    def __sample1_dcgan_forward(self, optparams=None):
        generator = self.__sample1_g_forward()
        discriminator = self.__sample1_d_forward()
