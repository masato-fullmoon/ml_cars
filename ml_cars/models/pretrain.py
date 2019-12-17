from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import model_from_json, model_from_yaml
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import ThresholdedReLU
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from termcolor import cprint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# ----- original modules -----
from utils.base_support import mkdirs, Timer
from utils.data_postprocessing import Visualizer
from models.set_optimizers import optimizer_setup
from models.learning_indicators import keras_losses
from models.learning_indicators import LearningIndicators
# ----- original modules end -----

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys

if os.name == 'nt':
    NEWLINECODE = '\r\n'
else:
    NEWLINECODE = '\n'

class __BaseModels:
    def __init__(self):
        '''
        現在は二つのサンプルCNNモデル, VGG16, VGG19を実装済み

        '''
        pass

    def sample_forward_1(self, Xshape, yshape, nodes=1024,
            activation='relu', alpha=0.3, theta=1.0):
        with tf.device('/cpu:0'):
            inputs = Input(shape=Xshape[1:])

            x = Conv2D(16,(5,5),strides=(2,2),padding='same')(inputs)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(32,(5,5),strides=(2,2),padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2,2))(x)
            x = Conv2D(64,(5,5),strides=(2,2),padding='same')(x)
            x = Activation('relu')(x)
            x = Conv2D(64,(5,5),strides=(2,2),padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2,2))(x)
            x = Dropout(0.25)(x)

            fc = Flatten()(x)
            fc = self.__fc_layer_activation_custom(
                    fc, nodes, activation,
                    alpha, theta
                    )
            fc = Dropout(0.1)(fc)

            outputs = Dense(yshape[-1],activation='softmax')(fc)

        ef_model = Model(inputs, x, name='sample1-extract-layers')
        pretrainmodel = Model(inputs, outputs, name='sample1-pretrain')

        return ef_model, pretrainmodel

    def sample_forward_2(self, Xshape, yshape, nodes=1024,
            activation='relu', alpha=0.3, theta=1.0):
        with tf.device('/cpu:0'):
            inputs = Input(shape=Xshape[1:])

            x = Conv2D(32,(3,3),padding='same',activation='relu')(inputs)
            x = MaxPooling2D((2,2))(x)
            x = Dropout(0.25)(x)
            x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
            x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
            x = MaxPooling2D((2,2))(x)
            x = Dropout(0.25)(x)

            fc = Flatten()(x)
            fc = self.__fc_layer_activation_custom(
                    fc, nodes, activation,
                    alpha, theta
                    )
            fc = Dropout(0.5)(fc)

            outputs = Dense(yshape[-1], activation='softmax')(fc)

        ef_model = Model(inputs, x, name='sample2-extract-layers')
        pretrainmodel = Model(inputs, outputs, name='sample2-pretrain')

        return ef_model, pretrainmodel

    def vgg16_forward(self, Xshape, yshape, custom=True,
            nodes=1024, activation='relu', alpha=0.3, theta=1.0):
        if custom:
            with tf.device('/cpu:0'):
                vgg16 = VGG16(weights=None, include_top=False,
                        input_shape=Xshape[1:], pooling=None)

                inputs = Input(shape=vgg16.output_shape[1:])

                fc = Flatten()(inputs)
                fc = self.__fc_layer_activation_custom(
                        fc, nodes, activation,
                        alpha, theta
                        )
                fc = BatchNormalization()(fc)

                outputs = Dense(yshape[-1],activation='softmax')(fc)

                top_model = Model(inputs,outputs,name='VGG16-FC')
                total_model = Model(vgg16.inputs,top_model(vgg16.outputs),
                        name='VGG16-custom')

            return top_model, total_model
        else:
            with tf.device('/cpu:0'):
                vgg16 = VGG16(weights=None,include_top=True,
                        input_shape=Xshape[1:],classes=yshape[-1])

            return None, vgg16

    def vgg19_forward(self, Xshape, yshape, custom=True,
            nodes=1024, activation='relu', alpha=0.3, theta=1.0):
        if custom:
            with tf.device('/cpu:0'):
                vgg19 = VGG19(weights=None,include_top=False,
                        input_shape=Xshape[1:],pooling=None)

                inputs = Input(shape=vgg19.output_shape[1:])

                fc = Flatten()(inputs)
                fc = self.__fc_layer_activation_custom(
                        fc, nodes, activation,
                        alpha, theta
                        )
                fc = BatchNormalization()(fc)

                outputs = Dense(yshape[-1],activation='softmax')(fc)

                top_model = Model(inputs,outputs,name='VGG19-FC')
                total_model = Model(vgg19.inputs,top_model(vgg19.outputs),
                        name='VGG19-custom')

            return top_model, total_model
        else:
            with tf.device('/cpu:0'):
                vgg19 = VGG19(weights=None,include_top=True,
                        input_shape=Xshape[1:],classes=yshape[-1])

            return None, vgg19

    def __fc_layer_activation_custom(self, inputs, nodes, activation, alpha, theta):
        activations = (
                'relu','tanh','softplus',
                'softsign','selu','sigmoid',
                'hard_sigmoid','linear'
                )
        advanced_activations = (
                'leakyrelu','prelu',
                'elu','threshold'
                )

        if activation in activations:
            x = Dense(nodes)(inputs)
            x = Activation(activation)(x)
        elif activation in advanced_activations:
            x = Dense(nodes)(inputs)

            if activation == 'leakyrelu':
                assert 0<alpha<1, 'LeakyReLU-alpha param: Incorrect.'

                x = LeakyReLU(alpha)(x)
            elif activation == 'prelu':
                x = PReLU()(x)
            elif activation == 'elu':
                x = ELU(alpha)(x)
            elif activation == 'threshold':
                assert theta>=0, 'ThresholdedReLU-theta param: Incorrect.'

                x = ThresholdedReLU(theta)(x)
        else:
            raise ValueError('Custom Activation Functions are incorrect.')

        return x

class Pretrain(__BaseModels):
    def __init__(self, Xtrain, ytrain, gpusave=False, summary=False, summaryout=False,
            modelname='sample1', optname='adam', fc_nodes=1024, fc_act='relu',
            alpha=0.3, theta=1.0, optflag=True, custom=True, **kwargs):
        super().__init__()

        if gpusave:
            phys_devices = tf.config.experimental.list_physical_devices('GPU')

            if len(phys_devices) > 0:
                for d in range(len(phys_devices)):
                    tf.config.experimental.set_memory_growth(phys_devices[d], gpusave)
                    cprint('memory growth: {}'.format(
                        tf.config.experimental.get_memory_growth(phys_devices[d])),
                        'cyan', attrs=['bold'])

        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.post = Visualizer()
