from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import ThresholdedReLU
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from keras import backend as K
from utils.base_support import mkdirs, Timer
from models.set_optimizers import optimizer_setup
from models.learning_indicators import LearningIndicators
from models.learning_indicators import keras_losses
from termcolor import cprint
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys

class NormalDNN:
    def __init__(self, Xtrain, ytrain, gpusave=False, summary=False, summaryout=False,
            modelname='sample1', optname='adam', fc_nodes=1024, fc_act='relu',
            alpha=0.3, theta=1.0, optflag=True, **kwargs):
        if gpusave:
            cfg = tf.ConfigProto(allow_soft_placement=True)
            cfg.gpu_options.allow_growth = True
            K.set_session(tf.Session(config=cfg))

        self.Xtrain = Xtrain
        self.ytrain = ytrain

        self.nodes = fc_nodes
        self.activation = fc_act
        self.alpha = alpha
        self.theta = theta

        if modelname == 'sample1':
            self.model = self.__sample_forward_1()
        elif modelname == 'sample2':
            self.model = self.__sample_forward_2()
        else:
            sys.exit(1)

        if optflag:
            opt = optimizer_setup(optname=optname, param_dict=kwargs)
            loss = keras_losses(losstype='multi_classification')

            indic_obj = LearningIndicators()
            met = indic_obj.classify_metrics(num_classes=ytrain.shape[-1])

            self.model.compile(optimizer=opt, loss=loss, metrics=met)

        if summary:
            self.__summary(summaryout=summaryout)

    def get_model(self):
        return self.model

    def load_architectures(self, optflag=False):
        pass

    def model_train(self, Xval, yval, epochs=100, batch_size=20, verbose=1,
            savedir=None, use_cb=False, *args, **kwargs):
        if (savedir is not None) and (type(savedir) is str):
            savedir = mkdirs(savedir)

            if use_cb:
                pass
            else:
                callbacks = None

        history = self.model.fit(
                self.Xtrain, self.ytrain, epochs=epochs,
                batch_size=batch_size, verbose=verbose,
                validation_data=(Xval,yval), callbacks=callbacks
                )

    def __sample_forward_1(self):
        inputs = Input(shape=self.Xtrain.shape[1:])

        x = Conv2D(16,(5,5),padding='same',activation='relu')(inputs)
        x = Conv2D(32,(5,5),padding='same',activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.3)(x)
        x = Conv2D(64,(5,5),padding='same',activation='relu')(x)
        x = Conv2D(64,(5,5),padding='same',activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)

        fc = Flatten()(x)
        fc = self.__fc_layer_activation_custom(
                inputs=fc, nodes=self.nodes, fc_act=self.activation,
                alpha=self.alpha, theta=self.theta
                )
        fc = Dropout(0.1)(fc)

        outputs = Dense(self.ytrain.shape[-1],activation='softmax')(fc)

        model = Model(inputs, outputs, name='normal-sample-1')

        return model

    def __sample_forward_2(self):
        inputs = Input(shape=self.Xtrain.shape[1:])

        x = Conv2D(32,(3,3),padding='same',activation='relu')(inputs)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
        x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)

        fc = Flatten()(x)
        fc = self.__fc_layer_activation_custom(
                inputs=fc, nodes=self.nodes, fc_act=self.activation,
                alpha=self.alpha, theta=self.theta
                )
        fc = Dropout(0.5)(fc)

        outputs = Dense(self.ytrain.shape[-1], activation='softmax')(fc)

        model = Model(inputs, outputs, name='normal-sample-2')

        return model

    def __fc_layer_activation_custom(self, inputs, nodes, fc_act, alpha, theta):
        activations = (
                'relu','tanh','softplus',
                'softsign','selu','sigmoid',
                'hard_sigmoid','linear'
                )
        advanced_activations = (
                'leakyrelu','prelu','elu','threshold'
                )

        if fc_act in activations:
            x = Dense(nodes,activation=fc_act)(inputs)
        elif fc_act in advanced_activations:
            x = Dense(nodes)(inputs)

            if fc_act == 'leakyrelu':
                assert 0<alpha<1, 'LeakyReLU-alpha param: Incorrect.'
                x = LeakyReLU(alpha)(x)
            elif fc_act == 'prelu':
                x = PReLU()(x)
            elif fc_act == 'elu':
                x = ELU(alpha)(x)
            elif fc_act == 'threshold':
                assert theta>=0, 'ThresholdedReLU-theta param: Incorrect.'
                x = ThresholdedReLU(theta)(x)
        else:
            raise ValueError('Custom Activation Functions are incorrect.')

        return x

    def __summary(self, summaryout):
        if summaryout:
            with open('./{}_summary.txt'.format(self.model.name), 'w') as s:
                self.model.summary(print_ln=lambda x:s.write(x+NEWLINECODE))
        else:
            self.model.summary()

    def __callbacks(self, savedir, cbs_tuple, cb_param_dict):
        cbs = list()
        for name in cbs_tuple:
            if 'csvlogger' in cbs_tuple:

                try:
                    csvfilepath = os.path.join(savedir,'{}.csv'.format(self.model.name))
                except:
                    cprint('Model name is None.')
                    csvfilepath = os.path.join(savedir, 'unknownmodel.csv')
                finally:
                    csv_cb = CSVLogger(csvfilepath)
                    cbs.append(csv_cb)

            elif 'tensorboard' in cbs_tuple:

                try:
                    log_dir = os.path.join(savedir, 'tblogs')
                    histogram_freq = cb_param_dict['histogram_freq']
                    write_graph = cb_param_dict['write_graph']
                except Exception as err:
                    cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                    sys.exit(1)

                tb_cb = TensorBoard(
                        log_dir=log_dir,
                        histogram_freq=histogram_freq,
                        write_graph=write_graph
                        )
                cbs.append(tb_cb)
            elif 'earlystopping' in cbs_tuple:

                try:
                    monitor = cb_param_dict['monitor']
                    patience = cb_param_dict['patience']
                    verbose = cb_param_dict['verbose']
                    mode = cb_param_dict['mode']
                except Exception as err:
                    cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                    sys.exit(1)

                es_cb = EarlyStopping(
                        monitor=monitor, patience=patience,
                        verbose=verbose, mode=mode
                        )
                cbs.append(es_cb)
