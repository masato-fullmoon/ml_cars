from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import model_from_json, model_from_yaml
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import concatenate, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
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

class UnetAE:
    def __init__(self, Xtrain, gpusave=False, summary=False, summaryout=False,
            optflag=True, optname='adam', activation='relu', alpha=0.3, theta=1.0,
            out_act='linear', input_filters=512, **kwargs):
        if gpusave:
            phys_devices = tf.config.experimental.list_physical_devices('GPU')

            if len(phys_devices) > 0:
                for d in range(len(phys_devices)):
                    tf.config.experimental.set_memory_growth(phys_devices[d], gpusave)
                    cprint('memory growth: {}'.format(
                        tf.config.experimental.get_memory_growth(phys_devices[d])),
                        'cyan', attrs=['bold'])

        self.Xtrain = Xtrain

        self.summary = summary
        self.summaryout = summaryout
        self.activation = activation
        self.alpha = alpha
        self.theta = theta
        self.out_act = out_act
        self.input_f = input_filters

        enc, unet = self.__sample_unet_forward()

        if optflag:
            opt = optimizer_setup(optname=optname, param_dict=kwargs)
            loss = keras_losses(losstype='mse')

            enc.compile(optimizer=opt, loss=loss)
            unet.compile(optimizer=opt, loss=loss)

        if summary:
            self.__unet_summary(enc, unet)

        self.enc, self.unet = enc, unet
        self.post = Visualizer()

    def get_autoencoder(self):
        return self.enc, self.unet

    #@Timer.timer
    def unet_train(self, cbs_tuple, Xval, epochs=1000, batch_size=50, verbose=0,
            savebasedir=None, caption=True, dpi=500, savetype=None,
            include_opt=False, fmt='.json', traintype='unet', **kwargs):
        try:
            savedir = mkdirs(os.path.join(
                savebasedir,'{}-callbacks'.format(self.unet.name))
                )
        except:
            if savebasedir is None:
                savebasedir = os.path.join(os.path.expanduser('~'),'NormalDNN')
            elif self.model.name is None:
                self.unet.name = 'Unknown_unet'
            else:
                raise Exception('Sorry, non-activated...')

            savedir = mkdirs(os.path.join(
                savebasedir,'{}-callbacks'.format(self.unet.name))
                )

        if traintype == 'unet':
            self.model = self.unet
        elif traintype == 'encoder':
            self.model = self.enc
        else:
            raise ValueError('train model: unet/encoder')

        if len(cbs_tuple) > 1:
            callbacks = self.__callbacks(savedir, cbs_tuple, kwargs)
        else:
            callbacks = None

        history = self.model.fit(
                x=self.Xtrain, y=self.Xtrain, epochs=epochs,
                batch_size=batch_size, verbose=verbose,
                validation_data=(Xval,Xval), callbacks=callbacks
                )

        if savetype is not None:
            self.__save_learning_params(
                    savetype=savetype, include_opt=include_opt,
                    savedir=savedir, fmt=fmt
                    )

        if (dpi is not None) and (type(dpi) is int):
            for indicator in [indicator for indicator in history.history.keys() \
                    if not 'val_' in indicator]:
                self.post.history_plot(
                        history=history, indicator=indicator,
                        dpi=dpi, caption=caption, savedir=savedir
                        )

    #@Timer.timer
    def generate_images(self, Xpred, npred, batch_size=None, verbose=0, savebasedir=None,
            dpi=500, resize_shape=(500,500), normtype=None):
        try:
            savedir = mkdirs(os.path.join(
                savebasedir,'{}-genimages'.format(self.model.name))
                )
        except:
            if savebasedir is None:
                savebasedir = os.path.join(os.path.expanduser('~'),'NormalDNN')
            elif self.model.name is None:
                self.model.name = 'Unknown_unet'
            else:
                raise Exception('Sorry, non-activated...')

            savedir = mkdirs(os.path.join(
                savebasedir,'{}-genimages'.format(self.unet.name))
                )

        prods = self.model.predict(Xpred, verbose=verbose, batch_size=batch_size)

        self.post.autoencoder_recodec(
                prods=prods, names=npred,
                savedir=savedir, dpi=dpi,
                resize_shape=resize_shape, normtype=normtype
                )

    def __sample_unet_forward(self):
        inputs = Input(shape=self.Xtrain.shape[1:])

        down1, p1 = self.__encode_layer(inputs, self.input_f)
        next_f = self.input_f*2
        down2, p2 = self.__encode_layer(p1, next_f)
        next_f *= 2
        down3, p3 = self.__encode_layer(p2, next_f)
        next_f *= 2
        down4, p4 = self.__encode_layer(p3, next_f)
        #next_f *= 2
        down5, p5 = self.__encode_layer(p4, next_f)
        #next_f *= 2
        down6 = self.__encode_layer(p5, next_f, False)
        #next_f = round(next_f/2)
        up7 = self.__decode_layer(down6,down5,next_f)
        #next_f = round(next_f/2)
        up8 = self.__decode_layer(up7,down4,next_f)
        next_f = round(next_f/2)
        up9 = self.__decode_layer(up8,down3,next_f)
        next_f = round(next_f/2)
        up10 = self.__decode_layer(up9,down2,next_f)
        next_f = round(next_f/2)
        up11 = self.__decode_layer(up10,down1,next_f)

        outputs = Conv2DTranspose(self.Xtrain.shape[-1],(3,3),
                strides=1,padding='same',activation=self.out_act)(up11)

        encoder = Model(inputs, down6, name='unet-encoder')
        unet = Model(inputs, outputs, name='unet-total')

        return encoder, unet

    def __encode_layer(self, inputs, num_f, maxpool=True):
        down = Conv2D(num_f,(3,3),strides=1,padding='same')(inputs)
        down = Activation('relu')(down)
        down = Conv2D(num_f,(3,3),strides=1,padding='same')(down)
        down = self.__custom_activations(down)
        pool = MaxPooling2D((1,1))(down)

        if maxpool:
            return down, pool
        else:
            return down

    def __decode_layer(self, input1, input2, num_f):
        up_merge = concatenate([UpSampling2D((1,1))(input1), input2],axis=-1)

        up = Conv2DTranspose(num_f,(3,3),strides=1,padding='same')(up_merge)
        up = self.__custom_activations(up)
        up = Conv2DTranspose(num_f,(3,3),strides=1,padding='same')(up)
        up = Activation('relu')(up)

        return up

    def __custom_activations(self, inputs):
        activations = (
                'relu','tanh','softplus',
                'softsign','selu','sigmoid',
                'hard_sigmoid','linear'
                )
        advanced_activations = (
                'leakyrelu','prelu',
                'elu','threshold'
                )

        if self.activation in activations:
            x = Activation(self.activation)(inputs)
        elif self.activation in advanced_activations:
            if self.activation == 'leakyrelu':
                assert 0. < self.alpha < 1., 'alpha: from 0 to 1'

                x = LeakyReLU(self.alpha)(inputs)
            elif self.activation == 'prelu':
                x = PReLU()(inputs)
            elif self.activation == 'elu':
                x = ELU(self.alpha)(inputs)
            elif self.activation == 'threshold':
                assert self.theta >= 0., 'theta: more than 0'

                x = ThresholdedReLU(self.theta)
        else:
            raise ValueError('activation: Incorrect.')

        return x

    def __unet_summary(self, encoder, unet):
        if self.summaryout:
            try:
                with open('./{}_summary.log'.format(unet.name), 'w') as s:
                    encoder.summary(print_ln=lambda x:s.write(x+NEWLINECODE))
                    unet.summary(print_ln=lambda x:s.write(x+NEWLINECODE))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('model.name: None', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            encoder.summary()
            unet.summary()

    def __callbacks(self, savedir, cbs_tuple, cb_param_dict):
        cbs = list()
        for name in cbs_tuple:
            if 'csvlogger' in cbs_tuple:

                try:
                    csvfilepath = os.path.join(
                            savedir,'{}_learninglog.csv'.format(self.model.name)
                            )
                except:
                    cprint('model.name: None, auto-setup.', 'yellow', attrs=['bold'])
                    csvfilepath = os.path.join(savedir,'unknown_learninglog.csv')
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
                    min_delta = cb_param_dict['min_delta']
                    patience = cb_param_dict['patience']
                    verbose = cb_param_dict['verbose']
                    mode = cb_param_dict['mode']
                except Exception as err:
                    cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                    sys.exit(1)

                es_cb = EarlyStopping(
                        monitor=monitor, patience=patience,
                        verbose=verbose, mode=mode, min_delta=min_delta
                        )
                cbs.append(es_cb)

        return cbs

    def __save_learning_params(self, savetype, include_opt, savedir, **kwargs):
        if savetype == 'total':
            savepath = os.path.join(savedir,'{}.{}.h5'.format(self.model.name,savetype))
            self.model.save(savepath, include_optimizer=include_opt)
        elif savetype == 'weight':
            savepath = os.path.join(savedir,'{}.{}.hdf5'.format(self.model.name,savetype))
            self.model.save_weights(savepath)
        elif savetype == 'model':
            for k in kwargs.keys():

                if k == '.yaml':
                    savepath = os.path.join(
                            savedir,'{}.{}{}'.format(
                                self.model.name,savetype,k)
                            )
                    save_str = self.model.to_yaml()
                else:
                    savepath = os.path.join(
                            savedir,'{}.{}.json'.format(
                                self.model.name,savetype)
                            )
                    save_str = self.model.to_json()

                with open(savepath, 'w') as sm:
                    sm.write(save_str+NEWLINECODE)
        else:
            raise ValueError('savetype: total/weight/model')

class VAE:
    def __init__(self):
        pass
