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
from models.model_initializers import KernelInit, BiasInit
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

class GANmodel:
    def __init__(self, imgtensors, gpusave=False, summary=False, summaryout=False,
            auto_zdim=False, zdim=100, optname='adam', initflag=True, k_initname='he_normal',
            b_initname='zeros', **kwargs):
        if gpusave:
            phys_devices = tf.config.experimental.list_physical_devices('GPU')

            if len(phys_devices) > 0:
                for d in range(len(phys_devices)):
                    tf.config.experimental.set_memory_growth(phys_devices[d], gpusave)
                    cprint('memory growth: {}'.format(
                        tf.config.experimental.get_memory_growth(phys_devices[d])),
                        'cyan', attrs=['bold'])

        self.tensor = imgtensors

        self.summary = summary
        self.summaryout = summaryout

        if initflag:
            self.k_init = KernelInit().kernel_initializer(initname=k_initname, kwargs)
            self.b_init = BiasInit().bias_initializer(initname=b_initname, kwargs)

        if auto_zdim:
            try:
                self.zdim = int(np.prod(np.array(self.tensor.shape[1:])))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('Image tensor shape: not-4D.', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            self.zdim = zdim

        self.generator, self.discriminator,\
                self.model = self.__sample1_dcgan_forward(optname, kwargs)

        self.post = Visualizer()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def get_gan_model(self):
        return self.model

    def dcgan_train(self, epochs=2000, batch_size=200, num_div=2,
            save_iter=None, debug=False, breakpoint=1, gentype='tiled_generate',
            tile_h=10, tile_w=10, savedir=None, dpi=500):
        assert gentype in ('tiled_generate', 'each_save_generate'),\
                'gentype: tiled_generate/each_save_generate'

        samp_batch = batch_size//num_div
        savedir = mkdirs(os.path.join(savedir, gentype))

        loss_dict = {
                'd-real-loss':[],
                'd-fake-loss':[],
                'd-ave-loss':[],
                'g-ave-loss':[]
                }

        for e in range(1,epochs+1):
            generator_noise = np.random.normal(0,1,(samp_batch,self.zdim))
            gan_noise = np.random.normal(0,1,(batch_size,self.zdim))
            true_vector = np.array([1]*batch_size)

            fake_imgs = self.generator.predict(generator_noise)
            real_imgs = self.tensor[
                    np.random.randint(0,self.tensor.shape[0],samp_batch)
                    ]

            dloss_fake = self.discriminator.train_on_batch(
                    fake_imgs, np.zeros((samp_batch,1))
                    )
            dloss_real = self.discriminator.train_on_batch(
                    real_imgs, np.ones((samp_batch,1))
                    )
            dloss_ave = 0.5*np.add(dloss_fake, dloss_real)
            gloss_ave = self.model.train_on_batch(gan_noise, true_vector)

            loss_dict['d-fake-loss'].append(dloss_fake)
            loss_dict['d-real-loss'].append(dloss_real)
            loss_dict['d-ave-loss'].append(dloss_ave)
            loss_dict['g-ave-loss'].append(gloss_ave)

            if (save_iter is not None) and (type(save_iter) is int):
                if (e%save_iter == 0) and (gentype == 'tiled_generate'):
                    self.post.save_tiled_generate(
                            epoch=e, generator=self.generator,
                            zdim=self.zdim, tile_h=tile_h, tile_w=tile_w,
                            savedir=savedir, dpi=dpi
                            )
                elif (e%save_iter == 0) and (gentype == 'each_save_generate'):
                    self.post.each_save(
                            fake_img=fake_imgs[random.choice(range(samp_batch))],
                            epoch=e, tile_h=tile_h, tile_w=tile_w,
                            savedir=savedir, dpi=dpi
                            )

            if (debug) and (e == breakpoint):
                break

        return loss_dict

    #@Timer.timer
    def property_visualization(self, loss_dict=None, gentype='tiled_generate', savedir=None, dpi=500):
        df = pd.DataFrame(loss_dict)
        df.index = range(1, df.shape[0]+1)

        for c in df.columns:
            pass

    def __sample1_g_forward(self):
        if self.k_init and self.b_init:
            kernel_initializer = self.k_init
            bias_initializer = self.b_init
        else:
            kernel_initializer = 'glorot_uniform'
            bias_initializer = 'zeros'

        reshape_unit = self.tensor.shape[1]//4

        inputs = Input(shape=(self.zdim,))

        x = Dense(1024,kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(128*reshape_unit*reshape_unit,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        reshape = Reshape((reshape_unit,reshape_unit,128),
                input_shape=(128*reshape_unit*reshape_unit,))(x)

        x = UpSampling2D((2,2))(reshape)
        x = Conv2D(64,(5,5),strides=(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2,2))(x)

        outputs = Conv2D(self.tensor.shape[-1],(5,5),strides=(1,1),
                padding='same',activation='tanh')(x)

        return Model(inputs, outputs, name='sample1-generator')

    def __sample1_d_forward(self):
        if self.k_init and self.b_init:
            kernel_initializer = self.k_init
            bias_initializer = self.b_init
        else:
            kernel_initializer = 'glorot_uniform'
            bias_initializer = 'zeros'

        inputs = Input(shape=self.tensor.shape[1:])

        x = Conv2D(64,(5,5),strides=(2,2),padding='same')(inputs)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128,(5,5),strides=(2,2))(x)
        x = LeakyReLU(0.2)(x)

        fc = Flatten()(x)
        fc = Dense(256,kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer)(fc)
        fc = LeakyReLU(0.2)(fc)
        fc = Dropout(0.5)(fc)

        outputs = Dense(1,kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,activation='sigmoid')(fc)

        return Model(inputs, outputs, name='sample1-discriminator')

    def __sample1_dcgan_forward(self, optname, optparams):
        generator = self.__sample1_g_forward()
        discriminator = self.__sample1_d_forward()

        loss = keras_losses(losstype='binary_classification')
        opt = optimizer_setup(optname, optparams)

        discriminator.compile(loss=loss, optimizer=opt)
        discriminator.trainable = False

        inputs = Input(shape=(self.zdim,))
        outputs = discriminator(generator(inputs))

        model = Model(inputs, outputs, name='sample1-DCGAN')
        model.compile(loss=loss, optimizer=opt)

        if self.summary:
            self.__summary(g=generator, d=discriminator, t=model)

        return generator, discriminator, model

    def __summary(self, **kwargs):
        if self.summaryout:
            try:
                with open('{}_summary.log'.format(model.name), 'w') as s:
                    for k in kwargs.keys():
                        kwargs[k].summary(print_ln=lambda x:s.write(x+NEWLINECODE))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('model.name is None, or not model-object', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            for k in kwargs.keys():
                kwargs[k].summary()
