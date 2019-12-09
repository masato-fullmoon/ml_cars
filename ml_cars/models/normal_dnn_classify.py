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

        return None, Model(inputs, outputs, name='normal-sample-1')

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

        return None, Model(inputs, outputs, name='normal-sample-2')

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

class NormalDNN(__BaseModels):
    def __init__(self, Xtrain, ytrain, gpusave=False, summary=False, summaryout=False,
            modelname='sample1', optname='adam', fc_nodes=1024, fc_act='relu',
            alpha=0.3, theta=1.0, optflag=True, custom=True, **kwargs):
        super().__init__()

        cfg = tf.ConfigProto(allow_soft_placement=gpusave)
        cfg.gpu_options.allow_growth = gpusave
        K.set_session(tf.Session(config=cfg))

        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.post = Visualizer()

        if modelname == 'sample1':
            self.top, self.model = self.sample_forward_1(
                    Xshape=Xtrain.shape, yshape=ytrain.shape,
                    nodes=fc_nodes, activation=fc_act,
                    alpha=alpha, theta=theta
                    )
        elif modelname == 'sample2':
            self.top, self.model = self.sample_forward_2(
                    Xshape=Xtrain.shape, yshape=ytrain.shape,
                    nodes=fc_nodes, activation=fc_act,
                    alpha=alpha, theta=theta
                    )
        elif modelname == 'vgg16':
            self.top, self.model = self.vgg16_forward(
                    Xshape=Xtrain.shape, yshape=ytrain.shape,
                    custom=custom, nodes=fc_nodes, activation=fc_act,
                    alpha=alpha, theta=theta
                    )
        elif modelname == 'vgg19':
            self.top, self.model = self.vgg19_forward(
                    Xshape=Xtrain.shape, yshape=ytrain.shape,
                    custom=custom, nodes=fc_nodes, activation=fc_act,
                    alpha=alpha, theta=theta
                    )
        elif modelname == 'load':
            self.top, self.model = self.__load_model_arch()
        else:
            raise ValueError('Sorry, test now.')

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

    def get_top_layers(self):
        if self.top is not None:
            return self.top
        else:
            cprint('Top model is None.', 'yellow', attrs=['bold'])

    #@Timer.timer
    def model_train(self, cbs_tuple, Xval, yval, epochs=100, batch_size=20, verbose=1,
            savebasedir=None, dpi=None, caption=True, savetype=None,
            include_opt=False, fmt='.json', **kwargs):
        try:
            savedir = mkdirs(os.path.join(
                savebasedir,'{}-learning'.format(self.model.name))
                )
        except:
            if savebasedir is None:
                savebasedir = os.path.join(os.path.expanduser('~'),'NormalDNN')
            elif self.model.name is None:
                self.model.name = 'Unknown_DNN_classification'
            else:
                raise Exception('Sorry, non-activated...')

            savedir = mkdirs(os.path.join(
                savebasedir,'{}-learning'.format(self.model.name))
                )

        if len(cbs_tuple) > 1:
            callbacks = self.__callbacks(savedir, cbs_tuple, kwargs)
        else:
            callbacks = None

        history = self.model.fit(
                self.Xtrain, self.ytrain, epochs=epochs,
                batch_size=batch_size, verbose=verbose,
                validation_data=(Xval,yval), callbacks=callbacks
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
    def prediction(self, class_dict, Xpred, ypred, npred, batch_size=None, verbose=0,
            savebasedir=None, dpi=None, caption=True, predlogext=None, type_='confusion_matrix'):
        try:
            savedir = os.path.join(savebasedir,'{}'.format(self.model.name))
            savedir = mkdirs(savedir)
        except:
            if savebasedir is None:
                savebasedir = os.path.join(os.path.expanduser('~'),'NormalDNN')
            elif self.model.name is None:
                self.model.name = 'Unknown_DNN_classification'
            else:
                raise Exception('Sorry, non-activated...')

            savedir = mkdirs(os.path.join(savebasedir,self.model.name))

        probs = self.model.predict(
                Xpred, batch_size=batch_size, verbose=verbose
                )

        trues = [np.argmax(ypred[i]) for i in range(ypred.shape[0])]
        preds = [np.argmax(probs[i]) for i in range(probs.shape[0])]
        match = [0 for true, pred in zip(trues, preds) if true == pred]

        cprint('Prediction-Matching Rate: {:.3f} %'.format(len(match)/len(trues)*100),
                'cyan', attrs=['bold'])

        df = pd.DataFrame({
            'name':npred,
            'true':trues,
            'pred':preds
            })

        for i in range(ypred.shape[-1]):
            df['p-{}'.format(i)] = [probs[j,i] for j in range(probs.shape[0])]

        df['results'] = [class_dict[p] for p in preds]

        df = df.sort_values('name')
        df.index = range(1, probs.shape[0]+1)
        df.dtype = float

        if predlogext is not None:
            if predlogext == '.csv':
                df.to_csv(os.path.join(
                    savedir,'{}{}'.format(self.model.name, predlogext))
                    )
            elif predlogext == '.xlsx':
                try:
                    df.to_excel(os.path.join(
                        savedir,'{}{}'.format(self.model.name, predlogext))
                        )
                except Exception as err:
                    cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                    sys.exit(1)
            elif predlogext == '.html':
                df.to_html(os.path.join(
                    savedir,'{}{}'.format(self.model.name, predlogext))
                    )
            else:
                raise ValueError('predlogext: .csv/.xlsx/.html')

        if dpi is not None:
            if type_ == 'probability_frequency_histogram':
                self.post.predict_max_histogram(
                        probs=probs, dpi=dpi,
                        caption=caption, savedir=savedir
                        )
            elif type_ == 'confusion_matrix':
                self.post.class_confusion_matrix(
                        ytrues=trues, ypreds=preds,
                        dpi=dpi, annot=caption,
                        savedir=savedir, class_dict=class_dict
                        )
            else:
                raise ValueError('Sorry, test now.')

    def __load_modelarch(self, loadpath, optflag=False):
        _, e = os.path.splitext(loadpath)

        if e == '.json':
            with open(loadpath, 'r') as a:
                loadmodel = model_from_json(a)
        elif e == '.yaml':
            with open(loadpath, 'r') as a:
                loadmodel = model_from_yaml(a)
        elif e == '.h5':
            loadmodel = load_model(loadpath, compile=optflag)
        else:
            raise ValueError('load file extension: .json/.yaml/.h5/.hdf5')

        return None, loadmodel

    def __load_modelweight(self, weightpath, model=None, by_name=True):
        _, e = os.path.splitext(loadpath)

        if (e == '.h5') or (e == '.hdf5'):
            assert model is not None, 'Input loaded model architecture.'

            weighted_model = model
            weighted_model.load_weights(weightpath, by_name=by_name)

            return weighted_model
        else:
            raise FileNotFoundError('model weight extension: .h5/.hdf5')

    def __summary(self, summaryout):
        if summaryout:

            try:
                with open('./{}_summary.log'.format(self.model.name), 'w') as s:

                    if self.top is not None:
                        self.top.summary(print_ln=lambda x:s.write(x+NEWLINECODE))

                    self.model.summary(print_ln=lambda x:s.write(x+NEWLINECODE))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('model name is None.', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            if self.top is not None:
                self.top.summary()

            self.model.summary()

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

class NormalDNNCrossValidation(__BaseModels):
    def __init__(self, X, y, names, gpusave=False, summary=False, summaryout=False,
            modelname='sample1', optname='adam', fc_nodes=1024, fc_act='relu',
            alpha=0.3, theta=1.0, optflag=True, custom=True, splitrate=0.2):
        super().__init__()

        cfg = tf.ConfigProto(allow_soft_placement=gpusave)
        cfg.gpu_options.allow_growth = gpusave
        K.set_session(tf.Session(config=cfg))

        self.learn, self.pred = self.__data_split(X, y, names, splitrate)

        self.summary = summary
        self.summaryout = summaryout
        self.modelname = modelname
        self.optname = optname
        self.optflag = optflag

        self.nodes = fc_nodes
        self.activation = fc_act
        self.alpha = alpha
        self.theta = theta
        self.custom = custom

        self.post = Visualizer()

    def get_cvdata(self):
        return self.learn, self.pred

    #@Timer.timer
    def cv_train(self, cbs_tuple, epochs=100, batch_size=32, verbose=0, splitrate=0.2, K=5,
            savebasedir=None, dpi=None, caption=None, savetype=None,
            include_opt=False, fmt='.json', **kwargs):
        try:
            bd = os.path.join(savebasedir,'crossvalidation-learning')
        except:
            cprint(
                    'savebasedir is None, auto-setup savebasedir: [/home/user/NormalDNN]',
                    'yellow', attrs=['bold']
                    )

            savebasedir = os.path.join(os.path.expanduser('~'),'NormalDNN')
            bd = os.path.join(savebasedir,'crossvalidation-learning')

        Xlearn, ylearn = self.learn[0], self.learn[1]

        cnt = 1
        model_dict = {}

        # Kの値だけループする
        for train, val in KFold(K, shuffle=True).split(Xlearn,ylearn):
            Xtrain, ytrain = Xlearn[train], ylearn[train]
            Xval, yval = Xlearn[val], ylearn[val]

            model = self.__forward_optimize(
                    Xshape=Xlearn.shape, yshape=ylearn.shape,
                    param_dict=kwargs
                    )

            savedir = mkdirs(
                    os.path.join(bd,'{}-K-{}'.format(model.name, cnt))
                    )

            if len(cbs_tuple)>1:
                callbacks = self.__callbacks(model, savedir, cbs_tuple, kwargs)
            else:
                callbacks = None

            history = model.fit(
                    Xtrain, ytrain, epochs=epochs,
                    batch_size=batch_size, verbose=verbose,
                    validation_data=(Xval,yval), callbacks=callbacks
                    )

            results = model.evaluate(
                    Xval, yval, batch_size=batch_size, verbose=verbose
                    )

            cprint('***** loop-{} *****'.format(cnt), 'cyan', attrs=['bold'])
            cprint('results: {}'.format(results), 'cyan', attrs=['bold'])

            model_dict[str(cnt)] = model

            if savetype is not None:
                self.__save_learning_params(
                        model=model, savetype=savetype,
                        include_opt=include_opt, savedir=savedir, fmt=fmt
                        )

            if (dpi is not None) and (type(dpi) is int):
                for indicator in [indicator for indicator in history.history.keys() \
                        if not 'val_' in indicator]:
                    self.post.history_plot(
                            history=history, indicator=indicator,
                            dpi=dpi, caption=caption, savedir=savedir
                            )

            cnt += 1

        return model_dict

    #@Timer.timer
    def cv_predict(self, class_dict, model_dict, batch_size=None, verbose=0,
            savebasedir=None, dpi=None, caption=True, predlogext=None, type_='confusion_matrix'):
        try:
            bd = os.path.join(savebasedir,'crossvalidation-learning')
        except:
            cprint(
                    'savebasedir is None, auto-setup savebasedir: [/home/user/NormalDNN]',
                    'yellow', attrs=['bold']
                    )

            savebasedir = os.path.join(os.path.expanduser('~'),'NormalDNN')
            bd = os.path.join(savebasedir,'crossvalidation-learning')

        Xpred, ypred, npred = self.pred[0], self.pred[1], self.pred[2]

        for k in model_dict.keys():
            model = model_dict[k]

            probs = model.predict(
                    Xpred, batch_size=batch_size, verbose=verbose
                    )

            trues = [np.argmax(ypred[i]) for i in range(ypred.shape[0])]
            preds = [np.argmax(probs[i]) for i in range(probs.shape[0])]
            match = [0 for true, pred in zip(trues, preds) if true == pred]

            cprint('***** loop-{} *****'.format(int(k)), 'cyan', attrs=['bold'])
            cprint('Prediction-Matching Rate: {:.3f} %'.format(len(match)/len(trues)*100),
                    'cyan', attrs=['bold'])

            df = pd.DataFrame({
                'name':npred,
                'true':trues,
                'pred':preds
                })

            for i in range(ypred.shape[-1]):
                df['p-{}'.format(i)] = [probs[j,i] for j in range(probs.shape[0])]

            df['results'] = [class_dict[p] for p in preds]

            df = df.sort_values('name')
            df.index = range(1, probs.shape[0]+1)
            df.dtype = float

            savedir = mkdirs(
                    os.path.join(bd,'{}-K-{}'.format(model.name, k))
                    )

            if predlogext is not None:
                if predlogext == '.csv':
                    df.to_csv(os.path.join(
                        savedir,'{}{}'.format(model.name, predlogext))
                        )
                elif predlogext == '.xlsx':
                    try:
                        df.to_excel(os.path.join(
                            savedir,'{}{}'.format(model.name, predlogext))
                            )
                    except Exception as err:
                        cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                        sys.exit(1)
                elif predlogext == '.html':
                    df.to_html(os.path.join(
                        savedir,'{}{}'.format(model.name, predlogext))
                        )
                else:
                    raise ValueError('predlogext: .csv/.xlsx/.html')

            if dpi is not None:
                if type_ == 'probability_frequency_histogram':
                    self.post.predict_max_histogram(
                            probs=probs, dpi=dpi,
                            caption=caption, savedir=savedir
                            )
                elif type_ == 'confusion_matrix':
                    self.post.class_confusion_matrix(
                            ytrues=trues, ypreds=preds,
                            dpi=dpi, annot=caption,
                            savedir=savedir, class_dict=class_dict
                            )
                else:
                    raise ValueError('Sorry, test now.')

    def __data_split(self, X, y, names, splitrate):
        Xlearn, Xpred, ylearn, ypred, _, npred = train_test_split(
                X, y, names, test_size=splitrate
                )

        return (Xlearn, ylearn), (Xpred, ypred, npred)

    def __forwards(self, Xshape, yshape):
        if self.modelname == 'sample1':
            top, model = self.sample_forward_1(
                    Xshape=Xshape, yshape=yshape,
                    nodes=self.nodes, activation=self.activation,
                    alpha=self.alpha, theta=self.theta
                    )
        elif self.modelname == 'sample2':
            top, model = self.sample_forward_2(
                    Xshape=Xshape, yshape=yshape,
                    nodes=self.nodes, activation=self.activation,
                    alpha=self.alpha, theta=self.theta
                    )
        elif self.modelname == 'vgg16':
            top, model = self.vgg16_forward(
                    Xshape=Xshape, yshape=yshape,
                    custom=self.custom, nodes=self.nodes,
                    activation=self.activation, alpha=self.alpha, theta=self.theta
                    )
        elif modelname == 'vgg19':
            top, model = self.vgg19_forward(
                    Xshape=Xshape, yshape=yshape,
                    custom=self.custom, nodes=self.nodes,
                    activation=self.activation, alpha=self.alpha, theta=self.theta
                    )
        else:
            raise ValueError('Sorry, test now.')

        return top, model

    def __forward_optimize(self, Xshape, yshape, param_dict):
        top, model = self.__forwards(Xshape=Xshape, yshape=yshape)

        if self.optflag:
            opt = optimizer_setup(optname=self.optname, param_dict=param_dict)
            loss = keras_losses(losstype='multi_classification')

            indic_obj = LearningIndicators()
            met = indic_obj.classify_metrics(num_classes=yshape[-1])

            model.compile(optimizer=opt, loss=loss, metrics=met)

        if self.summary:
            self.__cv_model_summary(top, model)

        return model

    def __cv_model_summary(self, top, model):
        if self.summaryout:
            try:
                with open('./{}_summary.log'.format(model.name), 'w') as s:

                    if top is not None:
                        top.summary(print_ln=lambda x:s.write(x+NEWLINECODE))

                    model.summary(print_ln=lambda x:s.write(x+NEWLINECODE))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('model.name: None', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            if top is not None:
                top.summary()

            model.summary()

    def __callbacks(self, model, savedir, cbs_tuple, cb_param_dict):
        cbs = list()
        for name in cbs_tuple:
            if 'csvlogger' in cbs_tuple:

                try:
                    csvfilepath = os.path.join(
                            savedir,'{}_learninglog.csv'.format(model.name)
                            )
                except:
                    cprint('model.name: None, auto-named.', 'yellow', attrs=['bold'])
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

    def __save_learning_params(self, model, savetype, include_opt, savedir, **kwargs):
        if savetype == 'total':
            savepath = os.path.join(savedir,'{}.{}.h5'.format(model.name,savetype))
            model.save(savepath, include_optimizer=include_opt)
        elif savetype == 'weight':
            savepath = os.path.join(savedir,'{}.{}.hdf5'.format(model.name,savetype))
            model.save_weights(savepath)
        elif savetype == 'model':
            for k in kwargs.keys():

                if k == '.yaml':
                    savepath = os.path.join(
                            savedir,'{}.{}{}'.format(
                                model.name,savetype,k)
                            )
                    save_str = model.to_yaml()
                else:
                    savepath = os.path.join(
                            savedir,'{}.{}.json'.format(
                                model.name,savetype)
                            )
                    save_str = model.to_json()

                with open(savepath, 'w') as sm:
                    sm.write(save_str+NEWLINECODE)
        else:
            raise ValueError('savetype: total/weight/model')
