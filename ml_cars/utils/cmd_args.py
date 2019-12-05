from utils.base_support import mkdirs
from termcolor import cprint
import argparse
import os
import json
import datetime
import glob
import sys

''' Global variables '''
DEFAULT_SAVEDIRPATH = os.path.join(
        os.path.expanduser('~'), 'scraping-images')

CONFIG_PATH_COMPO = os.path.dirname(os.path.abspath(__file__)).split('/')
del CONFIG_PATH_COMPO[-2:]
CONFIG_BASEPATH = os.path.join('/'.join(CONFIG_PATH_COMPO),'config')

OSNAME = os.name

if OSNAME == 'nt':
    NEWLINECODE = '\r\n'
else:
    NEWLINECODE = '\n'

''' functions and classes '''
def config_logs(methodname, logtype=None, num_history=50, rm_history=10, param_string=None):
    cfg_path = os.path.join(CONFIG_BASEPATH, '{}{}'.format(methodname,logtype))

    if logtype is not None:
        if logtype == '.log':
            try:
                if not os.path.exists(cfg_path):
                    with open(cfg_path, 'w') as cfg:
                        cfg.write('*** [{}] ***'.format(datetime.datetime.now())+NEWLINECODE)
                        cfg.write(param_string+NEWLINECODE)
                else:
                    with open(cfg_path, 'a') as cfg:
                        cfg.write('*** [{}] ***'.format(datetime.datetime.now())+NEWLINECODE)
                        cfg.write(param_string+NEWLINECODE)
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                sys.exit(1)
        else:
            raise ValueError('Sorry, logtype is .log only (test now)')

        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as cfg:
                lines = cfg.readlines()
                num_compo = len(lines)

                if num_compo >= num_history*2:
                    del lines[:rm_history]

            with open(cfg_path, 'w') as cfg:
                cfg.write(''.join(lines))

class CommandLineArgs:

    @classmethod
    def scrape_args(cls, logtype=None):
        methodname = sys._getframe().f_code.co_name # メソッドとかの名称をこれで取得可能

        p = argparse.ArgumentParser(
                description='Image scraping command-line arguments'
                )

        p.add_argument(
                '-e', '--engine', help='Web-search engine name',
                type=str, choices=['google','yahoo','bing'], default='google',
                required=False
                )
        p.add_argument(
                '-d', '--savedir', help='Scraped image directory path',
                type=str, default=DEFAULT_SAVEDIRPATH, required=False
                )
        p.add_argument(
                '-k', '--keyword', help='a search keyword',
                type=str, required=True
                )
        p.add_argument(
                '-n', '--num_images', help='max-scrap data number',
                type=int, default=100, required=False
                )

        p.add_argument(
                '--seed', help='user-agent mask random-seed flag',
                action='store_true'
                )

        param_obj = p.parse_args()

        config_logs(
                methodname=methodname,
                logtype=logtype,
                param_string=str(param_obj)
                )

        return param_obj

    @classmethod
    def data_arrangement_args(cls, logtype=None):
        methodname = sys._getframe().f_code.co_name

        p = argparse.ArgumentParser(
                description='Normal DNN Image-classfication arguments.'
                )
        p.add_argument(
                '-C', '--color', help='Image color',
                type=str, choices=['gray','rgb'], default='rgb',
                required=False
                )
        p.add_argument(
                '-H', '--height', help='Image resize-height',
                type=int, default=224, required=False
                )
        p.add_argument(
                '-W', '--width', help='Image resize-width',
                type=int, default=224, required=False
                )
        p.add_argument(
                '-P', '--preprocessing', help='Image normalization or standarize',
                type=str, choices=['normalize','standard'], default='normalize',
                required=False
                )
        p.add_argument(
                '-S', '--split', help='dataset split method',
                type=str, choices=['holdout','kfold'], default='holdout',
                required=False
                )
        p.add_argument(
                '-R', '--splitrate', help='holdout split test-size rate',
                type=float, default=0.2, required=False
                )
        p.add_argument(
                '-K', '--ksize', help='K-Fold split group-size',
                type=int, default=5, required=False
                )

        param_obj = p.parse_args()

        config_logs(
                methodname=methodname,
                logtype=logtype,
                param_string=str(param_obj)
                )

        return param_obj

    @classmethod
    def finetuning_args(cls):
        pass

    @classmethod
    def pretrain_args(cls):
        pass

    @classmethod
    def normal_dnn_args(cls, logtype=None):
        methodname = sys._getframe().f_code.co_name

        p = argparse.ArgumentParser(
                description='Normal DNN Image-classfication arguments.'
                )

        p.add_argument(
                '-C', '--color', help='Image color',
                type=str, choices=['gray','rgb'], default='rgb',
                required=False
                )
        p.add_argument(
                '-H', '--height', help='Image resize-height',
                type=int, default=224, required=False
                )
        p.add_argument(
                '-W', '--width', help='Image resize-width',
                type=int, default=224, required=False
                )
        p.add_argument(
                '-P', '--preprocessing', help='Image normalization or standarize',
                type=str, choices=['normalize','standard'], default='normalize',
                required=False
                )
        p.add_argument(
                '-S', '--split', help='dataset split method',
                type=str, choices=['holdout','kfold'], default='holdout',
                required=False
                )
        p.add_argument(
                '-R', '--splitrate', help='holdout split test-size rate',
                type=float, default=0.2, required=False
                )
        p.add_argument(
                '-K', '--ksize', help='K-Fold split group-size',
                type=int, default=5, required=False
                )
        p.add_argument(
                '-M', '--modelname', help='modelname you use',
                type=str, choices=['sample1','sample2','vgg16','vgg19'], default='sample1',
                required=False
                )
        p.add_argument(
                '-O', '--optname', help='model optimizer you use',
                type=str, choices=[
                    'adam','adadelta','adamax',
                    'adagrad','sgd','nadam',
                    'rmsprop'
                    ], default='adam', required=False
                )
        p.add_argument(
                '-N', '--nodes', help='sample model custom leyer nodes',
                type=int, default=1024, required=False
                )
        p.add_argument(
                '-A', '--activation', help='sample model custom layer activaion',
                type=str, choices=[
                    'relu','tanh','softplus','softsign','elu',
                    'selu','sigmoid','hard_sigmoid','linear',
                    'leakyrelu','prelu','threshold'
                    ], default='relu', required=False
                )
        p.add_argument(
                '-E', '--epochs', help='learning iterations',
                type=int, default=100, required=False
                )
        p.add_argument(
                '-B', '--batchsize', help='training batch size',
                type=int, default=20, required=False
                )
        p.add_argument(
                '-V', '--verbose', help='show learning condition',
                type=int, choices=[0,1,2], default=0, required=False
                )

        p.add_argument(
                '--gpusave', help='GPU save flag',
                action='store_true', required=False
                )
        p.add_argument(
                '--summary', help='show model architecuree summary',
                action='store_true', required=False
                )
        p.add_argument(
                '--summaryout', help='write summary logfile',
                action='store_true', required=False
                )
        p.add_argument(
                '--optflag', help='model compile flag',
                action='store_true', required=False
                )
        p.add_argument(
                '--caption', help='plot caption flag',
                action='store_true', required=False
                )
        p.add_argument(
                '--alpha', help='LeakyReLU alpha', type=float,
                default=0.3, required=False
                )
        p.add_argument(
                '--theta', help='ThresholdedReLU theta', type=float,
                default=1.0, required=False
                )
        p.add_argument(
                '--dpi', help='Matplotlib dpi param', type=int,
                default=500, required=False
                )

        param_obj = p.parse_args()

        config_logs(
                methodname=methodname,
                logtype=logtype,
                param_string=str(param_obj)
                )

        return param_obj

    @classmethod
    def dcgan_args(cls):
        pass

    @classmethod
    def vae_args(cls):
        pass

    @classmethod
    def unet_args(cls):
        pass

    @classmethod
    def super_resolution_args(cls):
        pass
