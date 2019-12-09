from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from termcolor import cprint
import sys

def optimizer_setup(optname, param_dict):
    try:
        if type(param_dict) is dict:
            param_keys = list(param_dict.keys())
            decay_flag = False
        else:
            raise TypeError('keras optimization-parameters: dict type.')
    except Exception as err:
        cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
        cprint('keras_optparams: dict type.', 'red', attrs=['bold'])
        sys.exit(1)

    if 'decay' in param_dict.keys():
        decay_flag = True

    if optname == 'adam':
        if decay_flag:
            opt = Adam(
                    lr=param_dict['lr'],
                    beta_1=param_dict['beta_1'],
                    beta_2=param_dict['beta_2'],
                    amsgrad=param_dict['amsgrad'],
                    decay=param_dict['decay']
                    )
        else:
            opt = Adam(
                    lr=param_dict['lr'],
                    beta_1=param_dict['beta_1'],
                    beta_2=param_dict['beta_2'],
                    amsgrad=param_dict['amsgrad']
                    )
    elif optname == 'adamax':
        if decay_flag:
            opt = Adamax(
                    lr=param_dict['lr'],
                    beta_1=param_dict['beta_1'],
                    beta_2=param_dict['beta_2'],
                    amsgrad=param_dict['amsgrad'],
                    decay=param_dict['decay']
                    )
        else:
            opt = Adamax(
                    lr=param_dict['lr'],
                    beta_1=param_dict['beta_1'],
                    beta_2=param_dict['beta_2'],
                    amsgrad=param_dict['amsgrad']
                    )
    elif optname == 'adagrad':
        opt = Adagrad() # なるべくパラメータをいじらないようにしてとのこと
    elif optname == 'adadelta':
        opt = Adadelta() # 同様
    elif optname == 'sgd':
        opt = SGD(
                lr=param_dict['lr'],
                momentum=param_dict['momentum'],
                decay=param_dict['decay'],
                nesterov=param_dict['nesterov']
                )
    elif optname == 'nadam':
        opt = Nadam(
                lr=param_dict['lr'],
                beta_1=param_dict['beta_1'],
                beta_2=param_dict['beta_2'],
                schedule_decay=param_dict['decay']
                )
    elif optname == 'rmsprop':
        opt = RMSprop(
                lr=param_dict['lr'],
                rho=0.9, # lr以外は基本的にいじらないようにとのこと, 心配ならdecay=0.0に
                decay=param_dict['decay']
                )
    else:
        raise ValueError('Optimizer: adam/adamax/adagrad/adadelta/sgd/nadam/rmsprop')

    return opt
