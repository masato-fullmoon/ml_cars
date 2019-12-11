from tensorflow.keras.regularizers import *
from termcolor import cprint
import sys

def regularizers(regname='lasso', l1=0.01, l2=0.01):
    try:
        if regname == 'lasso':
            return l1(l=l1)
        elif regname == 'ridge':
            return l2(l=l2)
        elif regname == 'elasticnet':
            return l1_l2(l1=l1, l2=l2)
        elif regname == 'nothing':
            return None
        else:
            raise ValueError('Sorry, test now.')
    except Exception as err:
        msg = '''
---------- Layer Regularizers ----------

lasso     : L1 regularizer, what we call, Lasso regularization
ridge     : L2 regularizer, what we call, Ridge regularization
elasticnet: Layer regularization with L1 and L2, what we call ElasticNet regularization

---------- regularization parameters ----------

l : lasso and ridge param, positive float type.
l1: elasticnet only, positive float type.
l2: elasticnet only, positive float type.

---------- example usage ----------

>>> from models.layer_regularizers import regularizers
>>> reg = regularizers(regname='lasso', l1=0.01)
>>> ...
>>> inputs = Input(shape=(100,))
>>> x = Dense(64, kernel_regularizer=reg)(inputs)
>>> ...

or

>>> ...
>>> x = Dense(64, kernel_regularizer=regularizers(
    regname='lasso', l1=0.001))(inputs)
>>> x = Dense(64, kernel_regularizer=regularizers(
    regname='elasticnet', l1=0.001, l2=0.05))(x)
>>> ...

---------- end ----------
        '''

        cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
        cprint(msg, 'yellow', attrs=['bold'])
        sys.exit(1)
