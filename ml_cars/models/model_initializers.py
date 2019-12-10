from tensorflow.keras.initializers import *
from termcolor import cprint
import sys

class __InitializerBase:
    def __init__(self):
        self.msg = '''
---------- Initializer names ----------

base            : Layer initializer basis
zeros           : Initialize a layer with 0-tensor weight
ones            : Initialize a layer with 1-tensor weight
constant        : Initialize a layer with constant value tensor weight
random_normal   : Random normal distribution weight
random_uniform  : Random uniform distribution weight
variance_scaling: Auto scaling tensor weight
orthogonal      : Orthogonal tensor weight
identify        : Identify tensor weight

********** Should set these initializers on a layer **********

glorot_normal
glorot_uniform
he_normal
he_uniform
lecun_normal
lecun_uniform

---------- Initializer arguments ----------

value       : 'constant' Initializer arg, set positive float value.
mean        : mean float value
std         : standard deviation float value
min_        : 'random_uniform' only, any minimum float value
max_        : 'random_uniform' only, any maximum float value
scale       : 'variance_scaling' only, positive float real number value
mode        : 'variance_scaling' only, standard deviation scaling

        fan_in => input unit
        fan_out => output unit
        fan_ave => average between input and output unit

distribution: 'variance_scaling' only, normal/uniform
gain        : tensor multipul float value
seed        : random seed

[example]

        >>> obj = KernelInit()
        >>> k_init = obj.kernel_initializer(
                initname='variance_scaling',
                scale=1.0,
                mode='fan_in',
                distribution='normal',
                seed=None
                )
        >>> ...
        >>> x = Dense(64, kernel_initializer=k_init)(x)

        '''

    def base_initializer(self):
        return Initializer()

    def zeros(self):
        return Zeros()

    def ones(self):
        return Ones()

    def constant(self, value=0):
        assert isinstance(value, (int,float)), \
                'value: integer/float type.'

        return Constant(value=value)

    def random_normal(self, mean=0.0, std=0.05, seed=None):
        return RandomNormal(mean=mean, stddev=std, seed=seedf)

    def random_uniform(self, min_=-0.05, max_=0.05, seed=None):
        return RandomUniform(minval=min_, maxval=max_, seed=seed)

    def truncated_normal(self, mean=0.0, std=0.05, seed=None):
        return TruncatedNormal(mean=mean, stddev=std, seed=seed)

    def variance_scaling(self, scale=1.0, mode='fan_in', distribution='normal', seed=None):
        assert scale > 0.0, 'scale: positive real number'
        assert mode in ('fan_in','fan_out','fan_ave'), \
                'mode: fan_in/fan_out/fan_ave'
        assert distribution in ('normal','uniform'), \
                'distribution: normal/uniform'

        return VarianceScaling(scale=scale, mode=mode, distribution=distribution, seed=seed)

    def orthogonal(self, gain=1.0, seed=None):
        return Orthogonal(gain=gain, seed=seed)

    def identify(self, gain=1.0):
        return Identify(gain=gain)

    def glorot_normal(self, seed=None):
        return glorot_normal(seed=seed)

    def glorot_uniform(self, seed=None):
        return glorot_uniform(seed=seed)

    def he_normal(self, seed=None):
        return he_normal(seed=seed)

    def he_uniform(self, seed=None):
        return he_uniform(seed=seed)

    def lecun_normal(self, seed=None):
        return lecun_normal(seed=seed)

    def lecun_uniform(self, seed=None):
        return lecun_uniform(seed=seed)

class KernelInit(__InitializerBase):
    def __init__(self):
        super().__init__()

    def kernel_initializer(self, initname='he_normal', **kwargs):
        try:
            if initname == 'base':
                return self.base_initializer()
            elif initname == 'zeros':
                return self.zeros()
            elif initname == 'ones':
                return self.ones()
            elif initname == 'constant':
                return self.constant(kwargs['value'])
            elif initname == 'random_normal':
                return self.random_normal(kwargs['mean'], kwargs['std'], kwargs['seed'])
            elif initname == 'random_uniform':
                return self.random_uniform(kwargs['min_'], kwargs['max_'], kwargs['seed'])
            elif initname == 'truncated_normal':
                return self.truncated_normal(kwargs['mean'], kwargs['std'], kwargs['seed'])
            elif initname == 'variance_scaling':
                return self.variance_scaling(kwargs['scale'], kwargs['mode'],
                        kwargs['distribution'], kwargs['seed'])
            elif initname == 'orthogonal':
                return self.orthogonal(kwargs['gain'], kwargs['seed'])
            elif initname == 'identify':
                return self.identify(kwargs['gain'])
            elif initname == 'glorot_normal':
                return self.glorot_normal(kwargs['seed'])
            elif initname == 'glorot_uniform':
                return self.glorot_uniform(kwargs['seed'])
            elif initname == 'he_normal':
                return self.he_normal(kwargs['seed'])
            elif initname == 'he_uniform':
                return self.he_uniform(kwargs['seed'])
            elif initname == 'lecun_normal':
                return self.lecun_normal(kwargs['seed'])
            elif initname == 'lecun_uniform':
                return self.lecun_uniform(kwargs['seed'])
            else:
                raise ValueError('Incorrect initname.')
        except Exception as err:
            cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
            cprint(self.msg, 'yellow', attrs=['bold'])
            sys.exit(1)

class BiasInit(__InitializerBase):
    def __init__(self):
        super().__init__()

    def bias_initializer(self, initname='zeros', **kwargs):
        try:
            if initname == 'base':
                return self.base_initializer()
            elif initname == 'zeros':
                return self.zeros()
            elif initname == 'ones':
                return self.ones()
            elif initname == 'constant':
                return self.constant(kwargs['value'])
            elif initname == 'random_normal':
                return self.random_normal(kwargs['mean'], kwargs['std'], kwargs['seed'])
            elif initname == 'random_uniform':
                return self.random_uniform(kwargs['min_'], kwargs['max_'], kwargs['seed'])
            elif initname == 'truncated_normal':
                return self.truncated_normal(kwargs['mean'], kwargs['std'], kwargs['seed'])
            elif initname == 'variance_scaling':
                return self.variance_scaling(kwargs['scale'], kwargs['mode'],
                        kwargs['distribution'], kwargs['seed'])
            elif initname == 'orthogonal':
                return self.orthogonal(kwargs['gain'], kwargs['seed'])
            elif initname == 'identify':
                return self.identify(kwargs['gain'])
            elif initname == 'glorot_normal':
                return self.glorot_normal(kwargs['seed'])
            elif initname == 'glorot_uniform':
                return self.glorot_uniform(kwargs['seed'])
            elif initname == 'he_normal':
                return self.he_normal(kwargs['seed'])
            elif initname == 'he_uniform':
                return self.he_uniform(kwargs['seed'])
            elif initname == 'lecun_normal':
                return self.lecun_normal(kwargs['seed'])
            elif initname == 'lecun_uniform':
                return self.lecun_uniform(kwargs['seed'])
            else:
                raise ValueError('Incorrect initname.')
        except Exception as err:
            cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
            cprint(self.msg, 'yellow', attrs=['bold'])
            sys.exit(1)
