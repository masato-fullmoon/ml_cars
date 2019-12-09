from functools import partial
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from termcolor import cprint
import tensorflow as tf

def keras_losses(losstype='multi_classification'):
    if losstype == 'multi_classification':
        try:
            return losses.categorical_crossentropy
        except Exception as err:
            cprint('Error: {}'.format(str(err)), 'red', attrs['bold'])
            sys.exit(1)
    elif losstype == 'sparse_classification':
        return losses.sparse_categorical_crossentropy
    elif losstype == 'binary_classification':
        return losses.binary_crossentropy
    elif losstype == 'mse':
        return losses.mean_squared_error
    elif losstype == 'mae':
        return losses.mean_absolute_error
    elif losstype == 'mape':
        return losses.mean_absolute_percentage_error
    elif losstype == 'msle':
        return losses.mean_absolute_logarithmic_error
    elif losstype == 'hinge':
        return losses.hinge
    elif losstype == 's_hinge':
        return losses.squared_hinge
    elif losstype == 'c_hinge':
        return losses.categorical_hinge
    elif losstype == 'logcosh':
        return losses.logcosh
    elif losstype == 'kl_divergence':
        return losses.kullback_leibler_divergence
    elif losstype == 'poisson':
        return losses.poisson
    elif losstype == 'cosine':
        return losses.cosine_proximity
    else:
        err_msg = '''
--------- Loss Functions ---------

multi_classification : categorical crossentropy (multi-class classification, images-CNN)
sparse_classification: sparse classification (unknown detail)
binary_classification: binary classification (0 or 1, GAN)
mse                  : mean squared error (regression)
mae                  : mean absolute error (regression)
mape                 : mean absolute percentage error (Unknown detail)
msle                 : mean squared logarithmic error (Unknown detail)
hinge                : hinge (Unknown detail)
s_hinge              : squared hinge (Unknown detail)
c_hinge              : categorical hinge (Unknown detail)
logcosh              : log(cosh), if inputs>>0, logcosh = (x**2)/2
kl_divergence        : VAE loss function, KL divergence
poisson              : poisson coefficient (regression)
cosine               : cosine proximity, cosine similarity (data-relation)

-----------------------------------

        '''
        cprint(err_msg, 'yellow', attrs=['bold'])

        raise ValueError('Incorrect loss functions.')

class __IndicatorsBase:
    def __init__(self):
        pass

    # ---------- classfication indicators
    def normalize_y_pred(self, y_pred):
        return K.one_hot(K.argmax(y_pred), y_pred.shape[-1])

    def class_true_positive(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)

        return K.cast(K.equal(y_true[:, class_label]+y_pred[:, class_label],2), K.floatx())

    def class_accuracy(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)

        return K.cast(K.equal(y_true[:, class_label], y_pred[:, class_label]), K.floatx())

    def class_precision(self, class_label, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)

        return K.sum(self.class_true_positive(class_label, y_true, y_pred))/\
                (K.sum(y_pred[:, class_label])+K.epsilon())

    def class_recall(self, class_label, y_true, y_pred):
        return K.sum(self.class_true_positive(class_label, y_true, y_pred))/\
                (K.sum(y_true[:, class_label])+K.epsilon())

    def class_f_measure(self, class_label, y_true, y_pred):
        precision = self.class_precision(class_label, y_true, y_pred)
        recall = self.class_recall(class_label, y_true, y_pred)

        return (2.*precision*recall)/(precision+recall+K.epsilon())

    def true_positive(self, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)

        return K.cast(K.equal(y_true+y_pred, 2), K.floatx())

    def micro_precision(self, y_true, y_pred):
        y_pred = self.normalize_y_pred(y_pred)

        return K.sum(self.true_positive(y_true, y_pred))/(K.sum(y_pred)+K.epsilon())

    def micro_recall(self, y_true, y_pred):
        return K.sum(self.true_positive(y_true, y_pred))/(K.sum(y_true)+K.epsilon())

    def micro_f_measure(self, y_true, y_pred):
        precision = self.micro_precision(y_true, y_pred)
        recall = self.micro_recall(y_true, y_pred)

        return (2.*precision*recall)/(precision+recall+K.epsilon())

    def average_accuracy(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        class_acc_list = [self.class_accuracy(i, y_true, y_pred) for i in range(class_count)]
        class_acc_matrix = K.concatenate(class_acc_list, axis=0)

        return K.mean(class_acc_matrix, axis=0)

    def macro_precision(self, y_true, y_pred):
        class_count = y_pred.shape[-1]

        return K.sum([self.class_precision(i, y_true, y_pred) for i in range(class_count)]) \
                / K.cast(class_count, K.floatx())

    def macro_recall(self, y_true, y_pred):
        class_count = y_pred.shape[-1]

        return K.sum([self.class_recall(i, y_true, y_pred) for i in range(class_count)]) \
                / K.cast(class_count, K.floatx())

    def macro_f_measure(self, y_true, y_pred):
        precision = self.macro_precision(y_true, y_pred)
        recall = self.macro_recall(y_true, y_pred)

        return (2.*precision*recall)/(precision+recall+K.epsilon())

    # ---------- regression indicators
    def rms(self, y_true, y_pred):
        return ((K.mean(y_true-y_pred)**2))**0.5

    def r2(self, y_true, y_pred):
        return 1.-(K.sum((y_true-y_pred)**2)/(K.sum((y_true-K,mean(y_true))**2)+K.epsilon()))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean((y_true-y_pred)**2))

    def mce(self, y_true, y_pred):
        return K.mean(K.abs(y_true-y_pred))

    # ---------- autoencoder indicators
    def dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        return (2.*K.sum(y_true*y_pred))/(K.sum(y_true)+K.sum(y_pred)+K.epsilon())

    def jaccard_index(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        return (2.*K.sum(y_true*y_pred))/(K.sum(y_true)+\
                K.sum(y_pred)-K.sum(y_true*y_pred)+K.epsilon())

    def overlap_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        return (2.*K.sum(y_true*y_pred))/(K.min([K.sum(y_true),K.sum(y_pred)])+K.epsilon())

class LearningIndicators(__IndicatorsBase):
    def classify_metrics(self, num_classes=None):
        if (num_classes is not None) and (type(num_classes) is int):
            metrics = ['accuracy']

            func_list = [
                    self.class_accuracy,
                    self.class_precision,
                    self.class_recall,
                    self.class_f_measure
                    ]
            name_list = ['acc','precision','recall','f_measure']

            for i in range(num_classes):
                for func, name in zip(func_list, name_list):
                    func = partial(func, i)
                    func.__name__ = '{}-{}'.format(name, i)
                    metrics.append(func)

            metrics.append(self.average_accuracy)
            metrics.append(self.macro_precision)
            metrics.append(self.macro_recall)
            metrics.append(self.macro_f_measure)

            return metrics
        else:
            raise TypeError('num_classes: int type')
