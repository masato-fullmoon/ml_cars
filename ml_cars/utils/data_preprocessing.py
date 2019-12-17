from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from termcolor import cprint
from utils.base_support import mkdirs
import numpy as np
import os
import sys
import glob

def dataset_rename(in_dirpath=None, out_dirpath=None, renametype='supervised'):
    '''
    以下のようなディレクトリ構成を想定している.
    それぞれの画像を格納したディレクトリの名称と画像インデックスを参照して
    別のディレクトリにrename保存する

    ex. ./cars
            |-----> ./cars/audi
                            |-----> 1-500 images

        (./cars/audi/img_1.jpg -> ./cars/rename_audi/audi_00001.jpg)

    '''
    assert in_dirpath is not out_dirpath,\
            'The same directories both input and output.'

    dirnames = os.listdir(in_dirpath)
    dirnames.sort()

    for d in dirnames:
        dirpath = os.path.join(in_dirpath,d)

        if renametype == 'supervised':
            savedir = mkdirs(os.path.join(out_dirpath,d))
        elif renametype == 'unsupervised':
            savedir = mkdirs(out_dirpath)
        else:
            raise ValueError('renametype: supervised/unsupervised')

        # dirpathがディレクトリであれば中まで参照してrenameする
        if os.path.isdir(dirpath):
            imgs = glob.glob(os.path.join(dirpath,'*'))
            imgs.sort()

            for i, img in enumerate(imgs):
                _, e = os.path.splitext(img)

                outname = '{}_{:05}{}'.format(d,i+1,e)
                outpath = os.path.join(savedir,outname)
                # 単純に同じディレクトリ内のファイル名を変更しようとするとバグるから注意
                os.rename(img, outpath)
        # ちがったらすっ飛ばす
        else:
            cprint('{} is not directory, pass'.format(d),
                    'yellow', attrs=['bold'])
            continue

        # 入力ディレクトリに画像がなくなったら消す
        if len(glob.glob(os.path.join(dirpath,'*'))) == 0:
            os.rmdir(dirpath)

    if len(glob.glob(os.path.join(in_dirpath,'*'))) == 0:
        cprint('No files in [{}], remove this dir.'.format(os.path.abspath(in_dirpath)),
                'cyan', attrs=['bold'])
        os.rmdir(in_dirpath)

class SampleImageArrangement:
    def __init__(self, imgtype='mnist', normtype='normalize'):
        if imgtype == 'mnist':
            # grayscale digit images (28,28)
            (Xlearn, ylearn), (Xpred, ypred) = mnist.load_data()

            # grayscaleはむりやり三次元テンソルにする (28,28,1)
            Xlearn = Xlearn[:,:,:,np.newaxis]
            Xpred = Xpred[:,:,:,np.newaxis]
        elif imgtype == 'cifar10':
            # RGB object images (28,28,3)
            (Xlearn, ylearn), (Xpred, ypred) = cifar10.load_data()
        else:
            raise ValueError('imgtype: mnist/cifar10')

        if normtype == 'normalize':
            Xlearn = Xlearn.astype('float32')/256.
            Xpred = Xpred.astype('float32')/256.
        elif normtype == 'standard':
            Xlearn = Xlearn.astype('float32')/127.5-1.0
            Xpreds = Xpreds.astype('float32')/127.5-1.0
        else:
            raise ValueError('normtype: normalize/standard')

        npred = np.array([str(n) for n in ypred], dtype=str)
        self.class_dict = {str(d):i for i,d in enumerate(set(ylearn.flatten()))}

        ylearn = to_categorical(ylearn, len(set(ylearn.flatten())))
        ypred = to_categorical(ypred, len(set(ypred.flatten())))

        self.learn_data = (Xlearn, ylearn)
        self.pred_data = (Xpred, ypred, npred)

    def get_datasets(self):
        return self.learn_data, self.pred_data

    def get_classdict(self, inverse=True):
        if inverse:
            return {v:k for k,v in self.class_dict.items()}
        else:
            return self.class_dict

    def split_dataset(self, X, y, method='holdout', splitrate=0.2, K=5):
        if method == 'holdout':
            assert splitrate is not None, 'holdout-splitrate is None.'

            data_g = self.__holdout_cv(X, y, splitrate=splitrate)
        elif method == 'kfold':
            assert K is not None, 'K-params of K-Fold is None.'

            data_g = self.__kfold_cv(X, y, K=K)
        else:
            raise ValueError('split-method: holdout or kfold')

        data = [next(data_g) for _ in range(2)] # リスト0: train, 1: validation

        return data

    def __holdout_cv(self, X, y, splitrate=0.2):
        if type(splitrate) is float:
            assert 0.0 < splitrate < 1.0, 'holdout-splitrate: from 0 to 1'
        else:
            raise TypeError('splitrate: float type between 0 and 1')

        Xtrain, Xval, ytrain, yval = train_test_split(
                X, y, test_size=splitrate)

        splitdata = [(Xtrain,ytrain),(Xval,yval)]
        for data in splitdata:
            yield data

    def __kfold_cv(self, X, y, K=5):
        if isinstance(K,(int,float)):
            if type(K) is float:
                K = round(K)
        else:
            raise TypeError('K-param: int or float type.')

        for train, val in KFold(n_splits=K, shuffle=True).split(X):
            Xtrain, Xval = Xlearn[train], Xlearn[val]
            ytrain, yval = ylearn[train], ylearn[val]

        splitdata = [(Xtrain,ytrain),(Xval,yval)]
        for data in splitdata:
            yield data

class ImageArrangement:
    def __init__(self, dataset_dirpath=None, color='rgb', height=224, width=224):
        self.datasetpath = dataset_dirpath
        self.h = height
        self.w = width

        if color == 'gray':
            self.mode = 'L'
        elif color == 'rgb':
            self.mode = 'RGB'
        else:
            raise ValueError('Sorry, PIL-color mode: L, RGB only.')

    def preprocessing(self, savenpz=None, normtype='normalize'):
        dirs = os.listdir(self.datasetpath)
        dirs.sort()

        dirs_dict = {
                d:i for i,d in enumerate(dirs) \
                if os.path.isdir(os.path.join(self.datasetpath,d))
                }
        imgs_dict = {
                d:glob.glob(os.path.join(self.datasetpath,d,'*')) \
                for d in dirs_dict.keys()
                }

        X_items = self.__X_generator(dirs_dict, imgs_dict)
        y_items = self.__y_generator(dirs_dict, imgs_dict)
        n_items = self.__n_generator(dirs_dict, imgs_dict)

        X_base = list()
        y_base = list()
        n_base = list()
        for _ in range(len(dirs_dict)):
            X_base += next(X_items)
            y_base += next(y_items)
            n_base += next(n_items)

        if self.mode == 'L':
            if normtype == 'normalize':
                X_tensor = np.array(X_base, dtype=np.float32)[:,:,:,np.newaxis]/256.
            elif normtype == 'standard':
                X_tensor = np.array(X_base, dtype=np.float32)[:,:,:,np.newaxis]/127.5-1.
        elif self.mode == 'RGB':
            if normtype == 'normalize':
                X_tensor = np.array(X_base, dtype=np.float32)/256.
            elif normtype == 'standard':
                X_tensor = np.array(X_base, dtype=np.float32)/127.5-1.

        y_vector = to_categorical(np.array(y_base, dtype=np.int32), len(dirs_dict))
        n_vector = np.array(n_base, dtype=str)

        if savenpz is not None:
            try:
                _, e = os.path.splitext(savenpz)

                if e == '.npz':
                    np.savez(savenpz, X=X_tensor, y=y_vector, names=n_vector)
                else:
                    raise Exception('Savefile extension: npz only.')
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('savenpz: Path-type string, extension is .npz', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            self.datasets = (X_tensor, y_vector, n_vector)

        self.class_dict = dirs_dict

    def get_datasets(self):
        return self.datasets

    def get_classdict(self, inverse=True):
        if inverse:
            # 後々予測のために逆引きが可能なようにしている
            return {v:k for k,v in self.class_dict.items()}
        else:
            return self.class_dict

    def load_datasets(self, loadnpz=None):
        if loadnpz is not None:
            X_tensor, y_vector, n_vector = np.load(loadnpz)

        if loadnpz is not None:
            try:
                _, e = os.path.splitext(loadnpz)

                if e == '.npz':
                    data = np.load(loadnpz)
                else:
                    raise Exception('Savefile extension: npz only')
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('savenpz: Path-type string, extension is .npz', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            raise TypeError('loadnpz is None type.')

        return (data['X_tensor'], data['y_vector'], data['n_vector'])

    def data_split(self, X, y, names, method='holdout', splitrate=None, K=None, savenpz=None):
        if method == 'holdout':
            assert splitrate is not None, 'holdout-splitrate is None.'

            data_g = self.__holdout_cv(X, y, names, splitrate=splitrate)
        elif method == 'kfold':
            assert K is not None, 'K-params of K-Fold is None.'

            data_g = self.__kfold_cv(X, y, names, K=K)
        else:
            raise ValueError('split-method: holdout or kfold')

        data = [next(data_g) for _ in range(3)] # リスト0: train, 1: validation, 2: テスト(nameもある)

        if savenpz is not None:
            try:
                _, e = os.path.splitext(savenpz)

                if e == '.npz':
                    trains, vals, preds = data # リストを展開

                    np.savez(savenpz, Xtrain=trains[0], ytrain=trains[1],
                            Xval=vals[0], yval=vals[1],
                            Xpred=preds[0], ypred=preds[1], npred=preds[2])
                else:
                    raise Exception('Savefile extension: npz only.')
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('savenpz: Path-type string, extension is .npz', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            self.splitdata = data

    def get_splitdata(self):
        return self.splitdata

    def load_splitnpz(self, loadnpz=None):
        if loadnpz is not None:
            try:
                _, e = os.path.splitext(loadnpz)

                if e == '.npz':
                    data = np.load(loadnpz)
                else:
                    raise Exception('Savefile extension: npz only')
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                cprint('savenpz: Path-type string, extension is .npz', 'red', attrs=['bold'])
                sys.exit(1)
        else:
            raise TypeError('loadnpz is None type.')

        return (data['Xtrain'],data['ytrain']), \
                (data['Xval'],data['yval']), \
                (data['Xpred'],data['ypred'],data['npred'])

    def __X_generator(self, dirs_dict, imgs_dict):
        for d in dirs_dict.keys():
            imgs_list = imgs_dict[d]
            imgs_list.sort()

            X_items = [
                    np.array(Image.open(img).convert(self.mode).resize((self.h,self.w)),
                        dtype=np.float32) for img in imgs_list
                    ]

            yield X_items

    def __y_generator(self, dirs_dict, imgs_dict):
        for d in dirs_dict.keys():
            imgs_list = imgs_dict[d]
            imgs_list.sort()

            y_items = [dirs_dict[d] for _ in range(len(imgs_list))]

            yield y_items

    def __n_generator(self, dirs_dict, imgs_dict):
        for d in dirs_dict.keys():
            imgs_list = imgs_dict[d]
            imgs_list.sort()

            n_items = [
                    os.path.basename(img).split('.')[0] \
                    for img in imgs_list
                    ]

            yield n_items

    def __holdout_cv(self, X, y, names, splitrate=0.2):
        if type(splitrate) is float:
            assert 0<splitrate<1, 'holdout-splitrate: from 0 to 1'
        else:
            raise TypeError('splitrate: float type between 0 and 1')

        Xlearn, Xpred, ylearn, ypred, _, npred = train_test_split(
                X, y, names, test_size=splitrate)

        Xtrain, Xval, ytrain, yval = train_test_split(
                Xlearn, ylearn, test_size=splitrate)

        splitdata = [(Xtrain,ytrain),(Xval,yval),(Xpred,ypred,npred)]
        for data in splitdata:
            yield data

    def __kfold_cv(self, X, y, names, K=5):
        if isinstance(K,(int,float)):
            if type(K) is float:
                K = round(K)
        else:
            raise TypeError('K-param: int or float type.')

        for learn, pred in KFold(n_splits=K, shuffle=True).split(X):
            Xlearn, Xpred = X[learn], X[pred]
            ylearn, ypred = y[learn], y[pred]
            _, npred = names[learn], names[pred]

        for train, val in KFold(n_splits=K, shuffle=True).split(Xlearn):
            Xtrain, Xval = Xlearn[train], Xlearn[val]
            ytrain, yval = ylearn[train], ylearn[val]

        splitdata = [(Xtrain,ytrain),(Xval,yval),(Xpred,ypred,npred)]
        for data in splitdata:
            yield data
