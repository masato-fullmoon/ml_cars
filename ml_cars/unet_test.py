from models.autoencoders import UnetAE
from utils.data_preprocessing import ImageArrangement
from utils.cmd_args import CommandLineArgs
import os
import sys

# ----- global path setup -----

DATADIR = os.path.join(
        os.path.expanduser('~'),
        'work/git_work/datasets/supervised/rename_cars'
        )
RESULT_BASEDIR = os.path.join(
        os.path.expanduser('~'),
        'research_results/ml/original_ml'
        )

# ----- global commandline arguments setup -----

CMD_ARGS = CommandLineArgs.unet_args(logtype='.log')

COLOR_MODE = CMD_ARGS.color
HEIGHT = CMD_ARGS.height
WIDTH = CMD_ARGS.width
NORMTYPE = CMD_ARGS.preprocessing
METHOD = CMD_ARGS.split
SPLITRATE = CMD_ARGS.splitrate
K = CMD_ARGS.ksize

GPUSAVE = CMD_ARGS.gpusave
SUMMARY = CMD_ARGS.summary
SUMMARYOUT = CMD_ARGS.summaryout
OPT = CMD_ARGS.optname
ACTIVATION = CMD_ARGS.activation
OUTPUT_ACTIVATION = 'linear'
FILTERS = CMD_ARGS.filters
ALPHA = CMD_ARGS.alpha
THETA = CMD_ARGS.theta
OPTFLAG = CMD_ARGS.optflag

EPOCHS = CMD_ARGS.epochs
BATCHSIZE = CMD_ARGS.batchsize
VERBOSE = CMD_ARGS.verbose
DPI = CMD_ARGS.dpi
CAPTION = CMD_ARGS.caption

SAVEMODELTYPE = 'total'
TRAINMODELTYPE = 'unet'
#TRAINMODELTYPE = 'encoder'

if __name__ == '__main__':
    ''' ----- dataset preprocessing ----- '''

    img_arrange = ImageArrangement(
            dataset_dirpath=DATADIR, color=COLOR_MODE,
            height=HEIGHT, width=WIDTH
            )

    img_arrange.preprocessing(normtype=NORMTYPE)
    X, y, names = img_arrange.get_datasets()

    img_arrange.data_split(X, y, names, method=METHOD, splitrate=SPLITRATE, K=K)
    trains, vals, preds = img_arrange.get_splitdata()

    ''' ----- U-net autoencoder model ----- '''

    # 適当な勾配器を使う時はこれらを**kwargsとしてインスタンスに入れる
    lr = 2e-4
    beta_1 = 0.05
    beta_2 = 0.67
    amsgrad = False
    decay = 1e-5
    momentum = 0.9
    nesterov = True

    unet = UnetAE(
            Xtrain=trains[0], gpusave=GPUSAVE, summary=SUMMARY,
            summaryout=SUMMARYOUT, optflag=OPTFLAG, optname=OPT,
            activation=ACTIVATION, alpha=ALPHA, theta=THETA,
            out_act=OUTPUT_ACTIVATION, input_filters=FILTERS, traintype=TRAINMODELTYPE,
            lr=lr, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad,
            decay=decay, momentum=momentum, nesterov=nesterov
            )

    # 使用するcallback名はタプルで渡す, 現在はcsvとes, tensorboardのみ
    cbs_tuple = ('csvlogger','earlystopping')

    # callbackの引数は**kwargsでメソッドに渡す
    monitor = 'val_loss'
    min_delta = 0
    patience = EPOCHS//2
    mode = 'auto'

    # cross-validationなしでのtrain
    unet.unet_train(
            cbs_tuple=cbs_tuple, Xval=vals[0], epochs=EPOCHS,
            batch_size=BATCHSIZE, verbose=VERBOSE, savebasedir=RESULT_BASEDIR,
            caption=CAPTION, dpi=DPI, savetype=SAVEMODELTYPE,
            include_opt=True,
            monitor=monitor, min_delta=min_delta, patience=patience, mode=mode
            )

    # predデータを使用した画像の復元(cross-validationなし)
    unet.generate_images(
            Xpred=preds[0], npred=preds[2], batch_size=BATCHSIZE,
            verbose=VERBOSE, savebasedir=RESULT_BASEDIR,
            normtype=NORMTYPE, color=COLOR_MODE
            )
