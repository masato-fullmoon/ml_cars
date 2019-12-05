from models.normal_dnn_classify import NormalDNN
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

CMD_ARGS = CommandLineArgs.normal_dnn_args(logtype='.log')

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
MODEL = CMD_ARGS.modelname
OPT = CMD_ARGS.optname
FC_NODES = CMD_ARGS.nodes
FC_ACTIVATION = CMD_ARGS.activation
ALPHA = CMD_ARGS.alpha
THETA = CMD_ARGS.theta
OPTFLAG = CMD_ARGS.optflag
EPOCHS = CMD_ARGS.epochs
BATCHSIZE = CMD_ARGS.batchsize
VERBOSE = CMD_ARGS.verbose
DPI = CMD_ARGS.dpi
CAPTION = CMD_ARGS.caption

if __name__ == '__main__':
    ''' ----- dataset preprocessing ----- '''

    img_arrange = ImageArrangement(
            dataset_dirpath=DATADIR, color=COLOR_MODE,
            height=HEIGHT, width=WIDTH
            )
    img_arrange.preprocessing(normtype=NORMTYPE)
    X, y, names = img_arrange.get_datasets()

    class_dict = img_arrange.get_classdict() # 引数をFalseにしないと逆引き用の辞書になる

    img_arrange.data_split(X, y, names, method=METHOD, splitrate=SPLITRATE, K=K)
    trains, vals, preds = img_arrange.get_splitdata()

    ''' ----- dnn model classification ----- '''

    # Adamを使う時はこれらを**kwargsとしてインスタンスに入れる
    lr = 0.005
    #beta_1 = 0.001
    #beta_2 = 0.8
    #amsgrad = False
    decay = 1e-4
    momentum = 0.9
    nesterov = True

    normaldnn = NormalDNN(
            Xtrain=trains[0], ytrain=trains[1], gpusave=GPUSAVE,
            summary=SUMMARY, summaryout=SUMMARYOUT, modelname=MODEL,
            optname=OPT, fc_nodes=FC_NODES, fc_act=FC_ACTIVATION,
            alpha=ALPHA, theta=THETA, optflag=OPTFLAG,
            momentum=momentum, nesterov=nesterov, lr=lr,
            decay=decay
            )

    # 使用するcallback名はタプルで渡す, 現在はcsvとes, tensorboardのみ
    cbs_tuple = ('csvlogger','earlystopping')

    # callbackの引数は**kwargsでメソッドに渡す
    monitor = 'val_loss'
    min_delta = 0
    patience = EPOCHS//2
    mode = 'auto'

    # cross-validationなしでのtrain
    normaldnn.model_train(
            cbs_tuple=cbs_tuple, Xval=vals[0], yval=vals[1],
            epochs=EPOCHS, batch_size=BATCHSIZE, verbose=VERBOSE,
            savebasedir=RESULT_BASEDIR, dpi=DPI, caption=CAPTION,
            monitor=monitor, min_delta=min_delta, patience=patience,
            mode=mode
            )

    # predデータを使用した予測(cross-validationなし)
    normaldnn.prediction(
            class_dict=class_dict, Xpred=preds[0], ypred=preds[1],
            npred=preds[2], savebasedir=RESULT_BASEDIR, dpi=DPI,
            caption=CAPTION, predlogext='.xlsx', type_='confusion_matrix'
            )
