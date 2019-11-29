from models.normal_dnn_classify import NormalDNN
from utils.data_preprocessing import ImageArrangement
from utils.cmd_args import CommandLineArgs
import os
import sys

DATADIR = os.path.join(
        os.path.expanduser('~'),
        'work/git_work/datasets/supervised/rename_cars'
        )

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

    ''' ----- dnn model classification ----- '''

    # Adamを使う時はこれらを**kwargsとしてインスタンスに入れる
    lr = 5e-5
    beta_1 = 1e-6
    beta_2 = 0.9
    amsgrad = True

    normaldnn = NormalDNN(
            Xtrain=trains[0], ytrain=trains[1], gpusave=GPUSAVE,
            summary=SUMMARY, summaryout=SUMMARYOUT, modelname=MODEL,
            optname=OPT, fc_nodes=FC_NODES, fc_act=FC_ACTIVATION,
            alpha=ALPHA, theta=THETA, optflag=OPTFLAG,
            lr=lr, beta_1=beta_1, beta_2=beta_2,
            amsgrad=amsgrad
            )
