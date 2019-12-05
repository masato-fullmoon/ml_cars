from models.normal_dnn_classify import NormalDNNCrossValidation
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

    ''' ----- cross validation normal DNN train ----- '''

    normaldnn = NormalDNNCrossValidation()
