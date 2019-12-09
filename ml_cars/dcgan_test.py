from models.gans import GANmodel
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

CMD_ARGS = CommandLineArgs.dcgan_args(logtype='.log')

COLOR_MODE = CMD_ARGS.color
HEIGHT = CMD_ARGS.height
WIDTH = CMD_ARGS.width
NORMTYPE = CMD_ARGS.preprocessing

GPUSAVE = CMD_ARGS.gpusave
SUMMARY = CMD_ARGS.summary
SUMMARYOUT = CMD_ARGS.summaryout
AUTO_ZDIM = CMD_ARGS.autozdim
ZDIM = CMD_ARGS.zdim
OPTNAME = CMD_ARGS.optname

EPOCHS = CMD_ARGS.epochs
BATCHSIZE = CMD_ARGS.batchsize
NUM_DIV = 2
SAVE_ITERATION = 100
DEBUG = CMD_ARGS.debug
GENERATE_TYPE = 'tiled_generate'
TILE_HEIGHT = 10
TIME_WIDTH = 10
DPI = CMD_ARGS.dpi

if __name__ == '__main__':
    ''' ----- dataset preprocessing ----- '''

    img_arrange = ImageArrangement(
            dataset_dirpath=DATADIR, color=COLOR_MODE,
            height=HEIGHT, width=WIDTH
            )
    img_arrange.preprocessing(normtype=NORMTYPE)
    X, _, _ = img_arrange.get_datasets()

    ''' ----- DCGAN images generate '''

    # Adamを使う時はこれらを**kwargsとしてインスタンスに入れる
    lr = 0.005
    beta_1 = 0.001
    beta_2 = 0.8
    amsgrad = False
    #decay = 1e-4
    #momentum = 0.9
    #nesterov = True

    dcgan = GANmodel(
            imgtensors=X, gpusave=GPUSAVE, summary=SUMMARY,
            summaryout=SUMMARYOUT, auto_zdim=AUTO_ZDIM, zdim=ZDIM,
            optnam=OPTNAME,
            lr=lr, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad
            )

    loss_dict = dcgan.dcgan_train(
            epochs=EPOCHS, batch_size=BATCHSIZE, num_div=NUM_DIV,
            save_iter=SAVE_ITERATION, debug=DEBUG, gentype=GENERATE_TYPE,
            tile_h=TILE_HEIGHT, tile_w=TIME_WIDTH, savedir=RESULT_BASEDIR,
            dpi=DPI
            )

    dcgan.property_visualization(
            loss_dict=loss_dict, gentype=GENERATE_TYPE,
            savedir=RESULT_BASEDIR, dpi=DPI
            )
