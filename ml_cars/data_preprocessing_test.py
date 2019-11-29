from utils.data_preprocessing import dataset_rename
from utils.data_preprocessing import ImageArrangement
from utils.cmd_args import CommandLineArgs
import os
import sys

BASEDATASET_DIR = os.path.join(
        os.path.expanduser('~'),'work/git_work/datasets'
        )
ORIGINAL_DIR = os.path.join(BASEDATASET_DIR, 'cars')
RENAMED_DIR = os.path.join(BASEDATASET_DIR, 'supervised/rename_cars')

CMD_OBJ = CommandLineArgs.data_arrangement_args(logtype='.log')

COLOR_MODE = CMD_OBJ.color
HEIGHT = CMD_OBJ.height
WIDTH = CMD_OBJ.width
NORMTYPE = CMD_OBJ.preprocessing
METHOD = CMD_OBJ.split
SPLITRATE = CMD_OBJ.splitrate
K = CMD_OBJ.ksize

if __name__ == '__main__':
    #dataset_rename(ORIGINAL_DIR, RENAMED_DIR)

    img_arrange = ImageArrangement(
            dataset_dirpath=RENAMED_DIR, color=COLOR_MODE,
            height=HEIGHT, width=WIDTH
            )
    img_arrange.preprocessing(normtype=NORMTYPE)
    X, y, names = img_arrange.get_datasets()

    img_arrange.data_split(X, y, names, method=METHOD, splitrate=SPLITRATE, K=K)
    trains, vals, preds = img_arrange.get_splitdata()
