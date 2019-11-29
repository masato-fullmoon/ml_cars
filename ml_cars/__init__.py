from . import models
from . import scrape
from . import utils
from termcolor import cprint
import sys

if __name__ == '__main__':

    if sys.version_info[0] != 3:
        cprint('This python version is 2.x.x, Use 3.x.x', 'red', 'on_white')
        sys.exit(1)
