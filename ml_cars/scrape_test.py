from scrape.image_scrape import ImageScraping
from utils.cmd_args import CommandLineArgs
import os

'''
setup global variable parameters
'''
CMD_ARGS = CommandLineArgs.scrape_args(logtype='.log')

ENGINE = CMD_ARGS.engine
SAVEDIR = os.path.abspath(CMD_ARGS.savedir)
KEYWORD = CMD_ARGS.keyword
N_IMAGES = CMD_ARGS.num_images
SEED_FLAG = CMD_ARGS.seed

if SEED_FLAG:
    SEED = 100 # ご自由に変更してください
else:
    SEED = None

if __name__ == '__main__':
    img_scrape = ImageScraping(site=ENGINE, seed=SEED, savedir=SAVEDIR)
    img_scrape.run_scrape(keyword=KEYWORD, n_img=N_IMAGES)
