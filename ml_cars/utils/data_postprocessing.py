from termcolor import cprint
from PIL import Image
from sklearn.metrics import confusion_matrix
from functools import reduce
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import glob
import math
import json
import operator

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
finally:
    import matplotlib.animation as anime

CONFIG_PATH_COMPO = os.path.dirname(os.path.abspath(__file__)).split('/')
del CONFIG_PATH_COMPO[-2:]
CONFIG_BASEPATH = os.path.join('/'.join(CONFIG_PATH_COMPO),'config')

VISUALIZE_CFG = os.path.join(CONFIG_BASEPATH,'visualize_cfg.json')

def mplparams_rewrite(**kwargs):
    pass

class __VisualizerBase:
    def __init__(self):
        self.font_size = 10
        self.xtick_direction = 'in'
        self.ytick_direction = 'in'
        self.xtick_major_width = 1.2
        self.ytick_major_width = 1.2
        self.axes_linewidth = 1.2
        self.axes_grid = True
        self.grid_linestyle = '--'
        self.grid_linewidth = 0.3
        self.legend_markerscale = 2
        self.legend_fancybox = False
        self.legend_framealpha = 1
        self.legend_edgecolor = 'black'

        if not os.path.exists(VISUALIZE_CFG):
            mpl_dict = self.__init_params_setup()

            for k in mpl_dict.keys():
                try:
                    plt.rcParams[k] = mpl_dict[k]
                except Exception as err:
                    cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                    cprint('Incorrect Param: [{}]'.format(k), 'red', attrs=['bold'])
                    sys.exit(1)
        else:
            with open(VISUALIZE_CFG, 'r') as mpl_json:
                mpl_dict = json.load(mpl_json)

                for k in mpl_dict.keys():
                    try:
                        plt.rcParams[k] = mpl_dict[k]
                    except Exception as err:
                        cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                        cprint('Incorrect Param: [{}]'.format(k), 'red', attrs=['bold'])
                        sys.exit(1)

    def __init_params_setup(self):
        mpl_dict = {
                'font.size':self.font_size,
                'xtick.direction':self.xtick_direction,
                'ytick.direction':self.ytick_direction,
                'xtick.major.width':self.xtick_major_width,
                'ytick.major.width':self.ytick_major_width,
                'axes.linewidth':self.axes_linewidth,
                'axes.grid':self.axes_grid,
                'grid.linestyle':self.grid_linestyle,
                'grid.linewidth':self.grid_linewidth,
                'legend.markerscale':self.legend_markerscale,
                'legend.fancybox':self.legend_fancybox,
                'legend.framealpha':self.legend_framealpha,
                'legend.edgecolor':self.legend_edgecolor
                }

        with open(VISUALIZE_CFG, 'w') as mpl_json:
            json.dump(mpl_dict, mpl_json)

        return mpl_dict

class Visualizer(__VisualizerBase):
    def __init__(self):
        super().__init__()

    def history_plot(self, history, indicator=None, dpi=500, caption=True, savedir=None):
        try:
            train_indices = history.history[indicator]
            val_indices = history.history['val_{}'.format(indicator)]
        except Exception as err:
            cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
            sys.exit(1)

        plot_path = os.path.join(savedir,'{}.jpg'.format(indicator))

        plt.plot(train_indices, label='train-{}'.format(indicator), color='red', lw=5)
        plt.plot(val_indices, label='val-{}'.format(indicator), color='green', lw=5)

        if caption:
            plt.xlabel('Iterations')
            plt.ylabel('{}'.format(indicator.capitalize()))
            plt.title('Learning Curve for {}'.format(indicator.capitalize()))

        plt.legend()
        plt.savefig(plot_path, dpi=dpi)
        plt.close()

    def predict_max_histogram(self, probs, dpi=500, caption=True, savedir=None):
        if isinstance(probs, (list, numpy.array)):
            hist_path = os.path.join(savedir,'pred_histogram.jpg')

            plt.hist(pd.Series(probs), lw=5, bins=100, color='green', normed=True, ec='black')

            if caption:
                plt.xlabel('Max Probability')
                plt.ylabel('Normalized Frequency')
                plt.title('Frequency for Max Probability')

            plt.savefig(hist_path, dpi=dpi)
            plt.close()
        else:
            raise TypeError('probs: list or numpy vector')

    def class_confusion_matrix(self, class_dict, ytrues, ypreds,
            dpi=500, annot=False, savedir=None):
        heatmap_path = os.path.join(savedir, 'class_cm_heatmap.jpg')

        # DataFrameにすると, ラベル名が変更できるので便利
        df = pd.DataFrame(
                data=confusion_matrix(ytrues, ypreds),
                index=list(class_dict.values()),
                columns=list(class_dict.values())
                )

        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(
                df, cmap='Blues', square=True,
                annot=annot, linecolor='white', linewidths=0.05,
                fmt='d'
                )
        ax.set_ylim(len(set(ytrues)),0) # これをしないとheatmapが見切れる

        plt.savefig(heatmap_path, dpi=dpi)
        plt.close()

    def save_tiled_generate(self, epoch, generator, zdim=None,
            tile_h=10, tile_w=10, savedir=None, dpi=500):
        generater_noise = np.random.normal(0,1,(reduce(operator.mul,(tile_h,tile_w)), zdim))
        gen_imgs = 0.5*generator.predict(generater_noise)+0.5

        fig, ax = plt.subplots(tile_h,tile_w)

        cnt = 0
        for h in range(tile_h):
            for w in range(tile_w):
                ax[h,w].imshow(gen_imgs[cnt,:,:,0])
                ax[h,w].axis('off')

                cnt += 1

        savepath = os.path.join(savedir, 'dcgan_{:06}.jpg'.format(epoch))
        fig.savefig(savepath, dpi=dpi)

    def each_save(self, fake_img, epoch, tile_h=10, tile_w=10, savedir=None, dpi=500):
        fake_img = fake_img*127.5+127.5

        if fake_img.shape[-1] == 1:
            fake_img = fake_img.reshape(fake_img.shape[0:2])

        fake_img = Image.fromarray(fake_img.astype(np.uint8)).resize((tile_h,tile_w))

        savepath = os.path.join(savedir, 'dcgan_{:06}.jpg'.format(epoch))
        fake_img.save(savepath, dpi=dpi)

    def gan_loss_vis(self, loss_df, savedir=None, dpi=500):
        pass

    def autoencoder_recodec(self, prods, names, savedir=None, dpi=500,
            resize_shape=None, normtype='noramlize'):
        assert type(resize_shape) is tuple, \
                'resize_shape: tuple, ex. (640,390).'

        if normtype == 'normalize':
            prods *= 256.
        elif normtype == 'standard':
            prods = prods*127.5+127.5
        else:
            raise ValueError('normtype: normalize/standard')

        for name, prod in zip(names, prods):
            try:
                gen_img = Image.fromarray(prod.astype(np.uint8)).resize(resize_shape)
                gen_img.save(os.path.join(savedir,'{}_gen.jpg'.format(name)))
            except Exception as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                sys.exit(1)
