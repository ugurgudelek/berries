# -*- coding: utf-8 -*-
# @Time   : 3/23/2020 11:45 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : plot_utils.py

from matplotlib import rc
from matplotlib.ticker import MultipleLocator

def image_folder_to_gif(fpath, glob=None):
    import imageio
    from pathlib import Path
    from pygifsicle import optimize
    from tqdm import tqdm
    fpath = Path(fpath)
    filenames = list(fpath.glob(glob or '*.jpg'))
    with imageio.get_writer(fpath/'animated.gif', mode='I') as writer:
        with tqdm(total=len(filenames), desc='Reading images..') as pbar:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                pbar.update(1)
    optimize(source=str(fpath)+'/animated.gif',
             destination=str(fpath)+'/optimized.gif',
             colors=10,
             options=["--verbose"])
def camera_ready_matplotlib_style(plot_func):
    def wrapper(*args, **kwargs):
        ax = plot_func(*args, **kwargs)

        # Font size
        # rc('font', size=28)
        # rc('font', family='serif')
        # rc('axes', labelsize=32)

        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        [t.set_va('center') for t in ax.get_zticklabels()]
        [t.set_ha('left') for t in ax.get_zticklabels()]

        # Background
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Tick Placement
        # ax.xaxis._axinfo['tick']['inward_factor'] = 0
        # ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
        # ax.yaxis._axinfo['tick']['inward_factor'] = 0
        # ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
        # ax.zaxis._axinfo['tick']['inward_factor'] = 0
        # ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
        # ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(5))
        # ax.zaxis.set_major_locator(MultipleLocator(0.01))
        return ax

    return wrapper