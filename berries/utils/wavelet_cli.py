# -*- coding: utf-8 -*-
# @Time   : 3/28/2020 6:27 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : wavelet_cli.py

from dataset.toolwear import Toolwear
from pathlib import Path
import numpy as np
import multiprocessing
import argparse
import itertools
from tqdm import tqdm
from utils.plot_utils import image_folder_to_gif

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cutno", help="Cutting number", type=int)
    parser.add_argument("--wavelet", help="Mother wavelet (default: cmor6-1.5)", type=str)
    parser.add_argument("--plot-func", help="Function to plot. own or lib (default: own)", type=str)
    parser.add_argument("--fpath", help="Result save path (default: wavelet-results)", type=str)
    parser.add_argument("--kind", help="acc or acoustic (default: acc)", type=str)
    parser.add_argument("--subset-min", help="Min time for first subset (default: time.min)", type=float)
    parser.add_argument("--subset-max", help="Max time for last subset (default: time.max)", type=float)
    parser.add_argument("--subset-len", help="Signal length for each subset (default: 1 sec)", type=float)
    parser.add_argument("--subset-stride", help="Stride for each subset (default: --subset-len)", type=float)
    parser.add_argument("--n-core", help="Number of cpu core (default: 1)", type=int)

    args = parser.parse_args()

    print(args)

    n_core = args.n_core or 1
    mother_wavelet = args.wavelet or 'cmor6-1.5'
    plot_func = args.plot_func or 'own'
    fpath = args.fpath or 'wavelet-results'

    vib = Toolwear.batch_read(fpath=Path('D:/YandexDisk/machining/data/raw'),
                              cut_no=args.cutno, kind=args.kind or 'acc', n_cycle=200)

    subset_min = args.subset_min or vib.reading['time'].iloc[0]  # sec
    subset_max = args.subset_max or vib.reading['time'].iloc[-1]  # sec
    subset_len = args.subset_len or 1  # sec
    subset_stride = args.subset_stride or subset_len

    print(f"Subset creation starting: [{subset_min}, {subset_max}) stride:{subset_stride} len:{subset_len}")
    subsets = [vib.reading.iloc[vib.second2index(sec0):vib.second2index(sec1), :]
               for sec0, sec1 in zip(np.arange(subset_min, subset_max - subset_len + subset_stride, subset_stride), # (0, 1.6+e, 0.2)
                                     np.arange(subset_min + subset_len, subset_max + subset_stride, subset_stride))] # (0.4, 2.0+e, 0.2)

    print(f"{len(subsets)} subsets are created.")


    print("Starting wavelet calculations. It can take too long... be patient :)")
    if n_core == 1:
        with tqdm(total=len(subsets), desc='Wavelet Calculations..') as pbar:
            for subset in subsets:
                Toolwear.wavelet(subset, mother_wavelet, plot_func, fpath)
                pbar.update(1)
    else:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=n_core) as executor:
        #     for subset in subsets:
        #         executor.submit(Toolwear.wavelet, subset, mother_wavelet, plot_func, fpath)
        with multiprocessing.Pool(processes=n_core) as pool:
            pool.starmap(Toolwear.wavelet, itertools.product(subsets,
                                                            [mother_wavelet],
                                                            [plot_func],
                                                            [fpath]))

    image_folder_to_gif(f"{args.fpath or 'wavelet-results'}/figures", glob='*.jpg')

