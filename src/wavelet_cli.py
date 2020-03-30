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
from utils.plot_utils import image_folder_to_gif

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cutno", help="Cutting number", type=int)
    parser.add_argument("--kind", help="acc or acoustic (default: acc)", type=str)
    parser.add_argument("--subset-min", help="Min time for first subset (default: time.min)", type=float)
    parser.add_argument("--subset-max", help="Max time for last subset (default: time.max)", type=float)
    parser.add_argument("--subset-len", help="Signal length for each subset (default: 1 sec)", type=float)
    parser.add_argument("--subset-stride", help="Stride for each subset (default: 1 sec)", type=float)

    args = parser.parse_args()

    print(args)
    vib = Toolwear.batch_read(fpath=Path('D:/YandexDisk/machining/data/raw'),
                              cut_no=args.cutno, kind=args.kind or 'acc', n_cycle=200)

    subset_min = args.subset_min or vib.reading['time'].iloc[0]  # sec
    subset_max = args.subset_max or vib.reading['time'].iloc[-1]  # sec
    subset_len = args.subset_len or 1  # sec
    subset_stride = args.subset_stride or 1


    print(f"Subset creation starting: [{subset_min}, {subset_max}) stride:{subset_stride} len:{subset_len}")
    subsets = [vib.reading.iloc[vib.second2index(sec0):vib.second2index(sec1), :]
               for sec0, sec1 in zip(np.arange(subset_min, subset_max - subset_len + subset_stride, subset_stride), # (0, 1.6+e, 0.2)
                                     np.arange(subset_min + subset_len, subset_max + subset_stride, subset_stride))] # (0.4, 2.0+e, 0.2)

    print(f"{len(subsets)} subsets are created.")
    print("Starting multiprocessed wavelet function. It can take too long... be patient :)")
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(Toolwear.wavelet, subsets)

    image_folder_to_gif('wavelet-results/figures', glob='*.jpg')

