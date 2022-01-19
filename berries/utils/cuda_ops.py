# %%
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch as t

import subprocess
from io import StringIO
import pandas as pd


def get_free_gpu(n=1, by="power.draw"):
    QUERY_GPU = [
        "index",
        "gpu_name",
        "memory.used",
        "memory.free",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
    ]

    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv,nounits,noheader", f"--query-gpu={','.join(QUERY_GPU)}",])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")), names=QUERY_GPU, index_col=0)
    gpu_df = gpu_df[:4]

    gpu_df["id"] = gpu_df.index

    gpu_df = gpu_df.sort_values(by)

    gpu_ids = list(gpu_df["id"].values[:n])

    print(gpu_df)
    # print(f"Returning GPU {gpu_ids} with {gpu_df.loc[gpu_df['id'] in gpu_ids, by]} {by}")
    return gpu_ids
