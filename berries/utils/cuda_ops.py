# %%
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch as t

import subprocess
from io import StringIO
import pandas as pd


def get_free_gpu():
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

    gpu_stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv,nounits,noheader",
            f"--query-gpu={','.join(QUERY_GPU)}",
        ]
    )
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")), names=QUERY_GPU, index_col=0)
    idx = gpu_df["memory.free"].idxmax()
    print(f"Returning GPU {idx} with {gpu_df.iloc[idx]['memory.free']} free MiB")
    return gpu_df
