from nptdms import TdmsFile
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

os.makedirs('./csv', exist_ok=True)
for tdms_path in tqdm(Path('.').glob('*.tdms')):
    TdmsFile(tdms_path).as_dataframe().to_csv(f'./csv/{tdms_path.stem}.csv', index=None)
