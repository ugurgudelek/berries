import requests
import os
from pathlib import Path
import pickle
from shutil import unpack_archive



nyc_taxi_raw_path = Path('dataset/nyc_taxi/raw/nyc_taxi.csv')
labeled_data = []
with open(str(nyc_taxi_raw_path),'r') as f:
    for i, line in enumerate(f):
        tokens = [float(token) for token in line.strip().split(',')[1:]]
        tokens.append(1) if 150 < i < 250 or   \
                            5970 < i < 6050 or \
                            8500 < i < 8650 or \
                            8750 < i < 8890 or \
                            10000 < i < 10200 or \
                            14700 < i < 14800 \
                          else tokens.append(0)
        labeled_data.append(tokens)
nyc_taxi_train_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled','train',nyc_taxi_raw_path.name).with_suffix('.pkl')
nyc_taxi_train_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(nyc_taxi_train_path),'wb') as pkl:
    pickle.dump(labeled_data[:13104], pkl)

nyc_taxi_test_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled','test',nyc_taxi_raw_path.name).with_suffix('.pkl')
nyc_taxi_test_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(nyc_taxi_test_path),'wb') as pkl:
    pickle.dump(labeled_data[13104:], pkl)