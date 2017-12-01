"""Image construction"""

from utils import Bucket,normalize_column_based
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

class ImageEngine:
    def __init__(self, stock_names, split_period=28, normalize=True):
        self.stock_names = stock_names
        self.split_period = split_period
        self.normalize = normalize
        self.image_buckets = defaultdict(Bucket).fromkeys(self.stock_names)
        for key, item in self.image_buckets.items():
            self.image_buckets[key] = Bucket(size=split_period)

        self.image_container = defaultdict(list).fromkeys(self.stock_names)
        for key, item in self.image_container.items():
            self.image_container[key] = []

    def save_instance(self, filepath, run_number):
        filename = filepath+'/{}_image_engine.pkl'.format(run_number)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def feed(self, row):
        """row should be a dict and should have 'stock_name','date' and 'data' keys
        row = dict('date','data','stock_name')
        """
        data = row['data']
        date = row['date']
        stock_name = row['stock_name']

        image_bucket = self.image_buckets[stock_name]
        image_bucket.put(data=data)

        if image_bucket.full():
            image = pd.DataFrame(image_bucket.get_all_bucket())
            if self.normalize:
                # normalization for image.
                image = (image - image.mean()) / image.std()
                # chunk = normalize_column_based(image=chunk)  # normalize image

            image = image.as_matrix().flatten()
            self.image_container[stock_name].append(image)
            return image

class Image:
    pass
