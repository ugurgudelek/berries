"""Image construction"""

from utils import Bucket,normalize_column_based
import numpy as np


class ImageEngine:
    def __init__(self, split_period=28, normalize=True):
        self.split_period = split_period
        self.normalize = normalize
        self.image_bucket = Bucket(size=self.split_period)

        self.image_container = []

    def feed(self, row, old_close=None):
        """row should be a dict and should have 'date' and 'data' keys
        row = dict('date','data')
        """
        data = row['data']
        date = row['date']

        self.image_container.append(data)

        self.image_bucket.put(data=data)

        if self.image_bucket.full():
            chunk = self.image_bucket.get_all_bucket()
            if self.normalize:
                chunk = normalize_column_based(image=chunk)  # normalize image
            chunk = np.array(chunk).flatten()
            return chunk
