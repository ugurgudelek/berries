__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from pathlib import Path
import os
from functools import reduce


class GenericDataset:
    """

    """

    def _check_exists(self, *args):
        return reduce(lambda a, b: os.path.exists(a) and
                                   os.path.exists(b), args)
