# Add duplicate flags from profile data tables to value vs depth tables

import numpy as np
import pandas as pd
import glob
from xarray import open_dataset
from copy import deepcopy
from tqdm import trange

