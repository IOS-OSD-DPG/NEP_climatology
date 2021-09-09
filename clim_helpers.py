"""Sept 8, 2021
Functions to facilitate working with climatology files
"""

import pandas as pd
from xarray import open_dataset


def date_string_to_datetime(df):
    # df MUST CONTAIN COLUMN TITLED "Date_string"
    # Create a new column for Date_string in pandas datetime format
    df.insert(len(df.columns), 'Time_pd',
              pd.to_datetime(df.Date_string, format='%Y%m%d%H%M%S'))

    return df


def open_by_source(full_path):
    # Open data file based on which data centre it came from
    # IOS and NODC files are netCDF
    # MEDS files are csv
    if full_path.endswith('.nc'):
        data = open_dataset(full_path)
    elif full_path.endswith('.csv'):
        data = pd.read_csv(full_path)
    return data
