# Filter the value vs depth tables to only include good non-duplicate data

import pandas as pd
import glob
from os.path import basename
import numpy as np
from xarray import open_dataset
from copy import deepcopy


vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\added_dup_flags\\exact_duplicate_double_check\\'

vvd_files = glob.glob(vvd_dir + '*.csv')
vvd_files.sort()

output_dir1 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\filtered_for_duplicates\\'

output_dir2 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\filtered_for_quality_flag\\'

for f in vvd_files:
    print(basename(f))
    df = pd.read_csv(f)

    # Remove by duplicate flag
    subsetter = np.where((df.Exact_duplicate_flag == True) |
                         (df.CTD_BOT_duplicate_flag == True) |
                         (df.Inexact_duplicate_flag == True))[0]

    print(len(subsetter))

    df.drop(index=subsetter, inplace=True)

    df.drop(columns=['Exact_duplicate_flag', 'CTD_BOT_duplicate_flag',
                     'Inexact_duplicate_flag'], inplace=True)

    # Save df
    outname = basename(f).replace('_dup', '_dup_rm')
    df.to_csv(output_dir1 + outname, index=False)


# Now for the qc flags

# IOS data ==1 means good quality

# NODC data ==0 means good quality; ==1 means range_out, so bad quality

# MEDS data: 1=data is good, 3=suspicious, 4=bad

nodc_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
            'WOD_extracts\\Oxy_WOD_May2021_extracts\\Oxy_1991_2020_AMJ_OSD.nc'

nc = open_dataset(nodc_file)

files = glob.glob(output_dir1 + '*.csv')

for f in files:
    print(basename(f))
    df = pd.read_csv(f)

    # Drop by quality flag
    if 'WOD' in basename(f):
        # keep source flag == 0, drop the rest
        subsetter = np.where(df.Source_flag.astype(int) == 0)[0]
    else:
        # keep source flag == 1, drop the rest
        subsetter = np.where(df.Source_flag.astype(int) == 1)[0]

    print(len(subsetter))
    df_new = deepcopy(df.loc[subsetter])

    df_new.drop(columns='Source_flag', inplace=True)

    outname = basename(f).replace('dup_rm', 'qc')
    df_new.to_csv(output_dir2 + outname, index=False)

