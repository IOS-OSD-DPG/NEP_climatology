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
              'value_vs_depth\\3_filtered_for_duplicates\\'

output_dir2 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\4_filtered_for_quality_flag\\'

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
        subsetter = np.where((df.Source_flag.astype(int) == 0) &
                             (df.Depth_flag.astype(int) == 0))[0]
    else:
        # keep source flag == 1, drop the rest
        subsetter = np.where((df.Source_flag.astype(int) == 1) &
                             (df.Depth_flag.astype(int)) == 1)[0]

    print(len(subsetter))
    df_new = deepcopy(df.loc[subsetter])

    df_new.drop(columns=['Depth_flag', 'Source_flag'], inplace=True)

    outname = basename(f).replace('dup_rm', 'qc')
    df_new.to_csv(output_dir2 + outname, index=False)


# Folder 5_filtered_for_nans
# Remove rows with variable == nan? (lots of IOS 1990's data)
nan_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\5_filtered_for_nans\\'

qc_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\4_filtered_for_quality_flag\\'

qc_files = glob.glob(qc_dir + '*.csv')

for f in qc_files:
    print(basename(f))
    df = pd.read_csv(f)
    print('Starting df length:', len(df))
    # Drop all rows that have df.Value == NaN
    df.dropna(axis='index', subset=['Value'], inplace=True)
    print('Ending df length:', len(df))
    # Export df
    outname = basename(f).replace('qc', 'nan_rm')
    df.to_csv(nan_dir + outname, index=False)


# Put all nan vvds into one file?
nan_files = glob.glob(nan_dir + '*.csv')

df_all = pd.DataFrame()

profile_counter = 0
for f in nan_files:
    df_add = pd.read_csv(f)
    if df_all.empty:
        df_add.loc[:, 'Profile_number'] += profile_counter
    else:
        df_add.loc[:, 'Profile_number'] += profile_counter + 1

    df_all = pd.concat([df_all, df_add])
    df_all.reset_index(drop=True, inplace=True)
    profile_counter += df_all.loc[len(df_all) - 1, 'Profile_number'] #index up to len - 1

all_name = 'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv'

df_all.to_csv(nan_dir + all_name, index=False)

