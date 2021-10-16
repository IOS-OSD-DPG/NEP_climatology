"""Sept 8, 2021
Functions to facilitate working with climatology files
"""

import pandas as pd
from xarray import open_dataset


def concat_vvd_files(flist, outdir, dfname):
    df_all = pd.DataFrame()

    profile_counter = 0
    for f in flist:
        df_add = pd.read_csv(f)
        if df_all.empty:
            df_add.loc[:, 'Profile_number'] += profile_counter
        else:
            df_add.loc[:, 'Profile_number'] += profile_counter + 1

        df_all = pd.concat([df_all, df_add])
        df_all.reset_index(drop=True, inplace=True)
        profile_counter += df_all.loc[len(df_all) - 1, 'Profile_number']  # index up to len - 1

    # all_name = 'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv'
    all_name = outdir + dfname

    df_all.to_csv(all_name, index=False)

    return all_name


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


def vvd_apply_value_flag(df, flag_name):
    # Apply flag and generate new copy of df
    # Flag=0 means data passed the check so want to keep that

    df = df.loc[df[flag_name] == 0]

    df_return = df.drop(columns=flag_name)

    return df_return
