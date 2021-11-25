"""Sept 8, 2021
Functions to facilitate working with climatology files
"""

import pandas as pd
from xarray import open_dataset
import csv
import numpy as np
import matplotlib.pyplot as plt
import os


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


def get_standard_levels(fpath_sl):
    # Return array of standard levels from the standard levels text file

    # Initialize list with each element being a row in file_sl
    sl_list = []
    with open(fpath_sl, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            sl_list += row

    # Remove empty elements: '' and ' '
    # Gotta love list comprehension
    sl_list_v2 = [int(x.strip(' ')) for x in sl_list if x not in ['', ' ']]

    # Convert list to array
    sl_arr = np.array(sl_list_v2)
    return sl_arr


def deg2km(dlat):
    # From DIVAnd.jl
    # Mean radius (http://en.wikipedia.org/wiki/Earth_radius) in km
    R = 6371.009

    return dlat * (2 * np.pi * R) / 360


def plot_divand_analysis(output_dir, lon2d, lat2d, var_field, var_cmap, var_name, var_units,
                         lon_obs, lat_obs, depth, yr, szn, nle_val):
    plt.pcolormesh(lon2d, lat2d, var_field, shading='auto', cmap=var_cmap)  # , vmin=150, vmax=400)
    plt.colorbar(label='{} [{}]'.format(var_name, var_units))  # ticks=range(150, 400 + 1, 50)

    # Scatter plot the observation points
    plt.scatter(lon_obs, lat_obs, c='k', s=0.1)
    plt.title('nl = ne = {}'.format(nle_val))

    # Set limits
    plt.xlim((-160., -102.))
    plt.ylim((25., 62.))

    plt_filename = os.path.join(output_dir + "{}_{}m_{}_{}_analysis2d_gebco_nle{}.png".format(
        var_name, depth, yr, szn, nle_val))
    plt.savefig(plt_filename, dpi=400)

    plt.close()
    return

