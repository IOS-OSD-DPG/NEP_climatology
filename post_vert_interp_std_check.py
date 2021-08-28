"""Post-vertical interpolation checks

* Stats checks
    - Separate 5-standard-deviation checks for each season
    - Separate 5-standard-deviation checks for above and including 50m
    - Or, separate 5-standard-deviation checks for each standard level?
    - Also need to decide whether to use 5-degree squares or not
"""

import pandas as pd
import numpy as np
import glob
from os.path import basename


def rr_std_check(df_in):
    # Initialize column for standard deviation flags
    df_in['STD_flag'] = np.zeros(len(df_in), dtype='int32')

    # Calculate the standard deviation of all values above and including 50m
    subsetter_le_50m = np.where(df_in.SL_depth_m <= 50)[0]

    mean_le_50m = np.mean(df_in.loc[subsetter_le_50m, 'SL_value'])
    std_le_50m = np.std(df_in.loc[subsetter_le_50m, 'SL_value'])

    # Calcuate the standard deviation of all values below 50m
    subsetter_gt_50m = np.where(df_in.SL_depth_m > 50)[0]

    mean_gt_50m = np.mean(df_in.loc[subsetter_gt_50m, 'SL_value'])
    std_gt_50m = np.std(df_in.loc[subsetter_gt_50m, 'SL_value'])

    std_flag_loc = np.where(
        (df_in.loc[subsetter_le_50m, 'SL_value'] > mean_le_50m + 5 * std_le_50m) |
        (df_in.loc[subsetter_gt_50m, 'SL_value'] > mean_gt_50m + 5 * std_gt_50m))[0]

    df_in.loc[std_flag_loc, 'STD_flag'] = 1

    # Repeat std check once on the data that haven't already been flagged

    # Calculate the standard deviation of all values above and including 50m
    subsetter_le_50m_2 = np.where((df_in.STD_check == 0) &
                                  (df_in.SL_depth_m <= 50))[0]

    mean_le_50m_2 = np.mean(df_in.loc[subsetter_le_50m_2, 'SL_value'])
    std_le_50m_2 = np.std(df_in.loc[subsetter_le_50m_2, 'SL_value'])

    # Calcuate the standard deviation of all values below 50m
    subsetter_gt_50m_2 = np.where((df_in.STD_check == 0) &
                                  (df_in.SL_depth_m > 50))[0]

    mean_gt_50m_2 = np.mean(df_in.loc[subsetter_gt_50m_2, 'SL_value'])
    std_gt_50m_2 = np.std(df_in.loc[subsetter_gt_50m_2, 'SL_value'])

    std_flag_loc_2 = np.where(
        (df_in.loc[subsetter_le_50m_2, 'SL_value'] > mean_le_50m_2 + 5 * std_le_50m_2) |
        (df_in.loc[subsetter_gt_50m_2, 'SL_value'] > mean_gt_50m_2 + 5 * std_gt_50m_2))[0]

    df_in.loc[std_flag_loc_2, 'STD_flag'] = 2

    # Export the updated df
    outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\9_vertical_interp\\by_season\\10_vertical_interp_check\\'

    # Shorten file name...
    outname = basename(infiles[0]).replace('_value_vs_depth', '')
    outname = outname.replace('.csv', 'std_check.csv')

    df_in.to_csv(outdir + outname, index=False)

    # Export the updated df with the flagged values removed

    # Remove flagged data
    df_out = df_in.loc[df_in['STD_flag'] == 0]

    # Drop the column containing the flags
    df_out.drop(columns='STD_flag', inplace=True)

    outname2 = outname.replace('.csv', '_done.csv')

    df_out.to_csv(outdir + outname2, index=False)

    return outdir + outname, outdir + outname2


in_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\9_vertical_interp\\by_season\\'

infiles = glob.glob(in_dir + '*.csv')

# Need the seasons in order
infiles.sort()

# Do separate calculations for each season
df = pd.read_csv(infiles[0])

# STANDARD DEVIATION CHECKS
path_list = rr_std_check(df)

