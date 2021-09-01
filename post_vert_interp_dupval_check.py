"""
Check vertically-interpolated data for duplicate values

* Check for duplicate interpolated values in a sequence
    - See if I can replicate this result in a small example?
    - Oxygen should not be the exact same between 1500m and 2000m
    - Once flagging complete, compare interpolated profiles with original profiles

"""

import pandas as pd
import numpy as np
import glob
from os.path import basename
from tqdm import trange

interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\9_vertical_interp\\'

infiles = glob.glob(interp_dir + '*value_vs_depth*.csv')

# Need the seasons in order
infiles.sort()

# Do separate calculations for each season
df_in = pd.read_csv(infiles[0])

# Initialize duplicate value flag
df_in['Replicate_val_flag'] = np.zeros(len(df_in), dtype=int)

prof_start_ind = np.unique(df_in.Profile_number, return_index=True)[1]

# print(prof_start_ind[i])
for i in trange(len(prof_start_ind)):
    # Set profile end index
    if i == len(prof_start_ind) - 1:
        end_ind = len(df_in)
    else:
        end_ind = prof_start_ind[i + 1]

    # Get profile data; np.arange not inclusive of end which we want here
    # Convert dataframe columns to numpy arrays
    indices = np.arange(prof_start_ind[i], end_ind)
    # depths = df_in.loc[indices, 'SL_depth_m']
    # values = df_in.loc[indices, 'SL_value']
    #
    # # Check whether there are duplicate values in the profile
    # # Need INVERSE of np.unique result
    # # Only flag duplicates if they do not include the first measurement?
    # # Flag first occurrances of replicates as well?
    # dupval_flag_loc = np.unique(values, return_index=True)[1]
    # df_in[indices[dupval_flag_loc], 'Replicate_val_flag'] = 0

    # Use pandas Dataframe methods instead of numpy?
    # keep=False to flag first occurrances True as well (so, all duplicates)
    # .duplicated() returns a Series -- must convert to array?
    where_to_flag = df_in.loc[indices].duplicated(subset='SL_value', keep=False)

    # Flag replicates
    df_in.loc[indices[where_to_flag], 'Replicate_val_flag'] = 1
    # continue

out_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\10_replicate_check\\'

outname = basename(infiles[0]).replace('.csv', '_rep_val_check.csv')

df_in.to_csv(out_dir + outname, index=False)

print(len(df_in))
print(len(df_in.loc[df_in.Replicate_val_flag == 1]))
