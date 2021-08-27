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

interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\9_vertical_interp\\by_season\\'

infiles = glob.glob(interp_dir + '*.csv')

# Need the seasons in order
infiles.sort()

# Do separate calculations for each season
df_in = pd.read_csv(infiles[0])

# Initialize duplicate value flag
df_in['Dup_val_flag'] = np.zeros(len(df_in), dtype=int)

prof_start_ind = np.unique(df_in.Profile_number, return_index=True)[1]

# print(prof_start_ind[i])
for i in range(len(prof_start_ind)):
    # Set profile end index
    if i == len(prof_start_ind) - 1:
        end_ind = len(df_in)
    else:
        end_ind = prof_start_ind[i + 1]

    # Get profile data; np.arange not inclusive of end which we want here
    # Convert dataframe columns to numpy arrays
    indices = np.arange(prof_start_ind[i], end_ind)
    values = df_in.loc[indices, 'Value']

    # Check whether there are duplicate values in the profile
    # Need INVERSE of np.unique result
    dupval_flag_loc = np.unique(values, return_index=True)[1]
    df_in[indices[dupval_flag_loc], 'Dup_val_flag'] = 0

    # continue

out_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\10_vertical_interp_check\\'

outname = basename(infiles[0]).replace('.csv', '_dupval_check.csv')

df_in.to_csv(out_dir + outname, index=False)

# Remove by flag

# Remove flagged data
df_out = df_in.loc[df_in['Dup_val_flag'] == 0]

# Drop the column containing the flags
df_out.drop(columns='Dup_val_flag', inplace=True)

outname2 = outname.replace('.csv', '_done.csv')

df_out.to_csv(out_dir + outname2, index=False)
