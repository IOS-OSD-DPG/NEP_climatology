# Separate vertically interpolated value vs depth file by season

import pandas as pd
import numpy as np
from tqdm import trange

infile = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\10_replicate_check\\' \
         'Oxy_1991_2020_value_vs_depth_rr_rep_val_check.csv'

df_in = pd.read_csv(infile)

df_winter = pd.DataFrame()
df_spring = pd.DataFrame()
df_summer = pd.DataFrame()
df_fall = pd.DataFrame()

# Profile start indices
# (index of the first measurement of each profile in the vvd df)
prof_start_ind = np.unique(df_in.Profile_number, return_index=True)[1]

for i in trange(len(prof_start_ind)):
    # Set profile end index
    if i == len(prof_start_ind) - 1:
        end_ind = len(df_in)
    else:
        end_ind = prof_start_ind[i + 1]

    # Get profile data; np.arange not inclusive of end which we want here
    # Convert dataframe columns to numpy arrays
    indices = np.arange(prof_start_ind[i], end_ind)

    # Extract month number from the date string YYYYMMDDhhmmss
    prof_mth = int(str(df_in.loc[indices[0], 'Date_string'])[4:6])

    if prof_mth in np.arange(1, 4):
        df_winter = pd.concat([df_winter, df_in.loc[indices]])
    elif prof_mth in np.arange(4, 7):
        df_spring = pd.concat([df_spring, df_in.loc[indices]])
    elif prof_mth in np.arange(7, 10):
        df_summer = pd.concat([df_summer, df_in.loc[indices]])
    elif prof_mth in np.arange(10, 13):
        df_fall = pd.concat([df_fall, df_in.loc[indices]])
    else:
        print('Invalid profile month')

# Export dfs

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\10_replicate_check\\by_season\\'

df_winter.to_csv(outdir + 'Oxy_1991_2020_value_vs_depth_rr_1_3.csv', index=False)
df_spring.to_csv(outdir + 'Oxy_1991_2020_value_vs_depth_rr_4_6.csv', index=False)
df_summer.to_csv(outdir + 'Oxy_1991_2020_value_vs_depth_rr_7_9.csv', index=False)
df_fall.to_csv(outdir + 'Oxy_1991_2020_value_vs_depth_rr_10_12.csv', index=False)
