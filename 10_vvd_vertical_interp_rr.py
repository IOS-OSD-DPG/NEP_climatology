"""Notes:
- Use the cleaned value vs depth data that passed the gradient check
- Apply Reiniger-Ross vertical interpolation scheme
    * Use R oce function for interpolation
- Export interpolated values to new CSV file
    * Format will be same as the input value vs depth format
    * The new csv file will be larger so will need to use chunking to read in

oceApprox docs: https://rdrr.io/cran/oce/man/oceApprox.html
"""

# Set R_HOME
# Retrieve from command prompt using:> R RHOME
# import os
# os.environ['R_HOME'] = 'C:\\Users\\HourstonH\\Miniconda3\\envs\\clim38\\lib\\R'

from rpy2 import robjects
from rpy2.robjects.packages import importr  # , isinstalled
import pandas as pd
import numpy as np
from tqdm import trange
# import csv
from clim_helpers import get_standard_levels
# from scipy import interpolate
import glob
from os.path import basename

# R PREPARATION

# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

# print(isinstalled('stats'))

stats = importr('stats')

# import R "oce" package
oce = importr('oce')

# Get the R function we want to use
roceApprox = robjects.r['oceApprox']

# Convert the numpy arrays to rpy2 (R) vectors (LATER)

# END OF R SETUP

# Import standard levels file
# file_sl = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
#           'WOA_Standard_Depths.txt'
file_sl = '/home/hourstonh/Documents/climatology/lu_docs/WOA_MForeman_Standard_Depths.txt'
sl_arr = get_standard_levels(file_sl)

# # Need to add some extra levels that the WOA is missing
# sl_arr = np.array([110, 120, 130, 140, 160, 170, 180, 190, 220, 240, 260,
#                    280, 320, 340, 360, 380])

# Check to make sure that the standard depths are correct
# Length should be 102 for WOA18 levels
print(len(sl_arr))

# Diffs of the WOA standard levels should be monotonous increasing
diffs2 = np.diff(np.diff(sl_arr))

print(np.all(diffs2 >= 0))  # Want True

# Begin processing

# Import value vs depth data
# df_indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#            'value_vs_depth\\9_gradient_check\\'
df_indir = '/home/hourstonh/Documents/climatology/data/value_vs_depth/9_gradient_check/'
df_infile = df_indir + 'Oxy_1991_2020_value_vs_depth_grad_check_done.csv'
# df_infile = 'WOD_PFL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

df_vvd = pd.read_csv(df_infile)

# Profile start indices
# (index of the first measurement of each profile in the vvd df)
prof_start_ind = np.unique(df_vvd.Profile_number, return_index=True)[1]

# Initialize output df
df_out = pd.DataFrame()

# Make list of column names for output df
sl_colnames = list(df_vvd.columns[:-2]) + ['SL_depth_m', 'SL_value']

# Iterate through all of the profiles
# int(np.floor(len(prof_start_ind)/2))
for i in trange(len(prof_start_ind)):  # len(prof_start_ind) 20
    # print(prof_start_ind[i])

    # Set profile end index
    if i == len(prof_start_ind) - 1:
        end_ind = len(df_vvd)
    else:
        end_ind = prof_start_ind[i + 1]

    # Get profile data; np.arange not inclusive of end which we want here
    # Convert dataframe columns to numpy arrays
    indices = np.arange(prof_start_ind[i], end_ind)
    depths = np.array(df_vvd.loc[indices, 'Depth_m'])
    values = np.array(df_vvd.loc[indices, 'Value'])

    # Re-sort the values if the profile is an upcast instead of a downcast
    # In this case, depths would be getting smaller
    if np.all(np.diff(depths) < 0):
        print('Warning: profile number', df_vvd.loc[indices[0], 'Profile_number'],
              'is an upcast')
        # Sort the depths in increasing order
        depths_sort_inds = depths.argsort()
        # Index depths by increasing order
        depths = depths[depths_sort_inds]
        # Index values by the increasing order of the depths
        values = values[depths_sort_inds]

    # Extract the subset of standard levels to use for vertical interpolation
    # The last element in depths is the deepest one
    # Index [0] is the first element of the returned tuple which is an array
    # of the indices
    if depths[0] < 5:
        # If there are data above 5m depth, then the surface value is taken to
        # equal to the shallowest recorded value.
        sl_subsetter = np.where(sl_arr <= depths[-1])[0]
    else:
        sl_subsetter = np.where((sl_arr >= depths[0]) & (sl_arr <= depths[-1]))[0]

    # Skip computations if no standard level matches
    if len(sl_subsetter) == 0:
        # print('Warning: No standard level matches for profile number',
        #       df_vvd.loc[indices[0], 'Profile_number'])
        pass
    elif len(sl_subsetter) > 0:
        # z_out is the standard levels to interpolate to in Python array format
        z_out = sl_arr[sl_subsetter]

        # print(z_out)

        # Convert profile measurements from Python array to R vector
        rdepths = robjects.FloatVector(depths)
        rvalues = robjects.FloatVector(values)

        # Convert standard levels from Python array to R vector
        rz_out = robjects.FloatVector(z_out)

        # Interpolate to standard levels
        # 'unesco' stands for the vertical interpolation method used by the WOA18
        # Reiniger-Ross (1968) interpolation used when 4 points are available
        # Lagrangian interpolation used when only 3 points available
        # See oceApprox() function docs for more details
        rsl_values = roceApprox(rdepths, rvalues, rz_out, 'unesco')

        # Convert result to Python array format
        sl_values = np.array(rsl_values)

        # Update length of profile information to length of interpolated value array
        profile_number = np.repeat(df_vvd.loc[indices[0], 'Profile_number'], len(z_out))
        cruise_number = np.repeat(df_vvd.loc[indices[0], 'Cruise_number'], len(z_out))
        instrument_type = np.repeat(df_vvd.loc[indices[0], 'Instrument_type'], len(z_out))
        date_string = np.repeat(df_vvd.loc[indices[0], 'Date_string'], len(z_out))
        latitude = np.repeat(df_vvd.loc[indices[0], 'Latitude'], len(z_out))
        longitude = np.repeat(df_vvd.loc[indices[0], 'Longitude'], len(z_out))

        # Put these updated values into a dataframe
        df_add = pd.DataFrame(
            data=np.array([profile_number, cruise_number, instrument_type, date_string,
                           latitude, longitude, z_out, sl_values]).transpose(),
            columns=sl_colnames)

        df_out = pd.concat([df_out, df_add])

        # Reset index in-place; do not add old indices as a new column
        df_out.reset_index(drop=True, inplace=True)

    # continue


# Convert dtypes
df_out.loc[:, 'Profile_number'] = df_out.Profile_number.astype('int')
df_out.loc[:, 'SL_depth_m'] = df_out.SL_depth_m.astype('int32')

# Export dataframe to csv file
# df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#             'value_vs_depth\\10_vertical_interp\\'
df_outdir = '/home/hourstonh/Documents/climatology/data/value_vs_depth/10_vertical_interp/'

# 'rr' stands for Reiniger-Ross vertical interpolation
# df_outname = 'Oxy_1991_2020_value_vs_depth_rr.csv'
df_outname = 'Oxy_1991_2020_value_vs_depth_rr_part2.csv'

df_out.to_csv(df_outdir + df_outname, index=False)


# Export list of profiles dropped to a file
prof_drop_arr = np.setdiff1d(df_vvd.Profile_number, df_out.Profile_number)

df_prof_drop = pd.Series(prof_drop_arr, name='Profile_number')
# series_name = 'Oxy_1991_2020_rr_prof_drop.csv'
series_name = 'Oxy_1991_2020_rr_prof_drop_part2.csv'
df_prof_drop.to_csv(df_outdir + series_name, index=False)


# Summary stats
print(len(df_out))
print(len(df_vvd))

# print(df_out.loc[:55, ['SL_depth_m', 'SL_value']])
# print(df_vvd.loc[:55, ['Depth_m', 'Value']])

# Find how many profiles were lost between df_vvd and df_out
num_prof_in = len(np.unique(df_vvd.loc[:, 'Profile_number']))
num_prof_out = len(np.unique(df_out.Profile_number))

print(num_prof_in, num_prof_out, num_prof_in-num_prof_out)


print('done')

""" Version 1
100%|██████████| 29118/29118 [46:30<00:00, 10.43it/s]
print(len(df_out))
835574
print(len(df_vvd))
8255554
num_prof_in = len(np.unique(df_vvd.Profile_number, return_index=True)[1])
num_prof_out = len(np.unique(df_out.Profile_number, return_index=True)[1])
print(num_prof_in, num_prof_out, num_prof_in-num_prof_out)
28485 28019 466

Version 2: More profiles were retained through this run
print(len(df_vvd))
8255554
print(len(df_out))
861891
num_prof_in = len(np.unique(df_vvd.loc[:, 'Profile_number']))
num_prof_out = len(np.unique(df_out.Profile_number))
print(num_prof_in, num_prof_out, num_prof_in-num_prof_out)
28485 28308 177
"""

print(df_out.loc[:, ['SL_depth_m', 'SL_value']])
