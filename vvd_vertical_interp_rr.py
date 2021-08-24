"""Notes:
- Use the cleaned value vs depth data that passed the gradient check
- Apply Reiniger-Ross vertical interpolation scheme
    * Use R oce function for interpolation
- Export interpolated values to new CSV file
    * Format will be same as the input value vs depth format
    * The new csv file will be larger so will need to use chunking to read in

"""

from rpy2 import robjects
from rpy2.robjects.packages import importr, isinstalled
import pandas as pd
import numpy as np
from tqdm import trange
import csv


# R PREPARATION

# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

print(isinstalled('stats'))

stats = importr('stats')

# import R "oce" package
oce = importr('oce')

# Get the R function we want to use
roceApprox = robjects.r['oceApprox']

# Convert the numpy arrays to rpy2 (R) vectors (LATER)

# END OF R SETUP

# Import standard levels file
file_sl = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
          'WOA_Standard_Depths.txt'

# Index column is the column of level number of each standard level
df_sl = pd.read_csv(file_sl, sep=', ', engine='python')

sl_list = []
with open(file_sl, 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        sl_list += row

# Remove empty elements: '' and ' '
# Gotta love list comprehension
sl_list_v2 = [int(x.strip(' ')) for x in sl_list if x not in ['', ' ']]

# Convert list to array
sl_arr = np.array(sl_list_v2)

# Check to make sure that the standard depths are correct

# Length should be 102 for WOA18 levels
print(len(sl_arr))

# Diffs of the standard levels should be monotonous increasing
diffs2 = np.diff(np.diff(sl_arr))

print(np.all(diffs2 >= 0))  # Want True

# Begin processing

# Import value vs depth data
df_indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
           'value_vs_depth\\8_gradient_check\\'

df_infile = 'ALL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

df_vvd = pd.read_csv(df_indir + df_infile)

# Profile start indices
# (index of the first measurement of each profile in the vvd df)
prof_start_ind = np.unique(df_vvd.Profile_number, return_index=True)[1]

# Initialize output df
df_out = pd.DataFrame()

# Make list of column names for output df
sl_colnames = list(df_vvd.columns[:-2]) + ['SL_depth_m', 'SL_value']

# Iterate through all of the profiles
for i in trange(int(np.floor(len(prof_start_ind)/2)),
                len(prof_start_ind)):  # len(prof_start_ind) 20
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

    # Check that depths are monotonously increasing
    if not np.all(np.diff(depths)):
        print('Index', prof_start_ind[i])
        print('Warning: Depths not monotonously increasing')

    # Extract the subset of standard levels to use for vertical interpolation
    # The last element in depths is the deepest one
    # Index [0] is the first element of the returned tuple which is an array
    # of the indices
    sl_subsetter = np.where((sl_arr >= depths[0]) & (sl_arr <= depths[-1]))[0]
    # z_out is the standard levels to interpolate to in Python array format
    z_out = sl_arr[sl_subsetter]

    # print(z_out)

    # Convert profile measurements from Python array to R vector
    rdepths = robjects.FloatVector(depths)
    rvalues = robjects.FloatVector(values)

    # Convert standard levels from Python array to R vector
    rz_out = robjects.FloatVector(z_out)

    # Interpolate to standard levels
    # 'rr' stands for the Reiniger-Ross interpolation method used by the WOA18
    rsl_values = roceApprox(rdepths, rvalues, rz_out, 'rr')

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

# Summary stats
print(len(df_out))

print(df_out.loc[:10, ['SL_depth_m', 'SL_value']])

# Export dataframe to csv file
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\9_vertical_interp\\'

# 'rr' stands for Reiniger-Ross vertical interpolation
df_outname = 'Oxy_1991_2020_value_vs_depth_rr_pt2.csv'

df_out.to_csv(df_outdir + df_outname, index=False)