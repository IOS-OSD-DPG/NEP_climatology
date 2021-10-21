import pandas as pd
import glob
from clim_helpers import get_standard_levels
from clim_helpers import date_string_to_datetime
import numpy as np
from tqdm import trange


# Make these files for Lu in ODV

# Find vvd files
indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        'value_vs_depth\\12_stats_check\\'

infiles = glob.glob(indir + '*done.csv')

print(len(infiles))

# Create a list matching infiles that matches each file to its season
szn_abbrev = []
for f in infiles:
    if 'JFM' in f:
        szn_abbrev.append('JFM')
    elif 'AMJ' in f:
        szn_abbrev.append('AMJ')
    elif 'JAS' in f:
        szn_abbrev.append('JAS')
    elif 'OND' in f:
        szn_abbrev.append('OND')

print(szn_abbrev)

# Get standard levels
file_sl = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
          'WOA_Standard_Depths.txt'

sl_arr = get_standard_levels(file_sl)

# Define output directory for standard level files
outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\13_separate_by_sl\\'

# Iterate through the standard levels
for i in trange(len(sl_arr)):
    # Want Date_string, Latitude, Longitude, and SL_value
    # Iterate through the value vs depth tables from the previous
    # processing step
    for f, szn in zip(infiles, szn_abbrev):
        # Get the season months abbreviation
        # szn = f[-7:-4]  # Change indices if file names change!!!

        # Read in vvd file
        vvd = pd.read_csv(f)

        # Create a column containing the year from Date_string
        vvd = date_string_to_datetime(vvd)

        # Subset the observations at standardlevel_array[i]
        sl_subsetter = np.where((vvd.SL_depth_m == sl_arr[i]))[0]

        # Concatenate to the corresponding dataframe
        df_sl = vvd.loc[sl_subsetter, ['Date_string', 'Latitude', 'Longitude',
                                       'SL_value']]

        # Add year column for Lu
        df_sl['Year'] = vvd.loc[sl_subsetter, 'Time_pd'].dt.year

        # Export df_sl as a csv file
        sl_out_name = 'Oxy_{}m_{}.csv'.format(sl_arr[i], szn)

        df_sl.to_csv(outdir + sl_out_name, index=False)
