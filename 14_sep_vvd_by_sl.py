# Separate value vs depth standard level values by standard level
# in preparation for spatial interpolation
# Also separate by season?

# 2 options:
# Create the standard level files and add to them as the standard levels are
# encountered in the vvd files, so all are being added to at the same time
# OR
# Iterate through all the standard level files to create and complete the
# standard level files one at a time

import pandas as pd
import glob
from get_standard_levels import get_standard_levels
from clim_helpers import date_string_to_datetime
import numpy as np

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
         'value_vs_depth\\14_sep_by_sl_and_year\\'

# Iterate through the standard levels
for i in range(len(sl_arr)):
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

        # Iterate through 30-year range: 1991-2020
        for yr in np.arange(1991, 2021):
            # Initialize dataframe for each standard level and year
            df_sl = pd.DataFrame()

            # Subset the observations at standardlevel_array[i]
            sl_subsetter = np.where((vvd.SL_depth_m == sl_arr[i]) &
                                    (vvd.Time_pd.dt.year == yr))[0]

            # Concatenate to the corresponding dataframe
            df_sl = pd.concat([df_sl, vvd.loc[sl_subsetter,
                               ['Date_string', 'Latitude', 'Longitude',
                                'SL_value']]])

            # Export df_sl as a csv file
            sl_out_name = 'Oxy_{}m_{}_{}.csv'.format(sl_arr[i], yr, szn)

            df_sl.to_csv(outdir + sl_out_name, index=False)

# Next take the anomaly of these files for spatial interpolation (analysis)..
