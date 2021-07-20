# Create value vs depth tables
# Use exact duplicate flags to remove data
# Retain the inexact duplicate flags but don't apply them

import numpy as np
import pandas as pd
import glob
from xarray import open_dataset

# Import IOS data
IOS_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\IOS_CIOOS\\'
IOS_files = glob.glob(IOS_dir + 'IOS_BOT_Profiles_Oxy*.nc', recursive=False)
IOS_files += glob.glob(IOS_dir + 'IOS_CTD_Profiles_Oxy*.nc', recursive=False)

# Import WOD data
WOD_nocad_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\WOD_extracts\\' \
                'Oxy_WOD_May2021_extracts\\'
WOD_nocad_files = glob.glob(WOD_nocad_dir + 'Oxy*OSD.nc', recursive=False)

# Returns no files since there are no Oxy OSD data
# WOD_cad_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\WOD_extracts\\' \
#               'WOD_July_CDN_nonIOS_extracts\\'
# WOD_cad_files = glob.glob(WOD_cad_dir + 'Oxy*OSD.nc', recursive=False)

# Import MEDS data
MEDS_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\meds_data_extracts\\' \
           'bo_extracts\\'
MEDS_files = glob.glob(MEDS_dir + '.csv', recursive=False)


# Open the flags dataset from the previous step
df_flags_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
                'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_pi.csv'

df_flags = pd.read_csv(df_flags_file)

# Name of output file
df_val_dep_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
                  'duplicates_flagged\\value_vs_depth.csv'

# Initialize value vs depth dataframe
df_val_dep = pd.DataFrame(columns=['Date_string', 'Latitude', 'Longitude', 'Depth_m',
                                   'Value', 'Source_flag', 'Inexact_duplicate_flag'])

# Open the raw data files and apply the flags
