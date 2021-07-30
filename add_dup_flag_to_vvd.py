# Add duplicate flags from profile data tables to value vs depth tables

import numpy as np
import pandas as pd
import glob
from copy import deepcopy
from tqdm import trange

vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\'
vvd_fname = vvd_dir + 'ALL_Oxy_1991_2020_value_vs_depth.csv'

# Find the duplicate flags file
pdt_fname = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
            'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_check2.csv'

# Open value vs depth table
df_vvd = pd.read_csv(vvd_fname)

# Open profile data table containing duplicate flags
df_pdt = pd.read_csv(pdt_fname)

# Initialize new columns for flags in df_vvd
df_vvd['Exact_duplicate_flag'] = np.repeat(False, len(df_vvd))
df_vvd['CTD_BOT_duplicate_flag'] = np.repeat(False, len(df_vvd))
df_vvd['Inxact_duplicate_flag'] = np.repeat(False, len(df_vvd))


# Iterate through the pdt or the vvd table?
for i in trange(len(df_pdt)):
    cruise_pdt = df_pdt.loc[i, 'Cruise_number']
    instrument_pdt = df_pdt.loc[i, 'Instrument_type']
    time_pdt = df_pdt.loc[i, 'Date_string']
    lat_pdt = df_pdt.loc[i, 'Latitude']
    lon_pdt = df_pdt.loc[i, 'Longitude']

    # Find the rows where this combination occurs
    indices = np.where((df_vvd == cruise_pdt) &
                       (df_vvd == instrument_pdt) &
                       (df_vvd == time_pdt) &
                       (df_vvd == lat_pdt) &
                       (df_vvd == lon_pdt))

    if indices == 0:
        print('Warning: No search matches')

    # Populate the duplicate flag columns in the value vs depth dataframe
    df_vvd.loc[indices, 'Exact_duplicate_flag'] = df_pdt.loc[i, 'Exact_duplicate_flag']
    df_vvd.loc[indices, 'CTD_BOT_duplicate_flag'] = df_pdt.loc[i, 'CTD_BOT_duplicate_flag']
    df_vvd.loc[indices, 'Inxact_duplicate_flag'] = df_pdt.loc[i, 'Inxact_duplicate_flag']

# Export the updated dataframe
out_name = vvd_fname.replace('.', '_flag.')

df_vvd.to_csv(out_name, index=False)