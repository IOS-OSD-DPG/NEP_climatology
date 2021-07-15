import glob
import pandas as pd
import numpy as np
from tqdm import trange


def similar_lat(r1, r2, offset=0.2):
    return abs(r1['Latitude'] - r2['Latitude']) < offset


def similar_lon(r1, r2, offset=0.2):
    return abs(r1['Longitude'] - r2['Longitude']) < offset


def similar_time(r1, r2, offset=1, offset_unit='h'):
    # Start with offset of 1 hour
    # Convert time string to pandas datetime
    # offset_unit: 'D', 'h', 'm', 'sec', 'ns'
    t1 = pd.to_datetime(r1['Date_string'])
    t2 = pd.to_datetime(r2['Date_string'])
    return abs(t1 - t2) < pd.Timedelta(offset, unit=offset_unit)


extract_folder = '/home/hourstonh/Documents/climatology/data_extracts/'
extracts = glob.glob(extract_folder + '*.csv', recursive=False)

colnames = ["Source_data_file_name", "Institute", "Cruise_number",
            "Instrument_type", "Date_string", "Latitude",
            "Longitude", "Quality_control_flag"]

df_all = pd.DataFrame(columns=colnames)

for f in extracts:
    df_add = pd.read_csv(f)
    df_all = pd.concat([df_all, df_add], ignore_index=True)
    
# Find duplicates
# Use the NODC flagging scheme to differentiate between exact and inexact duplicates?

# 1 2 3 4 5 6 7 8 9
# "accepted annual_sd_out density_inversion cruise seasonal_sd_out monthly_sd_out
# annual+seasonal_sd_out anomaly_or_annual+monthly_sd_out seasonal+monthly_sd_out
# annual+seasonal+monthly_sd_out" ;

# 1: exact duplicate 2: CTD/BOT duplicate 3: inexact duplicate 4: ....

df_all = df_all.drop(columns='Unnamed: 0')

df_copy = df_all.copy()

df_copy['Exact_duplicate_row'] = df_copy[
    ['Instrument_type', 'Date_string', 'Latitude', 'Longitude']].duplicated()

# How to specify to keep CTD data over BOT data regardless of order in the df?
df_copy['CTD_BOT_duplicate_row'] = df_copy[['Date_string', 'Latitude', 'Longitude']].duplicated()

# Check for inexact duplicates
df_copy['Inexact_duplicate_row'] = np.repeat(False, len(df_copy))
# Iterate through dataframe
for i in trange(len(df_copy)):
    for j in range(i, len(df_copy)):
        row1 = df_copy.iloc[i]
        row2 = df_copy.iloc[j]
        if similar_lat(row1, row2) and similar_lon(row1, row2) and similar_time(row1, row2):
            # Flag row2
            df_copy.loc[j, 'Inexact_duplicate_row'] = True

#################
mask = df_all[['Instrument_type', 'Date_string', 'Latitude', 'Longitude']].duplicated()
mask = df_all['Date_string'].astype(str).str.startswith('199')

df_all['Duplicate_row'] = np.zeros(len(df_all))
df_all['Duplicate_row'].loc[mask, 'Duplicate_row'] = 1
