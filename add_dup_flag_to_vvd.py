# Add duplicate flags from profile data tables to value vs depth tables

import numpy as np
import pandas as pd
from tqdm import trange
import glob


def vvd_add_dup_flags_v2(df_vvd, df_pdt):
    # VERSION 2
    # Initialize new columns for flags in df_vvd
    # Use zeros and ones instead of boolean True/False for ease?
    # np.zeros(len(df_vvd), dtype=int)
    df_vvd['Exact_duplicate_flag'] = np.repeat(False, len(df_vvd))
    df_vvd['CTD_BOT_duplicate_flag'] = np.repeat(False, len(df_vvd))
    df_vvd['Inxact_duplicate_flag'] = np.repeat(False, len(df_vvd))

    # Initialize column to count how many times each row is updated
    # during the iteration -- it should just be once
    # How to deal with exact duplicates though?
    df_vvd['Number_of_rewrites'] = np.zeros(len(df_vvd), dtype=int)

    # Get the starting index of each profile in the value vs depth df
    unique = np.unique(df_vvd.Profile_number, return_index=True)[1]

    # Iterate through all profiles of the value vs depth dataframe
    for i in trange(len(unique)):
        # Get the number of measurements in the profile
        if i == len(unique):
            end_of_prof = len(df_vvd.Profile_number)
        else:
            end_of_prof = unique[i + 1]
        prof_len = end_of_prof - unique[i]

        # Select information to use for profile matching
        # between the vvd df and the pdt df
        cruise_vvd = df_vvd.loc[unique[i], 'Cruise_number']
        instrument_vvd = df_vvd.loc[unique[i], 'Instrument_type']
        time_vvd = df_vvd.loc[unique[i], 'Date_string'].astype(str)
        lat_vvd = df_vvd.loc[unique[i], 'Latitude']
        lon_vvd = df_vvd.loc[unique[i], 'Longitude']

        print(cruise_vvd, instrument_vvd, time_vvd, lat_vvd, lon_vvd)

        print(np.where(pdt_df.Cruise_number == cruise_vvd)[0])
        print(np.where(pdt_df.Instrument_type == instrument_vvd)[0])
        print(np.where(pdt_df.Date_string == time_vvd)[0])
        # Test for inexact floats not equality, otherwise I get nothing
        # print(np.where(pdt_df.Latitude == lat_vvd)[0])
        # print(np.where(pdt_df.Longitude == lon_vvd)[0])
        print(np.where(abs(pdt_df.Latitude - lat_vvd) < 1e-5)[0])
        print(np.where(abs(pdt_df.Longitude - lon_vvd) < 1e-5)[0])

        # Find the rows where this combination occurs
        indices = np.where((df_pdt.Cruise_number == cruise_vvd) &
                           (df_pdt.Instrument_type == instrument_vvd) &
                           (df_pdt.Date_string == time_vvd) &
                           (abs(pdt_df.Latitude - lat_vvd) < 1e-5) &
                           (abs(pdt_df.Longitude - lon_vvd) < 1e-5))[0]

        if len(indices) == 0:
            print('Warning: No rows matching search in pdt')
        elif len(indices) > 1:
            print('Warning: More than one row match returned from pdt')

        # Index the pdt
        # Populate the duplicate flag columns in the value vs depth dataframe
        df_vvd.loc[unique[i]: prof_len, 'Exact_duplicate_flag'
                   ] = df_pdt.loc[indices[0], 'Exact_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: prof_len, 'CTD_BOT_duplicate_flag'
                   ] = df_pdt.loc[indices[0], 'CTD_BOT_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: prof_len, 'Inxact_duplicate_flag'
                   ] = df_pdt.loc[indices[0], 'Inexact_duplicate_check2'].astype(bool)

        # Remove the selected row from the pdt
        df_pdt = df_pdt.drop(index=indices[0])

    return df_vvd


def vvd_add_dup_flags(df_vvd, df_pdt):
    # Initialize new columns for flags in df_vvd
    # Use zeros and ones instead of boolean True/False for ease?
    # np.zeros(len(df_vvd), dtype=int)
    df_vvd['Exact_duplicate_flag'] = np.repeat(False, len(df_vvd))
    df_vvd['CTD_BOT_duplicate_flag'] = np.repeat(False, len(df_vvd))
    df_vvd['Inxact_duplicate_flag'] = np.repeat(False, len(df_vvd))

    # Initialize column to count how many times each row is updated
    # during the iteration -- it should just be once
    # How to deal with exact duplicates though?
    df_vvd['Number_of_rewrites'] = np.zeros(len(df_vvd), dtype=int)

    # Get the starting index of each profile in the value vs depth df
    unique = np.unique(df_vvd.Profile_number, return_index=True)[1]

    # Iterate through all profiles of the value vs depth dataframe
    for i in trange(len(unique)):
        # Select information to use for profile matching
        # between the vvd df and the pdt df
        cruise_vvd = df_vvd.loc[unique[i], 'Cruise_number']
        instrument_vvd = df_vvd.loc[unique[i], 'Instrument_type']
        time_vvd = df_vvd.loc[unique[i], 'Date_string'].astype(str)
        lat_vvd = df_vvd.loc[unique[i], 'Latitude']
        lon_vvd = df_vvd.loc[unique[i], 'Longitude']

        print(cruise_vvd, instrument_vvd, time_vvd, lat_vvd, lon_vvd)

        print(np.where(pdt_df.Cruise_number == cruise_vvd)[0])
        print(np.where(pdt_df.Instrument_type == instrument_vvd)[0])
        print(np.where(pdt_df.Date_string == time_vvd)[0])
        # Test for inexact floats not equality, otherwise I get nothing
        # print(np.where(pdt_df.Latitude == lat_vvd)[0])
        # print(np.where(pdt_df.Longitude == lon_vvd)[0])
        print(np.where(abs(pdt_df.Latitude - lat_vvd) < 1e-5)[0])
        print(np.where(abs(pdt_df.Longitude - lon_vvd) < 1e-5)[0])

        # Find the rows where this combination occurs
        indices = np.where((df_pdt.Cruise_number == cruise_vvd) &
                           (df_pdt.Instrument_type == instrument_vvd) &
                           (df_pdt.Date_string == time_vvd) &
                           (abs(pdt_df.Latitude - lat_vvd) < 1e-5) &
                           (abs(pdt_df.Longitude - lon_vvd) < 1e-5))[0]

        if indices == 0:
            print('Warning: No search matches')
        elif indices > 1:
            print('Warning: More than one row returned')

        # Index the pdt
        # Populate the duplicate flag columns in the value vs depth dataframe
        df_vvd.loc[unique[i]: unique[i + 1], 'Exact_duplicate_flag'
                   ] = df_pdt.loc[indices[0], 'Exact_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: unique[i + 1], 'CTD_BOT_duplicate_flag'
                   ] = df_pdt.loc[indices[0], 'CTD_BOT_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: unique[i + 1], 'Inxact_duplicate_flag'
                   ] = df_pdt.loc[indices[0], 'Inexact_duplicate_check2'].astype(bool)

    return df_vvd


vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\'
# vvd_fname = vvd_dir + 'ALL_Oxy_1991_2020_value_vs_depth.csv'

vvd_list = glob.glob(vvd_dir + '*0.csv')
vvd_list.sort()

# Find the duplicate flags file
pdt_fname = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
            'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_check2.csv'

# Open profile data table containing duplicate flags
pdt_df = pd.read_csv(pdt_fname)
# Drop rows that contain any nans/blank entries
pdt_df = pdt_df.dropna(axis='index', how='any')
# Convert date_string back to string format from float format ugh
pdt_df['Date_string'] = list(map(lambda x: str(x)[:-2], pdt_df['Date_string']))


# for f in vvd_list:

# Call the function
df_in = pd.read_csv(vvd_list[0])
updated_df = vvd_add_dup_flags_v2(df_in, pdt_df)

# Remove temporary columns
updated_df.drop(columns='Instrument_type')
# Export the updated dataframe
out_name = vvd_list[0].replace('.', '_flag.')
updated_df.to_csv(out_name, index=False)


# Approach #2
# Try iterating through the separate vvd dfs
meds_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
            'MEDS_BOT_Oxy_1991_1995_value_vs_depth_0.csv'

meds_df = pd.read_csv(meds_file)
meds_df['Date_string'] = meds_df['Date_string'].astype(str)

# Check for depth inversions to differentiate between profiles?
meds_df['Profile_number'] = np.zeros(len(meds_df), dtype=int)
for i in range(1, len(meds_df)):
    if meds_df.loc[i, 'Depth_m'] < meds_df.loc[i-1, 'Depth_m']:
        meds_df.loc[i, 'Profile_number'] = meds_df.loc[i-1, 'Profile_number'] + 1
    else:
        meds_df.loc[i, 'Profile_number'] = meds_df.loc[i-1, 'Profile_number']

profile_indices = np.unique(meds_df.Profile_number, return_index=True)[1]


# Check to see that the flagging was correct
print(max(pdt_df.loc[:, 'Number_of_rewrites']))

# Remove transitory columns
# df_vvd.drop(columns='Number_of_rewrites')


