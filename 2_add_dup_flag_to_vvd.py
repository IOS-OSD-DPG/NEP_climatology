# Add duplicate flags from profile data tables to value vs depth tables

import numpy as np
import pandas as pd
from tqdm import trange
import glob
from os.path import basename


def vvd_add_dup_flags(df_vvd, df_pdt, verbose=False):
    # VERSION 2
    # Initialize new columns for flags in df_vvd
    # Use zeros and ones instead of boolean True/False for ease?
    # np.zeros(len(df_vvd), dtype=int)
    df_vvd['Exact_duplicate_flag'] = np.repeat(False, len(df_vvd))
    df_vvd['CTD_BOT_duplicate_flag'] = np.repeat(False, len(df_vvd))
    df_vvd['Inexact_duplicate_flag'] = np.repeat(False, len(df_vvd))

    # Get the starting index of each profile in the value vs depth df
    # Index the second element, which is an array of the indices
    unique = np.unique(df_vvd.Profile_number, return_index=True)[1]

    # Iterate through all profiles of the value vs depth dataframe
    for i in trange(len(unique)):  #10 for testing small subset # len(unique)
        # Get the number of measurements in the profile
        if i == len(unique) - 1:
            # If the last profile in the df, index to the end of the df
            end_of_prof = None
        else:
            end_of_prof = unique[i + 1]
        # prof_len = end_of_prof - unique[i]

        # Select information to use for profile matching
        # between the vvd df and the pdt df
        cruise_vvd = df_vvd.loc[unique[i], 'Cruise_number']
        instrument_vvd = df_vvd.loc[unique[i], 'Instrument_type']
        time_vvd = df_vvd.loc[unique[i], 'Date_string'].astype(str)
        lat_vvd = df_vvd.loc[unique[i], 'Latitude']
        lon_vvd = df_vvd.loc[unique[i], 'Longitude']

        if verbose:
            print(cruise_vvd, instrument_vvd, time_vvd, lat_vvd, lon_vvd)

            # print(np.where(df_pdt.Cruise_number == cruise_vvd)[0])
            # print(np.where(df_pdt.Instrument_type == instrument_vvd)[0])
            # print(np.where(df_pdt.Date_string == time_vvd)[0])
            # # Test for inexact floats not equality, otherwise I get nothing
            # # print(np.where(pdt_df.Latitude == lat_vvd)[0])
            # # print(np.where(pdt_df.Longitude == lon_vvd)[0])
            # print(np.where(abs(df_pdt.Latitude - lat_vvd) < 1e-5)[0])
            # print(np.where(abs(df_pdt.Longitude - lon_vvd) < 1e-5)[0])

        # Find the rows where this combination occurs
        indices_pdt = np.where((df_pdt.Cruise_number == cruise_vvd) &
                               (df_pdt.Instrument_type == instrument_vvd) &
                               (df_pdt.Date_string == time_vvd) &
                               (abs(df_pdt.Latitude - lat_vvd) < 1e-5) &
                               (abs(df_pdt.Longitude - lon_vvd) < 1e-5))[0]

        if verbose:
            if len(indices_pdt) == 0:
                print('Warning: No rows matching search in pdt')
            elif len(indices_pdt) > 1:
                print('Warning: More than one row match returned from pdt')
            elif len(indices_pdt) == 1:
                print('Row match found')

            print(unique[i], indices_pdt[0])

        # Index the pdt
        # Populate the duplicate flag columns in the value vs depth dataframe
        df_vvd.loc[unique[i]: end_of_prof, 'Exact_duplicate_flag'
                   ] = df_pdt.loc[indices_pdt[0], 'Exact_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: end_of_prof, 'CTD_BOT_duplicate_flag'
                   ] = df_pdt.loc[indices_pdt[0], 'CTD_BOT_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: end_of_prof, 'Inexact_duplicate_flag'
                   ] = df_pdt.loc[indices_pdt[0], 'Inexact_duplicate_check2'].astype(bool)

        df_pdt.loc[indices_pdt[0], 'Number_of_uses'] += 1

        # Remove the selected row from the pdt
        # inplace=True instead of making a deep copy
        # df_pdt.drop(index=indices_pdt[0], inplace=True)

        # MAY NOT BE NECESSARY TO FIX INDEXING ERROR
        # Reset the index of the dataframe in-place
        # drop=True so that the old index isn't added as a new column
        # df_pdt.reset_index(drop=True, inplace=True)

    # if verbose:
    #   print('max PDT row number of uses:', max(df_pdt['Number_of_uses']))

    if max(df_pdt['Number_of_uses']) > 1:
        print('Warning: max PDT row number of uses,',
              max(df_pdt['Number_of_uses']), 'exceeds 1')

    # Return modified vvd and pdt
    return df_vvd, df_pdt


def prep_pdt_v2(pdt_fname):
    # Open profile data table containing duplicate flags
    pdt_df = pd.read_csv(pdt_fname)
    # Drop the Original_row_index column
    pdt_df.drop(columns='Original_row_index', inplace=True)

    # # Drop rows that contain all nans/blank entries
    # pdt_na = pdt_df.dropna(axis='index', how='all')
    # Reindex rows; use drop=True to avoid the old index being added as a column
    pdt_df.reset_index(drop=True, inplace=True)

    # Convert date_string back to string format from float format ugh
    pdt_df['Date_string'] = list(map(lambda x: str(x)[:-2], pdt_df['Date_string']))

    # Fix NODC cruise numbers in PDT that are like "b'XXXXXXXX'"
    pdt_df['Cruise_number'] = list(map(lambda x: str(x).strip("b'"), pdt_df['Cruise_number']))
    # print(pdt_df.loc[:5, 'Cruise_number'])

    # Initialize column to count how many times each row is used
    # during the iteration -- it should just be once
    # How to deal with exact duplicates though?
    pdt_df['Number_of_uses'] = np.zeros(len(pdt_df), dtype=int)

    return pdt_df


# Value vs depth table folder
vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\1_original\\'
# vvd_fname = vvd_dir + 'ALL_Oxy_1991_2020_value_vs_depth.csv'

vvd_list = glob.glob(vvd_dir + 'WOD_PFL_Oxy*0.csv')
vvd_list.sort()

# Find the duplicate flags file
# pdt_fpath = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#             'profile_data_tables\\duplicates_flagged\\' \
#             'ALL_Profiles_Oxy_1991_2020_ie_001ll_check2.csv'
pdt_fpath = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'profile_data_tables\\Argo\\' \
            'NODC_noCAD_PFL_Profiles_Oxy_1991_2020_cb_edf.csv'
# pdt = prep_pdt_v2(pdt_fpath)

pdt = pd.read_csv(pdt_fpath)
print(pdt.columns)
print(pdt.head())
# Convert date_string back to string format from float format ugh
pdt['Date_string'] = list(map(lambda x: str(x), pdt['Date_string']))
# Add column for inexact duplicate check
pdt['Inexact_duplicate_check2'] = np.repeat(False, len(pdt))
pdt['Number_of_uses'] = np.zeros(len(pdt), dtype=int)

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\2_added_dup_flags\\'

# Iterate through the files
for f in vvd_list:
    print(basename(f))
    # Read in csv file into pandas dataframe
    vvd_df = pd.read_csv(f)
    # Add flags to vvd dataframe
    df_out, pdt_out = vvd_add_dup_flags(vvd_df, pdt, verbose=False)
    # Update pdt for next iteration
    pdt = pdt_out
    # Export the returned dataframe COMMENT OUT FOR TESTING
    outname = basename(f).replace('0.csv', 'dup.csv')  # duplicate flags
    df_out.to_csv(output_dir + outname, index=False)

# Timing:
# 12:01 + 16:56 + 21.44 + 21:22 + 22:51 + 16:32 + 22:24 + 00:36 + 00:09 + 08:42
# = Timedelta('0 days 02:23:17')
# runtime = pd.Timedelta('12 min 1 s') + pd.Timedelta('16 min 56 s') + pd.Timedelta('21 min 44 s')
# runtime += pd.Timedelta('21 min 22 s') + pd.Timedelta('22 min 51 s') + pd.Timedelta('16 min 32 s')
# runtime += pd.Timedelta('22 min 24 s') + pd.Timedelta('36 s') + pd.Timedelta('9 s')
# runtime += pd.Timedelta('8 min 42 s')
#
# print(runtime)

# Check for exact duplicate rows again to be safe
vvd_dup_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\2_added_dup_flags\\'

vvd_dup_files = glob.glob(vvd_dup_dir + 'WOD_PFL_Oxy*.csv', recursive=False)

# Check all columns except for 'Profile_number'
cols_to_check = ['Cruise_number', 'Instrument_type', 'Date_string',
                 'Latitude', 'Longitude', 'Depth_m', 'Depth_flag', 'Value',
                 'Source_flag', 'Exact_duplicate_flag', 'CTD_BOT_duplicate_flag',
                 'Inexact_duplicate_flag']

vvd_dup_check2_dir = vvd_dup_dir + 'exact_duplicate_double_check\\'

for f in vvd_dup_files:
    print(basename(f))
    df = pd.read_csv(f)
    # Find the number of duplicate rows in df, if any
    # keep=False to mark all duplicates as True
    df['Exact_dup_check2'] = df.duplicated(subset=cols_to_check, keep='first')

    # Print the number of duplicate rows
    # Do an intersect with df['Exact_duplicate_flag']
    # Or just simply replace the Exact duplicate flag column?
    subsetter = (df.Exact_dup_check2 == True).values
    print(len(subsetter))
    df.loc[subsetter, 'Exact_duplicate_flag'] = True

    # Drop temporary column
    df.drop(columns='Exact_dup_check2', inplace=True)

    # Export the updated df
    df.to_csv(vvd_dup_check2_dir + basename(f), index=False)


##### TESTING #####
# Call the function
df_in = pd.read_csv(vvd_list[0])
updated_df = vvd_add_dup_flags(df_in, pdt)

# Check flags
print(updated_df[['Exact_duplicate_flag', 'CTD_BOT_duplicate_flag', 'Inexact_duplicate_flag']])

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

# Check to see that the flagging was correct
print(max(pdt.loc[:, 'Number_of_rewrites']))

# Remove transitory columns
# df_vvd.drop(columns='Number_of_rewrites')
