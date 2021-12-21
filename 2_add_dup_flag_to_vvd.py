# Add duplicate flags from profile data tables to value vs depth tables

import numpy as np
import pandas as pd
from tqdm import trange
import glob
from os.path import basename, join


def vvd_add_dup_flags(var_name, df_vvd, df_pdt, profs_to_recheck, verbose=False):
    """
    Match duplicate flags from the profile data tables to the correct profiles in
    the value vs depth tables
    :param var_name:
    :param df_vvd: dataframe
    :param df_pdt: dataframe
    :param profs_to_recheck: list
    :param verbose:
    :return:
    """
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
        time_vvd = df_vvd.loc[unique[i], 'Date_string']  # .astype(str)
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
        # Check lon and lat are "close enough"
        if pd.isna(cruise_vvd):
            # np.nan == np.nan is False, so need a different check method
            indices_pdt = np.where((pd.isna(df_pdt.Cruise_number)) &
                                   (df_pdt.Instrument_type == instrument_vvd) &
                                   (df_pdt.Date_string == time_vvd) &
                                   (abs(df_pdt.Latitude - lat_vvd) < 1e-5) &
                                   (abs(df_pdt.Longitude - lon_vvd) < 1e-5))[0]
        else:
            # The original checking method for cruise number
            indices_pdt = np.where((df_pdt.Cruise_number == cruise_vvd) &
                                   (df_pdt.Instrument_type == instrument_vvd) &
                                   (df_pdt.Date_string == time_vvd) &
                                   (abs(df_pdt.Latitude - lat_vvd) < 1e-5) &
                                   (abs(df_pdt.Longitude - lon_vvd) < 1e-5))[0]

        if verbose:
            print('Number of matching profiles:', len(indices_pdt))
            if len(indices_pdt) == 0:
                print('Warning: No rows matching search in pdt')
            elif len(indices_pdt) > 1:
                print('Warning: More than one row match returned from pdt')
                print('Number of uses per matching profile returned:',
                      df_pdt.loc[indices_pdt, 'Number_of_uses'])
                profs_to_recheck.append(
                    (var_name, cruise_vvd, instrument_vvd, time_vvd, lat_vvd, lon_vvd))
                # Use the profile that hasn't already been used
                prof_index_to_use = -1
                for ind in indices_pdt:
                    if df_pdt.loc[ind, 'Number_of_uses'] == 0:
                        prof_index_to_use = ind
                        break
                # Check if all profiles have been used ...
                if prof_index_to_use == -1:
                    print('Warning: all matching profiles have already been used')
                    prof_index_to_use = indices_pdt[0]
            elif len(indices_pdt) == 1:
                print('Row match found')
                prof_index_to_use = indices_pdt[0]
                print(unique[i], indices_pdt[0])

        # Index the pdt
        # Populate the duplicate flag columns in the value vs depth dataframe
        df_vvd.loc[unique[i]: end_of_prof, 'Exact_duplicate_flag'
                   ] = df_pdt.loc[prof_index_to_use, 'Exact_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: end_of_prof, 'CTD_BOT_duplicate_flag'
                   ] = df_pdt.loc[prof_index_to_use, 'CTD_BOT_duplicate_row'].astype(bool)
        df_vvd.loc[unique[i]: end_of_prof, 'Inexact_duplicate_flag'
                   ] = df_pdt.loc[prof_index_to_use, 'Inexact_duplicate_check2'].astype(bool)

        df_pdt.loc[prof_index_to_use, 'Number_of_uses'] += 1

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
    return df_vvd, df_pdt, profs_to_recheck


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
    # pdt_df['Date_string'] = list(map(lambda x: str(x)[:-2], pdt_df['Date_string']))

    # Fix NODC cruise numbers in PDT that are like "b'XXXXXXXX'"
    # pdt_df['Cruise_number'] = list(map(lambda x: str(x).strip("b'"), pdt_df['Cruise_number']))
    # print(pdt_df.loc[:5, 'Cruise_number'])

    # Initialize column to count how many times each row is used
    # during the iteration -- it should just be once
    # How to deal with exact duplicates though?
    pdt_df['Number_of_uses'] = np.zeros(len(pdt_df), dtype=int)

    return pdt_df


# --------------------------------------------------------------------------------------
for var in ['Temp', 'Sal']:
    variable_name = var
    # Value vs depth table folder
    vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\1_original\\'
    # vvd_fname = vvd_dir + 'ALL_Oxy_1991_2020_value_vs_depth.csv'
    vvd_list = glob.glob(vvd_dir + '*{}*value_vs_depth_0.csv'.format(variable_name))

    # vvd_list = glob.glob(vvd_dir + 'WOD_PFL_Oxy*0.csv')
    print(len(vvd_list))
    vvd_list.sort()

    # Find the duplicate flags file
    pdt_fpath = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                'profile_data_tables\\duplicates_flagged\\' \
                'ALL_Profiles_{}_1991_2020_ie_001ll_check2.csv'.format(variable_name)
    # pdt_fpath = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
    #             'profile_data_tables\\Argo\\' \
    #             'NODC_noCAD_PFL_Profiles_Oxy_1991_2020_cb_edf.csv'
    # pdt = prep_pdt_v2(pdt_fpath)

    output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                 'value_vs_depth\\2_added_dup_flags\\'

    pdt = pd.read_csv(pdt_fpath)
    print(pdt.columns)
    print(pdt.head())
    # Convert date_string back to string format from float format ugh
    # pdt['Date_string'] = list(map(lambda x: str(x), pdt['Date_string']))
    # Add column for inexact duplicate check
    # pdt['Inexact_duplicate_check2'] = np.repeat(False, len(pdt)) ??????
    pdt['Number_of_uses'] = np.zeros(len(pdt), dtype=int)

    # Initialize list of profiles to recheck matching
    profiles_to_recheck = []

    # Iterate through the files
    for f in vvd_list[:]:  # 9:
        print(basename(f))
        # Read in csv file into pandas dataframe
        vvd_df = pd.read_csv(f)
        # correct date string format of vvd df
        # vvd_df['Date_string'] = list(map(lambda x: str(x)[:-2], vvd_df['Date_string']))
        # Add flags to vvd dataframe
        df_out, pdt_out, profiles_to_recheck_out = vvd_add_dup_flags(
            variable_name, vvd_df, pdt, profiles_to_recheck, verbose=True)
        # Update pdt for next iteration
        pdt = pdt_out
        # Update list of profiles to recheck for next iteration
        profiles_to_recheck = profiles_to_recheck_out
        # Export the returned dataframe COMMENT OUT FOR TESTING
        outname = basename(f).replace('0.csv', 'dup.csv')  # duplicate flags
        df_out.to_csv(output_dir + outname, index=False)

    # Export list of tuples as pd dataframe csv
    # var_name, cruise_vvd, instrument_vvd, time_vvd, lat_vvd, lon_vvd
    df_prof2check = pd.DataFrame(
        profiles_to_recheck,
        columns=['Variable', 'Cruise_number', 'Instrument_type', 'Date_string',
                 'Latitude', 'Longitude']
    )
    df_prof2check_fname = join(output_dir, '{}_prof_matches_to_recheck.csv'.format(variable_name))
    df_prof2check.to_csv(df_prof2check_fname)

    print('Number of profiles to check for {}:'.format(variable_name), len(df_prof2check))
    print()

# Timing:
# 12:01 + 16:56 + 21.44 + 21:22 + 22:51 + 16:32 + 22:24 + 00:36 + 00:09 + 08:42
# = Timedelta('0 days 02:23:17')
# runtime = pd.Timedelta('12 min 1 s') + pd.Timedelta('16 min 56 s') + pd.Timedelta('21 min 44 s')
# runtime += pd.Timedelta('21 min 22 s') + pd.Timedelta('22 min 51 s') + pd.Timedelta('16 min 32 s')
# runtime += pd.Timedelta('22 min 24 s') + pd.Timedelta('36 s') + pd.Timedelta('9 s')
# runtime += pd.Timedelta('8 min 42 s')
#
# print(runtime)

# -----------------------------------DOUBLE-CHECK------------------------------------------
# Check for exact duplicate rows again to be safe
vvd_dup_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\2_added_dup_flags\\'

vvd_dup_files = glob.glob(vvd_dup_dir + '*{}*.csv'.format(variable_name), recursive=False)
print(vvd_dup_files)

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


# --------------------------------------TESTING----------------------------------------------
# # Call the function
# df_in = pd.read_csv(vvd_list[0])
# updated_df = vvd_add_dup_flags(df_in, pdt)
#
# # Check flags
# print(updated_df[['Exact_duplicate_flag', 'CTD_BOT_duplicate_flag', 'Inexact_duplicate_flag']])
#
# # Remove temporary columns
# updated_df.drop(columns='Instrument_type')
# # Export the updated dataframe
# out_name = vvd_list[0].replace('.', '_flag.')
# updated_df.to_csv(out_name, index=False)
#
# # Approach #2
# # Try iterating through the separate vvd dfs
# meds_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
#             'MEDS_BOT_Oxy_1991_1995_value_vs_depth_0.csv'
#
# meds_df = pd.read_csv(meds_file)
# meds_df['Date_string'] = meds_df['Date_string'].astype(str)
#
# # Check to see that the flagging was correct
# print(max(pdt.loc[:, 'Number_of_rewrites']))
#
# # Remove transitory columns
# # df_vvd.drop(columns='Number_of_rewrites')
