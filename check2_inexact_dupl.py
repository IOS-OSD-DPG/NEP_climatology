import glob
from xarray import open_dataset
from os.path import basename
import pandas as pd
import numpy as np
from tqdm import trange
from functools import reduce


def array_all_nan(arr):
    # Return true if input array contains only nans
    cond = len(np.isnan(arr)[np.isnan(arr) == True]) == len(arr)
    return cond


def open_by_source(full_path):
    # Open data file based on which data centre it came from
    # IOS and NODC files are netCDF
    # MEDS files are csv
    if full_path.endswith('.nc'):
        data = open_dataset(full_path)
    elif full_path.endswith('.csv'):
        data = pd.read_csv(full_path)
    return data


def get_ios_profile_data(ncdata, cruise_number, time, lat, lon):
    # Subset cruise, time, lat, lon
    # Use numpy.where, index[0] to access first element of tuple
    # Second element of tuple is empty
    cruise_subsetter = np.where(
        ncdata.mission_id.data == cruise_number)[0]
    time_subsetter = np.where(
        pd.to_datetime(ncdata.time.data) == time)[0]
    # Latitude and longitude floats need formatting to be equal
    lat_subsetter = np.where(
        ncdata.latitude.data == lat.astype(ncdata.latitude.data.dtype))[0]
    lon_subsetter = np.where(
        ncdata.longitude.data == lon.astype(ncdata.longitude.data.dtype))[0]

    # Intersect the subsetters to get the inexact duplicate profiles
    prof_subsetter = reduce(
        np.intersect1d, (cruise_subsetter, time_subsetter, lat_subsetter,
                         lon_subsetter))

    # This should have narrowed it down to the profiles we want
    # Otherwise, will have to subset out the exact duplicates and the CTD-BOT duplicates
    # Subset the umol/kg oxygen data

    prof = ncdata.DOXMZZ01.data[prof_subsetter]

    # Close dataset
    ncdata.close()

    # Return profile data of specified variable (oxygen)
    return prof


def get_ios_wp_profile_data(ncdata):
    # Return the oxygen values
    try:
        prof = ncdata.DOXMZZ01.data
    except AttributeError:
        prof = ncdata.DOXYZZ01.data
    return prof


def get_nodc_profile_data(ncdata, cruise_number, time, lat, lon):
    # Data are in xarray format
    # Extract profile

    # Need to strip 'b' and single quotes from WOD_cruise_identifier.data
    cruise_subsetter = np.where(
        ncdata.WOD_cruise_identifier.data.astype(str) == cruise_number.strip("b'"))[0]
    time_subsetter = np.where(
        pd.to_datetime(ncdata.time.data).strftime(
            '%Y-%m-%d %H:%M:%S') == time.strftime('%Y-%m-%d %H:%M:%S'))[0]
    lat_subsetter = np.where(ncdata.lat.data == lat)[0]
    lon_subsetter = np.where(ncdata.lon.data == lon)[0]

    # Intersect the subsetters to find the profile matching
    # the input cruise/time/lat/lon values
    prof_subsetter = reduce(
        np.intersect1d, (cruise_subsetter, time_subsetter, lat_subsetter,
                         lon_subsetter))

    prof_row_ind = ncdata.Oxygen_row_size.data[prof_subsetter].astype(int)

    # Don't need to extract the profile start indices
    # For subsetting profiles in flat Oxygen array
    # Get index of the first Oxygen observation of each profile in the
    # flat Oxygen array

    # Extract the matching profile based on selected indices
    prof = ncdata.Oxygen.data[prof_row_ind]

    # Close dataset
    ncdata.close()

    return prof


def get_meds_profile_data(df, cruise_number, time, lat, lon):
    # Data are in pandas dataframe format

    # Convert time data to pandas datetime format
    df['Hour'] = df.Time.astype(str).apply(lambda x: ('000' + x)[-4:][:-2])
    df['Minute'] = df.Time.astype(str).apply(lambda x: ('000' + x)[-4:][-2:])

    df['Time_pd'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    # Index [1] the indices not the returned unique elements
    # prof_start_indices = np.unique(df.RowNum.values, return_index=True)[1]

    # Find the profiles matching the user-entered values
    cruise_subsetter = np.where(
        df.loc[:, 'CruiseID'] == cruise_number)[0]
    time_subsetter = np.where(df.loc[:, 'Time_pd'] == time)[0]
    lat_subsetter = np.where(df.loc[:, 'Lat'] == lat)[0]
    lon_subsetter = np.where(df.loc[:, 'Lon'] == lon)[0]

    # Take multi-array intersection
    prof_subsetter = reduce(
        np.intersect1d, (cruise_subsetter, time_subsetter, lat_subsetter,
                         lon_subsetter))

    # Extract the matching profile (oxygen) values
    prof = df.loc[prof_subsetter, 'ProfParm']

    return prof


def get_profile_data(data, filename, cruise_number=None, time=None,
                     lat=None, lon=None):
    if 'IOS' in filename:
        prof = get_ios_profile_data(data, cruise_number, time, lat, lon)
    elif filename.endswith('.bot.nc') or filename.endswith('.ctd.nc'):
        # IOS Water Properties data
        prof = get_ios_wp_profile_data(data)
    elif filename.startswith('Oxy'):
        # NODC WOD data
        prof = get_nodc_profile_data(data, cruise_number, time, lat, lon)
    elif filename.startswith('MEDS'):
        prof = get_meds_profile_data(data, cruise_number, time, lat, lon)

    return prof


def prep_pdt():
    # Prepare the profile data table for inexact duplicate checking against raw profiles
    # Take full data file not the subs (subset) version
    df_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
              'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_pi.csv'

    df_pdt = pd.read_csv(df_name)

    # Find which files the inexact duplicates come from

    # Rename the first column of the original row indices
    df_pdt = df_pdt.rename(columns={'Unnamed: 0': 'Original_row_index'})

    # Initialize a new column for the second inexact duplicate check
    # Second check will check against the data profiles themselves
    df_pdt.insert(len(df_pdt.columns), 'Inexact_duplicate_check2',
                  np.repeat(False, len(df_pdt)))

    # Create a new column for Date_string in pandas datetime format
    # ValueError: time data '19910100000000' does not match format '%Y%m%d%H%M%S' (match)
    # Can't have a zeroth day of the month...
    df_pdt.insert(len(df_pdt.columns), 'Time_pd',
                  pd.to_datetime(df_pdt.Date_string, format='%Y%m%d%H%M%S'))

    # Rename MEDS source file names
    meds_subsetter = np.where(
        df_pdt.Source_data_file_name == 'MEDS_ASCII_1991_2000.csv')[0]

    df_pdt.loc[meds_subsetter,
               'Source_data_file_name'] = 'MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'

    # print(df_pdt.loc[meds_subsetter, 'Source_data_file_name'])

    return df_pdt


def get_filenames_dict():
    IOS_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\IOS_CIOOS\\'
    IOS_files = glob.glob(IOS_dir + 'IOS_BOT_Profiles_Oxy*.nc', recursive=False)
    IOS_files += glob.glob(IOS_dir + 'IOS_CTD_Profiles_Oxy*.nc', recursive=False)
    IOS_files.sort()

    # Create dictionary for file names and their base names
    # ios_fname_dict = {}
    # for f in IOS_files:
    #     ios_fname_dict.update({basename(f): f})

    ios_wp_path = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
                  'SHuntington\\'
    # Get bot files
    ios_wp_files = glob.glob(ios_wp_path + '*.bot.nc', recursive=False)
    # Get ctd files
    ios_wp_files += glob.glob(ios_wp_path + 'WP_unique_CTD_forHana\\*.ctd.nc', recursive=False)

    # Import WOD data
    WOD_nocad_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
                    'WOD_extracts\\Oxy_WOD_May2021_extracts\\'
    WOD_nocad_files = glob.glob(WOD_nocad_dir + 'Oxy*OSD.nc', recursive=False)
    WOD_nocad_files.sort()

    # Returns no files since there are no Oxy OSD data
    WOD_cad_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
                  'WOD_extracts\\WOD_July_CDN_nonIOS_extracts\\'
    WOD_cad_files = glob.glob(WOD_cad_dir + 'Oxy*OSD.nc', recursive=False)
    WOD_cad_files.sort()

    # Import MEDS data
    MEDS_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
               'meds_data_extracts\\bo_extracts\\'
    MEDS_files = glob.glob(MEDS_dir + '*.csv', recursive=False)
    MEDS_files.sort()

    all_files = IOS_files + ios_wp_files + WOD_nocad_files + WOD_cad_files + MEDS_files

    all_fname_dict = {}
    for f in all_files:
        all_fname_dict.update({basename(f): f})

    # print(len(all_fname_dict))

    return all_fname_dict


##########################################
# Verify the inexact duplicates against the raw data profiles


def run_check2():
    # Get dictionary of data files with their basenames as keys
    fname_dict = get_filenames_dict()

    # Dataframe containing the partner indices
    df = prep_pdt()

    # Things to check:
    # mission_id (cruise number), instrument, time, latitude, longitude

    # Create subsetter for df to extract only the inexact duplicate rows
    # Includes first occurrences
    subsetter = (df.Inexact_duplicate_row == True).values

    df_subset = df.loc[subsetter]

    # Iterate through df_subset
    for i in trange(len(df_subset)):  # 200
        # Check that the row is not the first occurrence of an inexact duplicate
        if df_subset.Partner_index.iloc[i] != -1:
            # np.where returns a tuple; tuple's first element is an array containing the index
            row1_ind = np.where(
                df_subset.Original_row_index.values == df_subset.Partner_index.iloc[i])[0][0]
            row2_ind = i
            print(row1_ind, row2_ind)

            # Find the files that both rows came from
            fname1 = df_subset.Source_data_file_name.iloc[row1_ind]
            fname2 = df_subset.Source_data_file_name.iloc[row2_ind]

            print(fname1, fname2)

            # Retrieve instrument type: Either BOT or CTD
            inst1 = df_subset.Instrument_type.iloc[row1_ind]
            inst2 = df_subset.Instrument_type.iloc[row2_ind]

            # Break out of loop if source file names are not the same
            if fname1 != fname2 and inst1 == inst2:
                print('Source file names are not equal')
                print('Instrument types {} are equal'.format(inst1))
                # print('Proceeding to next iteration')
                # continue

            #########################
            # Create a new function for this section
            # Instruments are the same, fnames may or may not be the same

            data1 = open_by_source(fname_dict[fname1])
            data2 = open_by_source(fname_dict[fname2])

            # Find the inexact duplicate profiles within the data files
            prof1 = get_profile_data(data1, fname1, df_subset.Cruise_number.iloc[row1_ind],
                                     df_subset.Time_pd.iloc[row1_ind],
                                     df_subset.Latitude.iloc[row1_ind],
                                     df_subset.Longitude.iloc[row1_ind])
            prof2 = get_profile_data(data2, fname2, df_subset.Cruise_number.iloc[row2_ind],
                                     df_subset.Time_pd.iloc[row2_ind],
                                     df_subset.Latitude.iloc[row2_ind],
                                     df_subset.Longitude.iloc[row2_ind])

            # print(prof1, prof2, sep='\n')
            #############################

            # Use lengths as a first quick check
            if not np.any(prof1) and not np.any(prof2):
                print('Returned arrays are both empty')
            elif array_all_nan(prof1) and array_all_nan(prof2):
                print('Returned arrays both contain only nans')
            elif len(prof1) != len(prof2):
                print('Profile lengths not equal: {} != {}'.format(len(prof1), len(prof2)))
            # Check set intersection instead of equality instead??
            elif np.array_equal(prof1, prof2, equal_nan=True):
                # If arrays are of equal length, then check if they are equal
                print('Inexact duplicate confirmed')
                # Leave row1_ind row flag as false since it's the
                # first occurrence of the duplicate
                df_subset.loc[row2_ind, 'Inexact_duplicate_check2'] = True
            else:
                print('Profiles of equal length but are not duplicates')

            # Proceed to next iteration

    # Print basic accounting statistics
    print('Accounting statistics:')
    print('Subset length', len(df_subset))
    print('Number of verified inexact duplicates:',
          len(df_subset.loc[(df_subset.Inexact_duplicate_check2 == True).values]))
    print('Number of inexact duplicates that failed the check:',
          len(df_subset.loc[(df_subset.Inexact_duplicate_check2 != True).values]))

    # Merge df and df_subset
    df_subset_inv = df.loc[~subsetter]

    df_out = pd.concat([df_subset, df_subset_inv])

    # Formatting df for export
    df_out = df_out.drop(columns='Time_pd')

    # Convert boolean flags to strings
    df_out.iloc[:, 9] = df_out.iloc[:, 9].astype(str)
    df_out.iloc[:, 10] = df_out.iloc[:, 10].astype(str)
    df_out.iloc[:, 11] = df_out.iloc[:, 11].astype(str)

    # Export file
    df_out_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
                  'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_check2.csv'
    df_out.to_csv(df_out_name, index=False)

    return


run_check2()

# f = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
#     'SHuntington\\WP_unique_CTD_forHana\\2018-106-0001.ctd.nc'
#
# data = open_dataset(f)