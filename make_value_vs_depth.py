# Create value vs depth tables
# Do not apply any flags or unit conversions just yet

import numpy as np
import pandas as pd
import glob
from xarray import open_dataset
from copy import deepcopy
from tqdm import trange
from gsw import z_from_p


# Start with IOS data
def ios_to_vvd0(ncdata, instrument='BOT'):
    # Get index of first measurement of each profile
    # indexer = np.unique(ncdata.profile.data, return_index=True)[1]

    # Initialize empty dataframe
    df_out = pd.DataFrame()

    # Add profile number as a column
    unique = np.unique(ncdata.profile.data, return_index=True)[1]
    df_out['Profile_number'] = np.zeros(len(ncdata.profile.data), dtype=int)

    # print(len(unique), len(ncdata.mission_id.data))

    num = 1
    # Skip the first profile since its number is already zero
    for j in range(1, len(unique) - 1):
        df_out.loc[unique[j]:unique[j + 1], 'Profile_number'] = num
        num += 1

    # Don't forget to number the last profile!
    df_out.loc[unique[-1]:, 'Profile_number'] = num

    print('Total number of profiles:', num + 1)  # Started from zero

    df_out['Cruise_number'] = ncdata.mission_id.data
    df_out['Instrument_type'] = np.repeat(instrument, len(df_out))  # To remove later
    df_out['Date_string'] = pd.to_datetime(ncdata.time.data).strftime('%Y%m%d%H%M%S')
    df_out['Latitude'] = ncdata.latitude.data
    df_out['Longitude'] = ncdata.longitude.data
    df_out['Depth_m'] = ncdata.depth.data
    df_out['Depth_flag'] = np.ones(len(ncdata.row), dtype=int)  # To remove later
    df_out['Value'] = ncdata.DOXMZZ01.data
    df_out['Source_flag'] = np.ones(len(ncdata.row), dtype=int)  # To remove later

    return df_out


# NODC data
def nodc_to_vvd0(ncdata, instrument='BOT', counter=0):
    # Transfer NODC data to value vs depth format
    # Add duplicate flags at a later time

    df_out = pd.DataFrame()

    profile_number = np.zeros(len(ncdata.Oxygen.data), dtype=int)
    cruise_number = np.repeat('XXXXXXXX', len(ncdata.Oxygen.data))
    date_string = np.repeat('YYYYMMDDhhmmss', len(ncdata.Oxygen.data))
    latitude = np.repeat(0., len(ncdata.Oxygen.data))
    longitude = np.repeat(0., len(ncdata.Oxygen.data))

    start_ind = 0
    for i in range(len(ncdata.Oxygen_row_size.data)):

        profile_number[start_ind: start_ind + int(ncdata.Oxygen_row_size.data[i])
                       ] = counter

        cruise_number[start_ind: start_ind + int(ncdata.Oxygen_row_size.data[i])
                      ] = ncdata.WOD_cruise_identifier.data[i].astype(str)

        # print(cruise_number)

        date_string[start_ind: start_ind + int(ncdata.Oxygen_row_size.data[i])
                    ] = pd.to_datetime(ncdata.time.data[i]).strftime('%Y%m%d%H%M%S')

        latitude[start_ind: start_ind + int(ncdata.Oxygen_row_size.data[i])
                 ] = ncdata.lat.data[i].astype(str)

        longitude[start_ind: start_ind + int(ncdata.Oxygen_row_size.data[i])
                  ] = ncdata.lon.data[i].astype(str)

        # print(longitude)

        counter += 1
        start_ind = start_ind + int(ncdata.Oxygen_row_size.data[i])

        # print(start_ind)

    # Write arrays to initialized dataframe
    df_out['Profile_number'] = profile_number
    df_out['Cruise_number'] = cruise_number
    df_out['Instrument_type'] = np.repeat(instrument, len(df_out))  # To remove later
    df_out['Date_string'] = date_string
    df_out['Latitude'] = latitude
    df_out['Longitude'] = longitude
    df_out['Depth_m'] = ncdata.z.data
    df_out['Depth_flag'] = ncdata.z_WODflag.data
    df_out['Value'] = ncdata.Oxygen.data
    df_out['Source_flag'] = ncdata.Oxygen_WODflag.data

    return df_out, counter


def nodc_to_vvd(ncdata, df_pdt):
    # Make sure to correct the format of cruise IDs (remove b')

    # Get first index of each profile in ncdata.Oxygen.data
    prof_start_indices = np.concatenate(
        [np.zeros(1, dtype=int), np.cumsum(ncdata.Oxygen_row_size.data,
                                           dtype='int')[:-1]])

    # Make column that flags when file name starts with Oxy (for NODC data)
    df_pdt['Fname_startswith_Oxy'] = list(map(lambda x: x.startswith('Oxy'),
                                              df_pdt['Source_data_file_name']))

    # Initialize list of dictionaries, to add to in for-loop
    dict_list = []

    # Iterate through the profiles in the NODC file
    for i in trange(len(prof_start_indices)):
        cruise_nodc = ncdata.WOD_cruise_identifier.data[i].astype(str)
        time_nodc = pd.to_datetime(ncdata.time.data[i]).strftime('%Y%m%d%H%M%S')
        lat_nodc = ncdata.lat.data[i]
        lon_nodc = ncdata.lon.data[i]

        indexer = np.where((df_pdt.Fname_startswith_Oxy == True) &
                           (df_pdt.Cruise_number == cruise_nodc) &
                           (df_pdt.Date_string == time_nodc) &
                           (df_pdt.Latitude == lat_nodc) &
                           (df_pdt.Longitude == lon_nodc))

        if len(indexer) > 1:
            print('Warning: indexer has length {}'.format(len(indexer)))

        # Find the duplicate flags using indexer
        ex_dup_flag = df_pdt.loc[indexer[0], 'Exact_duplicate_row']
        cb_dup_flag = df_pdt.loc[indexer[0], 'CTD_BOT_duplicate_row']
        ie_dup_flag = df_pdt.loc[indexer[0], 'Inexact_duplicate_check2']

        # Iterate over depth
        for j in range(prof_start_indices[i],
                       prof_start_indices[i] + ncdata.Oxygen_row_size.data[i]):
            dict_list.append({'CruiseID': cruise_nodc,
                              'Date_string': time_nodc,
                              'Latitude': lat_nodc,
                              'Longitude': lon_nodc,
                              'Depth_m': ncdata.z.data[j],
                              'Depth_flag': ncdata.z_WODflag.data[j],
                              'Value': ncdata.Oxygen.data[j],
                              'Source_flag': ncdata.Oxygen_WODflag.data[j],
                              'Exact_duplicate_flag': ex_dup_flag,
                              'CTD_BOT_duplicate_flag': cb_dup_flag,
                              'Inexact_duplicate_flag': ie_dup_flag})

    df_out = pd.DataFrame.from_dict(dict_list)

    return df_out


# MEDS data
def meds_to_vvd(df_meds, df_pdt):
    # vvd: value vs depth table
    # colnames: column names for output value vs depth table

    # meds_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\meds_data_extracts\\' \
    #             'bo_extracts\\MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'

    # Add pandas date string column to df_meds, as before
    df_meds['Hour'] = df_meds.Time.astype(str).apply(
        lambda x: ('000' + x)[-4:][:-2])
    df_meds['Minute'] = df_meds.Time.astype(str).apply(
        lambda x: ('000' + x)[-4:][-2:])

    df_meds['Date_string'] = pd.to_datetime(
        df_meds[['Year', 'Month', 'Day', 'Hour', 'Minute']]).strftime(
        '%Y%m%d%H%M%S')

    # Unit conversions for depth/pressure
    df_meds['Depth_m'] = df_meds['Depth/Press']
    pressure_subsetter = np.where((df_meds.loc[:, 'D_P_code'] == 'P').values)[0]
    # Not sure if df format is ok for z_from_p or if array type is required
    df_meds.loc[pressure_subsetter, 'Depth_m'] = z_from_p(
        df_meds.loc[pressure_subsetter, 'Depth_m'].values,
        df_meds.loc[pressure_subsetter, 'Lat'].values)

    # Remove Depth/Press and D_P_code after unit conversions done
    # Remove D_P_flag after source flagging is done

    # Subset the df_pdt dataframe for only meds data
    meds_fname = 'MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'
    pdt_subset = deepcopy(df_pdt.loc[df_pdt.Source_data_file_name == meds_fname])

    pdt_subset = pdt_subset.reset_index()

    # print(pdt_subset.index)

    # Iterate through MEDS dataframe
    dict_list = []

    # Should I iterate through the pdt instead? Should it matter?

    for i in trange(len(df_meds)):

        # Find the duplicate flags in df_pdt
        cruise_meds = df_meds.loc[i, 'CruiseID']
        time_meds = df_meds.loc[i, 'Date_string']
        lat_meds = df_meds.loc[i, 'Lat']
        lon_meds = -df_meds.loc[i, 'Lon']  # convert to positive towards east not west

        # Only use bottle data from MEDS so index by instrument type too
        indexer = np.where((pdt_subset.Cruise_number == cruise_meds) &
                           (pdt_subset.Date_string == time_meds) &
                           (pdt_subset.Latitude == lat_meds) &
                           (pdt_subset.Longitude == lon_meds))[0]

        print(np.where((pdt_subset.Cruise_number == cruise_meds))[0])
        print(np.where((pdt_subset.Date_string == time_meds))[0])
        print(np.where((pdt_subset.Latitude == lat_meds))[0])
        print(np.where((pdt_subset.Longitude == lon_meds))[0])

        if len(indexer) == 0:
            print('Warning: Row search returned no matches')
        elif len(indexer) > 1:
            print('Warning: More than one row match for finding duplicate flags')

        ex_dup_flag = pdt_subset.loc[indexer[0], 'Exact_duplicate_row']
        cb_dup_flag = pdt_subset.loc[indexer[0], 'CTD_BOT_duplicate_row']
        ie_dup_flag = pdt_subset.loc[indexer[0], 'Inexact_duplicate_check2']

        dict_list.append({'CruiseID': cruise_meds,
                          'Date_string': time_meds,
                          'Latitude': lat_meds,
                          'Longitude': lon_meds,
                          'Depth_m': df_meds.loc[i, 'Depth_m'],
                          'Depth_flag': df_meds.loc[i, 'D_P_flag'],
                          'Value': df_meds.loc[i, 'ProfParm'],
                          'Source_flag': df_meds.loc[i, 'PP_flag'],
                          'Exact_duplicate_flag': ex_dup_flag,
                          'CTD_BOT_duplicate_flag': cb_dup_flag,
                          'Inexact_duplicate_flag': ie_dup_flag})

        # Delete the row used so that it's only used once
        pdt_subset = pdt_subset.drop(index=indexer[0])

    # Convert list to dataframe
    df_out = pd.DataFrame.from_dict(dict_list)

    return df_out


def meds_to_vvd0(df_meds, instrument='BOT'):
    # Just convert to value-vs-depth format without adding duplicate flags
    # Add duplicate flags at a later step

    # Add profile number counter
    df_meds['Profile_number'] = np.zeros(len(df_meds), dtype=int)

    unique = np.unique(df_meds.RowNum, return_index=True)[1]

    counter = 1
    for i in range(1, len(unique) - 1):
        df_meds.loc[unique[i]:unique[i + 1], 'Profile_number'] = counter
        counter += 1

    # Don't forget last profile
    df_meds.loc[unique[-1]:, 'Profile_number'] = counter

    # Add pandas date string column to df_meds, as before
    df_meds['Hour'] = df_meds.Time.astype(str).apply(
        lambda x: ('000' + x)[-4:][:-2])
    df_meds['Minute'] = df_meds.Time.astype(str).apply(
        lambda x: ('000' + x)[-4:][-2:])

    df_meds['Date_string'] = pd.to_datetime(
        df_meds[['Year', 'Month', 'Day', 'Hour', 'Minute']]).dt.strftime(
        '%Y%m%d%H%M%S')

    # Unit conversions for depth/pressure
    df_meds['Depth_m'] = df_meds['Depth/Press']
    pressure_subsetter = np.where((df_meds.loc[:, 'D_P_code'] == 'P').values)[0]
    # Not sure if df format is ok for z_from_p or if array type is required
    df_meds.loc[pressure_subsetter, 'Depth_m'] = z_from_p(
        df_meds.loc[pressure_subsetter, 'Depth_m'].values,
        df_meds.loc[pressure_subsetter, 'Lat'].values)

    # Write to dataframe to output
    df_out = pd.DataFrame()
    df_out['Profile_number'] = df_meds['Profile_number']
    df_out['Instrument_type'] = np.repeat(instrument, len(df_out))  # To remove later
    df_out['Date_string'] = df_meds['Date_string']
    df_out['Latitude'] = df_meds['Lat']
    df_out['Longitude'] = -df_meds['Lon']  # Convert to positive East
    df_out['Depth_m'] = df_meds['Depth_m']
    df_out['Depth_flag'] = df_meds['D_P_flag']
    df_out['Value'] = df_meds['ProfParm']
    df_out['Source_flag'] = df_meds['PP_flag']

    return df_out


def meds_add_vvd_dup_flags(meds_df, df_pdt):
    # Add the profile data table flags to the meds value vs depth table
    # Subset the df_pdt dataframe for only meds data
    meds_fname = 'MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'
    pdt_subset = deepcopy(df_pdt.loc[df_pdt.Source_data_file_name == meds_fname])

    pdt_subset = pdt_subset.reset_index()

    # Initialize columns for Exact, CTD-bottle and inexact duplicate flags
    meds_df['Exact_duplicate_flag'] = np.repeat(-1, len(meds_df))
    meds_df['CTD_BOT_duplicate_flag'] = np.repeat(-1, len(meds_df))
    meds_df['Inxact_duplicate_flag'] = np.repeat(-1, len(meds_df))

    for i in trange(len(pdt_subset)):
        cruise_pdt = pdt_subset.loc[i, 'Cruise_number']
        time_pdt = pdt_subset.loc[i, 'Date_string']
        lat_pdt = pdt_subset.loc[i, 'Cruise_number']
        lon_pdt = pdt_subset.loc[i, 'Cruise_number']

    return


def get_pdt_df():
    # Open the flags dataset (profile data table, or PDT) from the previous step
    df_pdt_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
                  'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_check2.csv'

    pdt_df = pd.read_csv(df_pdt_file, index_col=False)
    # Don't need this column
    pdt_df = pdt_df.drop(columns='Original_row_index')
    # Drop rows that contain any nans/blank entries
    pdt_df = pdt_df.dropna(axis='index', how='any')
    # Convert date_string back to string format from float format ugh
    pdt_df['Date_string'] = list(map(lambda x: str(x)[:-2], pdt_df['Date_string']))

    return pdt_df


##################################
# Import all data; no args to pass
# fname_dict = get_filenames_dict()

pdt = get_pdt_df()

# Create a column for the set union of the flags we want to act on
# Do not remove data based on inexact duplicates yet
# df_pdt.insert(len(df_pdt.columns), 'Duplicates_to_remove',
#               df_pdt.Exact_duplicate_row | df_pdt.CTD_BOT_duplicate_row)

# Will iterate through this subset of the dataframe
# for efficiency
# df_pdt_subset = deepcopy(
#     df_pdt.loc[df_pdt.Exact_duplicate_row | df_pdt.CTD_BOT_duplicate_row])

# # Iterate through files in dictionary
# for key in fname_dict.keys():
#     data = open_by_source(fname_dict[key])
#
#     if 'IOS' in key:
#         ios_to_vvd(data, pdt)
#     elif key.startswith('Oxy'):
#         nodc_to_vvd(data, pdt)
#     elif 'MEDS' in key:
#         meds_to_vvd(data, pdt)
#
# print('Done')

# IOS data
ios_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
           'IOS_CIOOS\\IOS_BOT_Profiles_Oxy_19910101_20201231.nc'
ios_data = open_dataset(ios_file)

ios_df = ios_to_vvd0(ios_data)

ios_df_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\IOS_BOT_Oxy_1991_2020_value_vs_depth_0.csv'

ios_df.to_csv(ios_df_name, index=False)

ios_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\IOS_CIOOS\\'
ios_ctd = glob.glob(ios_dir + 'IOS_CTD_Profiles_Oxy*.nc')
ios_ctd.sort()

# ios_ctd_df = pd.DataFrame()

years = [(1991, 1995), (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
for i in trange(len(ios_ctd)):
    df_add = ios_to_vvd0(open_dataset(ios_ctd[i]), instrument='CTD')
    fname = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\IOS_CTD_Oxy_{}_{}_value_vs_depth_0.csv'.format(
        years[i][0], years[i][1])

    df_add.to_csv(fname, index=False)

    # ios_ctd_df = pd.concat([ios_ctd_df, df_add])

ios_ctd_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
               'value_vs_depth\\IOS_CTD_Oxy_1991_2020_value_vs_depth_0.csv'

# ios_ctd_df.to_csv(ios_ctd_name, index=False)

########################
# NODC OSD data

osd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'source_format\\WOD_extracts\\Oxy_WOD_May2021_extracts\\'

osd_files = glob.glob(osd_dir + 'Oxy_*_OSD.nc')
osd_files.sort()

osd_df = pd.DataFrame()

prof_count_old = 0
for i in trange(len(osd_files)):
    print(prof_count_old)
    df_add, prof_count_new = nodc_to_vvd0(open_dataset(osd_files[i]), counter=prof_count_old)
    prof_count_old = prof_count_new
    osd_df = pd.concat([osd_df, df_add])

osd_df_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\WOD_BOT_Oxy_1991_2020_value_vs_depth_0.csv'

osd_df.to_csv(osd_df_name, index=False)

########################
# Test out each function
meds_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
            'meds_data_extracts\\bo_extracts\\MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'

meds_data = pd.read_csv(meds_file)

# df_meds_vvd = meds_to_vvd(meds_data, pdt)
df_meds_vvd0 = meds_to_vvd0(meds_data)

# Write output dataframe to csv file
vvd0_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\MEDS_BOT_Oxy_1991_1995_value_vs_depth_0.csv'

df_meds_vvd0.to_csv(vvd0_name, index=False)

# Concatenate all dataframes together
vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\'
files = glob.glob(vvd_dir + '*.csv')
files.sort()

df_all = pd.DataFrame()
for f in files:
    df_add = pd.read_csv(f)
    df_all = pd.concat([df_all, df_add])

# Make sure columns do not contain mixed types
vvd_cols = ['Date_string', 'Instrument_type', 'Latitude', 'Longitude', 'Depth_m',
            'Depth_flag', 'Value', 'Source_flag']

df_all['Profile_number'] = df_all['Profile_number'].astype(int)
df_all['Date_string'] = df_all['Date_string'].astype(str)
df_all['Instrument_type'] = df_all['Instrument_type'].astype(str)
df_all['Latitude'] = df_all['Latitude'].astype(float)
df_all['Longitude'] = df_all['Longitude'].astype(float)
df_all['Depth_m'] = df_all['Depth_m'].astype(float)
df_all['Depth_flag'] = df_all['Depth_flag'].astype(int)
df_all['Value'] = df_all['Value'].astype(float)
df_all['Source_flag'] = df_all['Source_flag'].astype(int)

all_name = 'ALL_Oxy_1991_2020_value_vs_depth.csv'

df_all.to_csv(vvd_dir + all_name, index=False)

# df_all.shape
# (18549636, 10)

# Fix the index
df_all = df_all.reset_index(drop=True)

# TOO SLOW
# Redo the profile numbers?
df_all['All_profile_number'] = np.zeros(len(df_all), dtype=int)
number_count = 0
for i in trange(1, len(df_all)):
    if df_all.loc[i, 'Profile_number'] != df_all.loc[i - 1, 'Profile_number']:
        number_count += 1
        df_all.loc[i, 'All_profile_number'] = number_count
    else:
        df_all.loc[i, 'All_profile_number'] = number_count

# Remove the profile number column
df_all = df_all.drop(columns='Profile_number')