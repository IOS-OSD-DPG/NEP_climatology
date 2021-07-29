# Create value vs depth tables
# Do not apply any flags or unit conversions just yet

import numpy as np
import pandas as pd
import glob
from xarray import open_dataset
from copy import deepcopy
from check2_inexact_dupl import get_filenames_dict, open_by_source
from tqdm import trange
from gsw import z_from_p


# Start with IOS data
def ios_to_vvd0(ncdata):
    # Get index of first measurement of each profile
    # indexer = np.unique(ncdata.profile.data, return_index=True)[1]

    # Initialize empty dataframe
    df_out = pd.DataFrame()

    df_out['Cruise_number'] = ncdata.mission_id.data
    df_out['Date_string'] = pd.to_datetime(ncdata.time.data).strftime('%Y%m%d%H%M%S')
    df_out['Latitude'] = ncdata.latitude.data
    df_out['Longitude'] = ncdata.longitude.data
    df_out['Depth_m'] = ncdata.depth.data
    df_out['Depth_flag'] = np.ones(len(ncdata.row), dtype=int)
    df_out['Value'] = ncdata.DOXMZZ01.data
    df_out['Source_flag'] = np.ones(len(ncdata.row), dtype=int)

    return df_out


# NODC data
def nodc_to_vvd0(ncdata):
    # Transfer NODC data to value vs depth format
    # Add duplicate flags at a later time

    df_out = pd.DataFrame()

    cruise_number = np.repeat('XXXXXXXX', len(ncdata.Oxygen.data))
    date_string = np.repeat('YYYYMMDDhhmmss', len(ncdata.Oxygen.data))
    latitude = np.repeat(0., len(ncdata.Oxygen.data))
    longitude = np.repeat(0., len(ncdata.Oxygen.data))

    start_ind = 0
    for i in range(len(ncdata.Oxygen_row_size.data)):

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

        start_ind = start_ind + int(ncdata.Oxygen_row_size.data[i])

        # print(start_ind)

    # Write arrays to initialized dataframe
    df_out['Cruise_number'] = cruise_number
    df_out['Date_string'] = date_string
    df_out['Latitude'] = latitude
    df_out['Longitude'] = longitude
    df_out['Depth_m'] = ncdata.z.data
    df_out['Depth_flag'] = ncdata.z_WODflag.data
    df_out['Value'] = ncdata.Oxygen.data
    df_out['Source_flag'] = ncdata.Oxygen_WODflag.data

    return df_out


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


def meds_to_vvd0(df_meds):
    # Just convert to value-vs-depth format without adding duplicate flags
    # Add duplicate flags at a later step

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

    # Initialize list of dictionaries
    dict_list = []

    for i in trange(len(df_meds)):
        dict_list.append({'Cruise_number': df_meds.loc[i, 'CruiseID'],
                          'Date_string': df_meds.loc[i, 'Date_string'],
                          'Latitude': df_meds.loc[i, 'Lat'],
                          'Longitude': df_meds.loc[i, 'Lon'],
                          'Depth_m': df_meds.loc[i, 'Depth_m'],
                          'Depth_flag': df_meds.loc[i, 'D_P_flag'],
                          'Value': df_meds.loc[i, 'ProfParm'],
                          'Source_flag': df_meds.loc[i, 'PP_flag']})

    # Convert list to dataframe
    df_out = pd.DataFrame.from_dict(dict_list)

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
fname_dict = get_filenames_dict()

pdt = get_pdt_df()

# Create a column for the set union of the flags we want to act on
# Do not remove data based on inexact duplicates yet
# df_pdt.insert(len(df_pdt.columns), 'Duplicates_to_remove',
#               df_pdt.Exact_duplicate_row | df_pdt.CTD_BOT_duplicate_row)

# Will iterate through this subset of the dataframe
# for efficiency
# df_pdt_subset = deepcopy(
#     df_pdt.loc[df_pdt.Exact_duplicate_row | df_pdt.CTD_BOT_duplicate_row])

# Name of output file
df_val_dep_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
                  'value_vs_depth_noqc.csv'

# Initialize value vs depth dataframe
vvd_cols = ['Date_string', 'Latitude', 'Longitude', 'Depth_m',
            'Value', 'Source_flag', 'Inexact_duplicate_flag']
df_val_dep = pd.DataFrame(columns=vvd_cols)

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

ios_ctd_df = pd.DataFrame()

for i in trange(len(ios_ctd)):
    df_add = ios_to_vvd0(open_dataset(ios_ctd[i]))
    ios_ctd_df = pd.concat([ios_ctd_df, df_add])

ios_ctd_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
               'value_vs_depth\\IOS_CTD_Oxy_1991_2020_value_vs_depth_0.csv'

ios_ctd_df.to_csv(ios_ctd_name, index=False)

########################
# NODC OSD data

osd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'source_format\\WOD_extracts\\Oxy_WOD_May2021_extracts\\'

osd_files = glob.glob(osd_dir + 'Oxy_*_OSD.nc')
osd_files.sort()

osd_df = pd.DataFrame()

data = open_dataset(osd_files[0])

for i in trange(len(osd_files)):
    df_add = nodc_to_vvd0(open_dataset(osd_files[i]))
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
            'value_vs_depth\\MEDS_Oxy_1991_1995_value_vs_depth_0.csv'

df_meds_vvd0.to_csv(vvd0_name, index=False)

# Concatenate all dataframes together
vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\'
files = glob.glob(vvd_dir + '*.csv')

df_all = pd.DataFrame()
for f in files:
    df_add = pd.read_csv(f)
    df_all = pd.concat([df_all, df_add])

all_name = 'ALL_Oxy_1991_2020_value_vs_depth.csv'

df_all.to_csv(vvd_dir + all_name, index=False)
