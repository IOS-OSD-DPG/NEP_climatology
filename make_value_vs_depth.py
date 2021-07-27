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


# Import all data; no args to pass
fname_dict = get_filenames_dict()

# Open the flags dataset (profile data table, or PDT) from the previous step
df_pdt_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
                'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_001ll_check2.csv'

pdt = pd.read_csv(df_pdt_file)

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

# Open the raw data files and apply the flags


# Start with IOS data
def ios_to_vvd(ncdata, df_pdt):

    # Get index of first measurement of each profile
    indexer = np.unique(ncdata.profile.data, return_index=True)[1]

    dict_list = []

    # Iterate through nc data

    df_out = pd.DataFrame.from_dict(dict_list)

    return df_out


# NODC data
def nodc_to_vvd(ncdata, df_pdt):
    # Make sure to correct the format of cruise IDs (remove b')

    # Get first index of each profile in ncdata.Oxygen.data
    prof_start_indices = np.concatenate(
        [np.zeros(1, dtype=int), np.cumsum(ncdata.Oxygen_row_size.data,
                                           dtype='int')[:-1]])

    dict_list = []

    for i in trange(len(prof_start_indices)):
        cruise_nodc = ncdata.WOD_cruise_identifier.data[i].astype(str)
        time_nodc = pd.to_datetime(ncdata.time.data[i]).dt.strftime('%Y%m%d%H%M%S')
        lat_nodc = ncdata.lat.data[i]
        lon_nodc = ncdata.lon.data[i]

        indexer = np.where((df_pdt.Cruise_number == cruise_nodc) &
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
        df_meds[['Year', 'Month', 'Day', 'Hour', 'Minute']]).dt.strftime(
        '%Y%m%d%H%M%S')

    print(np.where(pd.isnull(df_meds.Timestring)))

    # Unit conversions for depth/pressure
    df_meds['Depth_m'] = df_meds['Depth/Press']
    pressure_subsetter = np.where((df_meds.loc[:, 'D_P_code'] == 'P').values)[0]
    # Not sure if df format is ok for z_from_p or if array type is required
    df_meds.loc[pressure_subsetter, 'Depth_m'] = z_from_p(
        df_meds.loc[pressure_subsetter, 'Depth_m'].values,
        df_meds.loc[pressure_subsetter, 'Lat'].values)

    # Remove Depth/Press and D_P_code after unit conversions done
    # Remove D_P_flag after source flagging is done
    vvd_cols = ['CruiseID', 'Date_string', 'Latitude', 'Longitude', 'Depth/Press',
                'D_P_code', 'D_P_flag', 'Value', 'Source_flag']
    # df_out = pd.DataFrame(columns=vvd_cols)

    # Iterate through MEDS dataframe
    dict_list = []
    for i in trange(len(df_meds)):

        # Find the duplicate flags in df_pdt
        cruise_meds = df_meds.loc[i, 'CruiseID']
        time_meds = df_meds.loc[i, 'Date_string']
        lat_meds = df_meds.loc[i, 'Latitude']
        lon_meds = df_meds.loc[i, 'Longitude']

        # Only use bottle data from MEDS so index by instrument type too
        indexer = np.where((df_pdt.Cruise_number == cruise_meds) &
                           (df_pdt.Date_string == time_meds) &
                           (df_pdt.Latitude == lat_meds) &
                           (df_pdt.Longitude == lon_meds) &
                           (df_pdt.Instrument_type == 'BOT'))[0]

        if len(indexer) > 1:
            print('Warning: More than one row match for finding duplicate flags')

        ex_dup_flag = df_pdt.loc[indexer[0], 'Exact_duplicate_row']
        cb_dup_flag = df_pdt.loc[indexer[0], 'CTD_BOT_duplicate_row']
        ie_dup_flag = df_pdt.loc[indexer[0], 'Inexact_duplicate_check2']

        dict_list.append({'CruiseID': df_meds.loc[i, 'CruiseID'],
                          'Date_string': df_meds.loc[i, 'Date_string'],
                          'Latitude': df_meds.loc[i, 'Lat'],
                          'Longitude': df_meds.loc[i, 'Lon'],
                          'Depth_m': df_meds.loc[i, 'Depth_m'],
                          'Depth_flag': df_meds.loc[i, 'D_P_flag'],
                          'Value': df_meds.loc[i, 'ProfParm'],
                          'Source_flag': df_meds.loc[i, 'PP_flag'],
                          'Exact_duplicate_flag': ex_dup_flag,
                          'CTD_BOT_duplicate_flag': cb_dup_flag,
                          'Inexact_duplicate_flag': ie_dup_flag})

    # Convert list to dataframe
    df_out = pd.DataFrame.from_dict(dict_list)

    return df_out


# Iterate through files in dictionary
for key in fname_dict.keys():
    data = open_by_source(fname_dict[key])

    if 'IOS' in key:
        ios_to_vvd(data, pdt)
    elif key.startswith('Oxy'):
        nodc_to_vvd(data, pdt)
    elif 'MEDS' in key:
        meds_to_vvd(data, pdt)

print('Done')

