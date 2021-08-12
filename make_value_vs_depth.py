# Create value vs depth tables
# Do not apply any flags or unit conversions just yet

import numpy as np
import pandas as pd
import glob
from xarray import open_dataset
from copy import deepcopy
from tqdm import trange
from gsw import z_from_p, p_from_z, CT_from_t, SA_from_SP
from gsw.density import rho


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


def ios_wp_to_vvd0(nclist):
    # Put IOS Water Properties data to a value vs depth table

    df_out = pd.DataFrame()

    # Iterate through the list of netcdf file paths
    for j, ncfile in enumerate(nclist):
        # Get instrument type
        if 'ctd' in ncfile:
            instrument_type = 'CTD'
        elif 'bot' in ncfile:
            instrument_type = 'BOT'

        # Open the netCDF file
        ncdata = open_dataset(ncfile)

        # Convert oxygen data to umol/kg if not already done
        try:
            oxygen = ncdata.DOXMZZ01.data
        except AttributeError:
            # Convert data from mL/L to umol/kg
            print('Converting oxygen data from mL/L to umol/kg')
            oxygen = mL_L_to_umol_kg(ncdata.DOXYZZ01)

        # Initialize dataframe to concatenate to df_out
        df_add = pd.DataFrame()
        # Populate the dataframe
        df_add['Profile_number'] = np.repeat(j, len(ncdata.depth.data))
        df_add['Cruise_number'] = np.repeat(ncdata.mission_id.data, len(ncdata.depth.data))
        df_add['Instrument_type'] = np.repeat(instrument_type, len(ncdata.depth.data))
        df_add['Date_string'] = np.repeat(pd.to_datetime(ncdata.time.data).strftime('%Y%m%d%H%M%S'),
                                          len(ncdata.depth.data))
        df_add['Latitude'] = np.repeat(ncdata.latitude.data, len(ncdata.depth.data))
        df_add['Longitude'] = np.repeat(ncdata.longitude.data, len(ncdata.depth.data))
        df_add['Depth_m'] = ncdata.depth.data
        df_add['Depth_flag'] = np.ones(len(ncdata.depth.data), dtype=int)
        df_add['Value'] = oxygen
        df_add['Source_flag'] = np.ones(len(ncdata.depth.data), dtype=int)

        # Concatenate to df_out
        df_out = pd.concat([df_out, df_add])

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


def mL_L_to_umol_kg(oxygen):
    # Oxygen in mL/L
    # Applies to some IOS Water Properties data

    mol_to_umol = 1e6

    # Molar mass of O2
    mm_O2 = 2 * 15.9994  # g/mol

    # Convert mL/L to L/L to kg/kg to g/kg to mol/kg to umol/kg
    oxygen_out = oxygen * mm_O2 * mol_to_umol

    return oxygen_out


def mmol_m3_to_umol_kg(oxygen, prac_sal, temp, press, lat, lon):
    # Oxygen in millimol/m^3
    # Applies to MEDS oxygen data

    mmol_to_umol = 1e3

    # Convert pressure to SP: Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    SP = press - 10.1325

    # Convert practical salinity to SA: Absolute salinity, g/kg
    # prac_sal is practical salinity unit (PSS-78)
    SA = SA_from_SP(prac_sal, SP, lon, lat)

    # Convert temperature to CT: Conservative Temperature (ITS-90), degrees C
    # temp parameter should be In-situ temperature (ITS-90), degrees C
    CT = CT_from_t(SA, temp, SP)

    # Calculate the in-situ density of seawater in kg/m^3
    insitu_density = rho(SA, CT, SP)

    # Convert mmol/m^3 to umol/m^3 to umol/kg
    oxygen_out = oxygen * mmol_to_umol / insitu_density

    return oxygen_out


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

    # Calculate pressure
    df_meds['Press_dbar'] = df_meds['Depth/Press']
    df_meds.loc[~pressure_subsetter, 'Press_dbar'] = p_from_z(
        df_meds.loc[~pressure_subsetter, 'Depth_m'].values,
        df_meds.loc[~pressure_subsetter, 'Lat'].values)

    # Unit conversions for oxygen from millimol/m^3 to umol/kg
    df_meds['Oxygen_umol'] = mmol_m3_to_umol_kg(df_meds['DOXY'], df_meds['PSAL'],
                                                df_meds['TEMP'], df_meds['Press_dbar'],
                                                df_meds['Lat'], df_meds['Lon'])

    # Write to dataframe to output
    df_out = pd.DataFrame()
    df_out['Profile_number'] = df_meds['Profile_number']
    df_out['Cruise_number'] = df_meds['CruiseID']
    df_out['Instrument_type'] = np.repeat(instrument, len(df_out))  # To remove later
    df_out['Date_string'] = df_meds['Date_string']
    df_out['Latitude'] = df_meds['Lat']
    df_out['Longitude'] = -df_meds['Lon']  # Convert to positive East
    df_out['Depth_m'] = df_meds['Depth_m']
    df_out['Depth_flag'] = df_meds['D_P_flag']
    df_out['Value'] = df_meds['Oxygen_umol']
    df_out['Source_flag'] = df_meds['DOXY_flag']

    return df_out


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

# ios_ctd_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#                'value_vs_depth\\IOS_CTD_Oxy_1991_2020_value_vs_depth_0.csv'

# ios_ctd_df.to_csv(ios_ctd_name, index=False)


########################
# IOS Water Properties data

wp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\SHuntington\\'

wp_list = glob.glob(wp_dir + 'WP_unique_CTD_forHana\\*.ctd.nc', recursive=False)

wp_list += glob.glob(wp_dir + '*.bot.nc', recursive=False)

# nc = open_dataset(wp_list[0])

df = ios_wp_to_vvd0(wp_list)

outname = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
          'IOS_WP_Oxy_1991_2020_value_vs_depth_0.csv'

df.to_csv(outname, index=False)

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
# MEDS data

meds_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
            'meds_data_extracts\\bo_extracts\\MEDS_19940804_19930816_BO_TSO_profiles_source.csv'

meds_data = pd.read_csv(meds_file)

# df_meds_vvd = meds_to_vvd(meds_data, pdt)
df_meds_vvd0 = meds_to_vvd0(meds_data)

# Write output dataframe to csv file
vvd0_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\MEDS_BOT_Oxy_1991_1995_value_vs_depth_0.csv'

df_meds_vvd0.to_csv(vvd0_name, index=False)


#####################################
# Concatenate all dataframes together OR NOT BC OF SIZE ISSUES
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