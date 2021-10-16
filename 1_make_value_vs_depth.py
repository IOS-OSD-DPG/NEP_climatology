# Create value vs depth tables
# Do not apply any flags or unit conversions just yet

import numpy as np
import pandas as pd
import glob
from xarray import open_dataset
# from copy import deepcopy
from tqdm import trange
from gsw import z_from_p, p_from_z, CT_from_t, SA_from_SP
from gsw.density import rho
from os.path import basename


# Start with IOS data
def ios_to_vvd0(ncdata, instrument='BOT', var='DOXMZZ01'):
    # Get index of first measurement of each profile
    # indexer = np.unique(ncdata.profile.data, return_index=True)[1]

    # Initialize empty dataframe
    df_out = pd.DataFrame()

    # Add profile number as a column
    unique = np.unique(ncdata.profile.data, return_index=True)[1]
    df_out['Profile_number'] = np.zeros(len(ncdata.profile.data), dtype=int)

    # print(len(unique), len(ncdata.mission_id.data))

    # Check that the variable is in ncdata
    try:
        var_values = ncdata[var].data
    except KeyError:
        print('Warning: Variable', var, 'not in dataset')
        return None

    num = 1
    # Skip the first profile since its number is already zero
    for j in range(1, len(unique)):
        if j == len(unique) - 1:
            end_prof_ind = None
        else:
            # Pandas indexes to inclusive end
            end_prof_ind = unique[j + 1] - 1
        df_out.loc[unique[j]:end_prof_ind, 'Profile_number'] = num
        num += 1

    print('Total number of profiles:', num + 1)  # Started from zero

    df_out['Cruise_number'] = ncdata.mission_id.data
    df_out['Instrument_type'] = np.repeat(instrument, len(df_out))  # To remove later
    df_out['Date_string'] = pd.to_datetime(ncdata.time.data).strftime('%Y%m%d%H%M%S')
    df_out['Latitude'] = ncdata.latitude.data
    df_out['Longitude'] = ncdata.longitude.data
    df_out['Depth_m'] = ncdata.depth.data
    df_out['Depth_flag'] = np.ones(len(ncdata.row), dtype=int)  # To remove later
    df_out['Value'] = var_values
    df_out['Source_flag'] = np.ones(len(ncdata.row), dtype=int)  # To remove later

    return df_out


def ios_wp_to_vvd0(nclist, var='DOXMZZ01'):
    # Put IOS Water Properties data to a value vs depth table

    df_out = pd.DataFrame()

    # Iterate through the list of netcdf file paths
    for j, ncfile in enumerate(nclist):
        # print(j, basename(ncfile))

        # Get instrument type
        if 'ctd' in ncfile:
            instrument_type = 'CTD'
        elif 'bot' in ncfile:
            instrument_type = 'BOT'

        # Open the netCDF file
        ncdata = open_dataset(ncfile)

        # print(basename(ncfile))
        # print(ncdata.data_vars)

        flag = 0

        # Convert oxygen data to umol/kg if not already done
        if var == 'DOXMZZ01':
            try:
                var_values = ncdata[var].data
            except KeyError:
                # Convert data from mL/L to umol/kg
                print('Converting oxygen data from mL/L to umol/kg')
                try:
                    var_values = mL_L_to_umol_kg(ncdata.DOXYZZ01.data)
                except AttributeError:
                    print('Warning: Variable DOXYZZ01 not present in file',
                          basename(ncfile))
                    flag += 1
        elif var == 'TEMPS901' or var == 'PSALST01':
            # Need unit conversions?
            try:
                var_values = ncdata[var].data
            except KeyError:
                print('Warning: Variable', var, 'not available in file',
                      basename(ncfile))
                # Want to skip to next iteration
                flag += 1

        if flag == 1:
            # Skip to next iteration
            continue

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
        df_add['Value'] = var_values
        df_add['Source_flag'] = np.ones(len(ncdata.depth.data), dtype=int)

        # Concatenate to df_out
        df_out = pd.concat([df_out, df_add])

    return df_out


# NODC data
def nodc_to_vvd0(ncdata, instrument='BOT', var='Oxygen', counter=0):
    # Transfer NODC data to value vs depth format
    # Add duplicate flags at a later time
    # var: Oxygen, Salinity, Temperature

    df_out = pd.DataFrame()

    profile_number = np.zeros(len(ncdata[var].data), dtype=int)
    cruise_number = np.repeat('XXXXXXXX', len(ncdata[var].data))
    date_string = np.repeat('YYYYMMDDhhmmss', len(ncdata[var].data))
    latitude = np.repeat(0., len(ncdata[var].data))
    longitude = np.repeat(0., len(ncdata[var].data))

    start_ind = 0
    for i in range(len(ncdata['{}_row_size'.format(var)].data)):
        end_ind = start_ind + int(ncdata['{}_row_size'.format(var)].data[i])

        profile_number[start_ind: end_ind] = counter

        # Need .astype(str) to get rid of the b'' chars
        cruise_number[start_ind: end_ind
                      ] = ncdata.WOD_cruise_identifier.data[i].astype(str)

        date_string[start_ind: end_ind
                    ] = pd.to_datetime(ncdata.time.data[i]).strftime('%Y%m%d%H%M%S')

        latitude[start_ind: end_ind] = ncdata.lat.data[i].astype(float)

        longitude[start_ind: end_ind] = ncdata.lon.data[i].astype(float)

        counter += 1
        start_ind = end_ind

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
    df_out['Value'] = ncdata[var].data
    df_out['Source_flag'] = ncdata['{}_WODflag'.format(var)].data

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


def meds_to_vvd0(df_meds, instrument='BOT', var='DOXY', counter=0):
    # Just convert to value-vs-depth format without adding duplicate flags
    # Add duplicate flags at a later step

    # Add profile number counter
    df_meds['Profile_number'] = np.zeros(len(df_meds), dtype=int)

    unique = np.unique(df_meds.RowNum, return_index=True)[1]

    for i in range(len(unique)):
        if i == len(unique) - 1:
            end_ind = None
        else:
            end_ind = unique[i + 1]
        df_meds.loc[unique[i]:end_ind, 'Profile_number'] = counter
        counter += 1

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
    if var == 'DOXY':
        df_meds['Value_out'] = mmol_m3_to_umol_kg(df_meds['DOXY'], df_meds['PSAL'],
                                                  df_meds['TEMP'], df_meds['Press_dbar'],
                                                  df_meds['Lat'], df_meds['Lon'])
        df_meds['Value_flag'] = df_meds['{}_flag'.format(var)]
    elif var == 'TEMP' or var == 'PSAL':
        # Units are degrees Celsius
        df_meds['Value_out'] = df_meds['ProfParm']
        df_meds['Value_flag'] = df_meds['PP_flag']

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
    df_out['Value'] = df_meds['Value_out']
    df_out['Source_flag'] = df_meds['Value_flag']

    return df_out, counter


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
           'IOS_CIOOS\\IOS_BOT_Profiles_Sal_19910101_20201231.nc'
ios_data = open_dataset(ios_file)

# TEMPS901, PSALST01, DOXMZZ01
ios_df = ios_to_vvd0(ios_data, instrument='BOT', var='PSALST01')

ios_df_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\1_original\\' \
              'IOS_BOT_Sal_1991_2020_value_vs_depth_0.csv'

ios_df.to_csv(ios_df_name, index=False)

# CTD data
ios_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'source_format\\IOS_CIOOS\\'
# ios_files = glob.glob(ios_dir + 'IOS_CTD_Profiles_Temp*.nc')
ios_files = glob.glob(ios_dir + 'IOS_CTD_Profiles_Sal*.nc')
ios_files.sort()
print(len(ios_files))

# ios_ctd_df = pd.DataFrame()

years = [(1991, 1995), (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015),
         (2015, 2020)]

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\1_original\\'

for i in trange(len(ios_files)):
    ncin = open_dataset(ios_files[i])
    df_add = ios_to_vvd0(ncin, instrument='CTD', var='PSALST01')  # 'DOXMZZ01'
    fname = 'IOS_CTD_Sal_{}_{}_value_vs_depth_0.csv'.format(
        years[i][0], years[i][1])

    df_add.to_csv(outdir + fname, index=False)


# ios_ctd_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#                'value_vs_depth\\IOS_CTD_Oxy_1991_2020_value_vs_depth_0.csv'

# ios_ctd_df.to_csv(ios_ctd_name, index=False)


########################
# IOS Water Properties data

wp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'source_format\\SHuntington\\'

wp_list = glob.glob(wp_dir + 'WP_unique_CTD_forHana\\*.ctd.nc', recursive=False)

wp_list += glob.glob(wp_dir + '*.bot.nc', recursive=False)
print(len(wp_list))

# nc = open_dataset(wp_list[0])

# vars: TEMPS901, DOXMZZ01, PSALST01
df = ios_wp_to_vvd0(wp_list, var='PSALST01')

outname = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
          '1_original\\IOS_WP_Sal_1991_2020_value_vs_depth_0.csv'

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
    df_add, prof_count_new = nodc_to_vvd0(open_dataset(osd_files[i]),
                                          counter=prof_count_old)
    prof_count_old = prof_count_new
    osd_df = pd.concat([osd_df, df_add])

osd_df_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\1_original\\WOD_BOT_Oxy_1991_2020_value_vs_depth_0.csv'

osd_df.to_csv(osd_df_name, index=False)

# NODC PFL, GLD, CTD, OSD and DRB files for TS data (NOT Oxy)
ts_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'source_format\\WOD_extracts\\'

# Assemble files
wod_var = 'Oxy'  # Oxy, Temp, Sal
osd_files = glob.glob(ts_dir + 'WOD_July_extracts\\{}*OSD.nc'.format(wod_var))
osd_files += glob.glob(ts_dir + 'WOD_July_CDN_nonIOS_extracts\\{}*OSD.nc'.format(wod_var))

ctd_files = glob.glob(ts_dir + 'WOD_July_extracts\\{}*CTD.nc'.format(wod_var))
ctd_files += glob.glob(ts_dir + 'WOD_July_CDN_nonIOS_extracts\\{}*CTD.nc'.format(wod_var))

drb_files = glob.glob(ts_dir + 'WOD_July_extracts\\{}*DRB.nc'.format(wod_var))
drb_files += glob.glob(ts_dir + 'WOD_July_CDN_nonIOS_extracts\\{}*DRB.nc'.format(wod_var))  # empty

pfl_files = glob.glob(ts_dir + 'WOD_July_extracts\\{}*PFL.nc'.format(wod_var))
pfl_files += glob.glob(ts_dir + 'WOD_July_CDN_nonIOS_extracts\\{}*PFL.nc'.format(wod_var))  # empty

pfl_files = glob.glob(ts_dir + 'Oxy_WOD_May2021_extracts\\{}*PFL.nc'.format(wod_var))

gld_files = glob.glob(ts_dir + 'WOD_July_extracts\\{}*GLD.nc'.format(wod_var))
gld_files += glob.glob(ts_dir + 'WOD_July_CDN_nonIOS_extracts\\{}*GLD.nc'.format(wod_var))

out_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\1_original\\'

# DO PFL SEPARATELY BECAUSE ITS BIG
inst_names = ['OSD', 'CTD', 'DRB', 'GLD']
inst_list = [osd_files, ctd_files, drb_files, gld_files]

# Convert 'OSD', 'CTD', 'DRB', 'GLD' to vvd format
for j in trange(len(inst_list)):
    prof_count_old = 0
    inst = inst_names[j]
    inst_df = pd.DataFrame()

    # Iterate through all files in each list
    for ncfile in inst_list[j]:
        data = open_dataset(ncfile)
        df_add, prof_count_new = nodc_to_vvd0(data, instrument=inst,
                                              var='Salinity',
                                              counter=prof_count_old)
        prof_count_old = prof_count_new
        inst_df = pd.concat([inst_df, df_add])

    # Export df to csv file
    out_name = 'WOD_{}_{}_1991_2020_value_vs_depth_0.csv'.format(inst, wod_var)

    inst_df.to_csv(out_dir + out_name, index=False)

    # continue

# Convert PFL to vvd format separately
for f in pfl_files:
    # Index the months the file covers from the file name
    months = basename(f)[-10:-7]
    inst = 'PFL'
    data = open_dataset(f)
    # Oxygen, Salinity, Temperature
    df_out = nodc_to_vvd0(data, instrument=inst, var='Oxygen',
                          counter=0)[0]
    out_name = 'WOD_{}_{}_{}_1991_2020_value_vs_depth_0.csv'.format(inst, wod_var,
                                                                    months)

    df_out.to_csv(out_dir + out_name, index=False)

    # continue

########################
# MEDS data

# Start with O data
meds_extracts_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                    'source_format\\meds_data_extracts\\'

meds_TSO_file = meds_extracts_dir + \
                'bo_extracts\\MEDS_19940804_19930816_BO_TSO_profiles_source.csv'

meds_data = pd.read_csv(meds_TSO_file)

# df_meds_vvd = meds_to_vvd(meds_data, pdt)
df_meds_vvd0, count = meds_to_vvd0(meds_data)

# Write output dataframe to csv file
vvd0_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\MEDS_BOT_Oxy_1991_1995_value_vs_depth_0.csv'

df_meds_vvd0.to_csv(vvd0_name, index=False)

# TS data: BO, CD, XB (T only)

meds_T_flist = glob.glob(meds_extracts_dir + '*_extracts\\*TEMP_profiles_source.csv')
meds_S_flist = glob.glob(meds_extracts_dir + '*_extracts\\*PSAL_profiles_source.csv')

# Initialize dataframes
df_T = pd.DataFrame()
df_S = pd.DataFrame()

# Initialize profile number counter
prof_count_old = 0

for f in meds_S_flist:
    # Get instrument and var type
    inst = basename(f)[23:25]
    if inst == 'BO':
        inst = 'BOT'
    elif inst == 'CD':  # CTD downcast
        inst = 'CTD'
    elif inst == 'XB':
        inst = 'XBT'
    # Get variable abbreviation
    meds_var = basename(f)[26:30]
    df_in = pd.read_csv(f)

    df_add, prof_count_new = meds_to_vvd0(df_in, instrument=inst, var=meds_var,
                                          counter=prof_count_old)
    df_S = pd.concat([df_S, df_add])

    prof_count_old = prof_count_new

out_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\1_original\\'

out_name = 'MEDS_{}_1991_2020_value_vs_depth_0.csv'.format(meds_var)

df_S.to_csv(out_dir + out_name, index=False)

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