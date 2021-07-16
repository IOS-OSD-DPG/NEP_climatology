# Export all oxygen climatology data to common csv format: Profile data table
# Sources: IOS, NODC, MEDS

from xarray import open_dataset
import pandas as pd
import numpy as np
import glob
from os.path import basename
import datetime

# Find all oxygen data
ios_path = '/home/hourstonh/Documents/climatology/data/IOS_CIOOS/'
ios_files = glob.glob(ios_path + '*Oxy*.nc', recursive=False)
ios_files.sort()

# nodc_nocad_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\data\\WOD_extracts\\Oxy_WOD_May2021_extracts\\'
nodc_nocad_path = '/home/hourstonh/Documents/climatology/data/WOD_extracts/Oxy_WOD_May2021_extracts/'
nodc_nocad_files = glob.glob(nodc_nocad_path + '*.nc', recursive=False)
nodc_nocad_files.sort()

# nodc_cad_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\data\\WOD_extracts\\WOD_July_CDN_nonIOS_extracts\\'
nodc_cad_path = '/home/hourstonh/Documents/climatology/data/WOD_extracts/WOD_July_CDN_nonIOS_extracts/'
nodc_cad_files = glob.glob(nodc_cad_path + 'Oxy*.nc', recursive=False)
nodc_cad_files.sort()

meds_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\data\\meds_data_extracts\\bo_extracts\\'
meds_files = glob.glob(meds_path + '*DOXY*.csv', recursive=False)
meds_files.sort()


# DataFrame columns
df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
           "Instrument_type", "Date_string", "Latitude",
           "Longitude", "Quality_control_flag"]


ios_df = pd.DataFrame(columns=df_cols)


# Open IOS files
for i, f in enumerate(ios_files):
    print(i, f)
    ios_data = open_dataset(f)
    
    # Get unique profile indices to allow filtering through "row" dimension
    indices = np.unique(ios_data.profile.data, return_index=True)[1]
    
    ios_fname_array = np.repeat(basename(ios_files[0]), len(indices))
    ios_institute_array = np.repeat(ios_data.institution, len(indices))
    
    if 'CTD' in ios_files[0]:
        inst = 'CTD'
    elif 'BOT' in ios_files[0]:
        inst = 'BOT'
        
    ios_instrument_type_array = np.repeat(inst, len(indices))
    
    # Time strings: yyyymmddhhmmsszzz; slow to run
    ios_time_strings = pd.to_datetime(ios_data.time.data).strftime('%Y%m%d%H%M%S%z')
    
    # QC flags: good data by default, according to Germaine
    ios_flags = np.ones(len(indices))
    
    # Take transpose of arrays?
    ios_df_add = pd.DataFrame(
        data=np.array([ios_fname_array,
                       ios_institute_array,
                       ios_data.mission_id.data[indices],
                       ios_instrument_type_array,
                       ios_time_strings[indices],
                       ios_data.longitude.data[indices],
                       ios_data.latitude.data[indices],
                       ios_flags]).transpose(), columns=df_cols)
    
    ios_df = pd.concat([ios_df, ios_df_add])


ios_df_name = '/home/hourstonh/Documents/climatology/data_extracts/IOS_Profiles_Oxy_1991_2020.csv'
ios_df.to_csv(ios_df_name)

# Flag duplicates in the dataframe
ios_df_edr = ios_df.copy()
ios_df_edr['Duplicate_row'] = ios_df.duplicated()

# dr stands for duplicate rows
ios_df_edr_name = '/home/hourstonh/Documents/climatology/data_extracts/IOS_Profiles_Oxy_1991_2020_dr.csv'
ios_df_edr.to_csv(ios_df_edr_name)


### NODC WOD ###
def nodc_to_common_csv(nodc_files, sourcetype):
    colnames = ["Source_data_file_name", "Institute", "Cruise_number",
                "Instrument_type", "Date_string", "Latitude",
                "Longitude", "Quality_control_flag"]
    
    nodc_df = pd.DataFrame(columns=df_cols)
    
    for f in nodc_files:
        # Read in netCDF file
        nodc_nocad_data = open_dataset(f)
        
        # Casts is the dim counting the number of profiles
        nodc_nocad_fname_array = np.repeat(
            basename(nodc_nocad_files[0]), len(nodc_nocad_data.casts.data))
        
        # Make array of institute name
        nodc_nocad_institute_array = np.repeat(
            nodc_nocad_data.institution, len(nodc_nocad_data.casts.data))
        
        # Get instrument type from file name
        if 'CTD' in nodc_nocad_files[0]:
            inst = 'CTD'
        elif 'OSD' in nodc_nocad_files[0]:
            inst = 'BOT'
        
        nodc_nocad_instrument_array = np.repeat(inst, len(nodc_nocad_data.casts.data))
        
        # Convert time data to time string type
        nodc_nocad_timestring = pd.to_datetime(
            nodc_nocad_data.time.data).strftime('%Y%m%d%H%M%S%z')
        
        nodc_df_add = pd.DataFrame(
            data=np.array([nodc_nocad_fname_array,
                           nodc_nocad_institute_array,
                           nodc_nocad_data.WOD_cruise_identifier.data,
                           nodc_nocad_instrument_array,
                           nodc_nocad_timestring,
                           nodc_nocad_data.lat.data,
                           nodc_nocad_data.lon.data,
                           nodc_nocad_data.Oxygen_WODprofileflag.data]).transpose(),
            columns=colnames)
    
        # Append the new dataframe to the existing dataframe
        nodc_df = pd.concat([nodc_df, nodc_df_add],
                            ignore_index=True)
    
    print(nodc_df.columns)
    print(nodc_df)
    
    print(min(nodc_df['Date_string']), max(nodc_df['Date_string']))
    
    # Export to csv file
    output_folder = '/home/hourstonh/Documents/climatology/data_extracts/'
    nodc_name = 'NODC_{}_Profiles_Oxy_1991_2020.csv'.format(sourcetype)
    nodc_df.to_csv(output_folder + nodc_name)
    
    # Flag duplicate rows in the dataframe
    nodc_df_edr = nodc_df.copy()
    nodc_df_edr['Duplicate_row'] = nodc_df.duplicated()
    
    # dr stands for duplicate rows
    nodc_nocad_edr_name = nodc_name.replace('.', '_dr.')
    nodc_df_edr.to_csv(output_folder + nodc_nocad_edr_name)
    
    return


nodc_to_common_csv(nodc_nocad_files, sourcetype='noCAD')
nodc_to_common_csv(nodc_cad_files, sourcetype='CAD')


##### MEDS Data #####

# MEDS data: initialize empty dataframe
meds_df = pd.DataFrame(columns=df_cols)

meds_data = pd.read_csv(meds_files[0])

meds_data.head()

# Get number of unique profiles
unique = np.unique(meds_data.loc[:, 'RowNum'], return_index=True)[1]

# Oxy data spans 1991-01-22 05:13:00 to 1995-03-09 23:35:00
meds_fname_array = np.repeat('MEDS_ASCII_1991_2000.csv', len(unique))

# Get instrument from file name
if 'CTD' in meds_files[0]:
    inst = 'CTD'
elif 'BO' in meds_files[0]:
    inst = 'BOT'

meds_instrument_array = np.repeat(inst, len(unique))

# Time string data
meds_data['Hour'] = meds_data.Time.astype(str).apply(lambda x: x[:-2])
meds_data['Minute'] = meds_data.Time.astype(str).apply(lambda x: x[-2:])

meds_data['Timestring'] = pd.to_datetime(
    meds_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]).dt.strftime(
    '%Y%m%d%H%M%S%z')

meds_data['Time_pd'] = pd.to_datetime(
    meds_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])

print(min(meds_data['Time_pd']), max(meds_data['Time_pd']))

# # DataFrame columns
# df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
#            "Instrument_type", "Date_string", "Latitude",
#            "Longitude", "Quality_control_flag"]

meds_df_add = pd.DataFrame(
    data=np.array([meds_fname_array,
                   meds_data.loc[:, 'SourceID'][unique],
                   meds_data.loc[:, 'CruiseID'][unique],
                   meds_instrument_array,
                   meds_data.loc[:, 'Timestring'][unique],
                   meds_data.loc[:, 'Lat'][unique],
                   meds_data.loc[:, 'Lon'][unique],
                   meds_data.loc[:, 'PP_flag'][unique]]).transpose(),
    columns=df_cols
)

meds_df = pd.concat([meds_df, meds_df_add])

output_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\'

meds_csv_name = 'MEDS_Profiles_Oxy_1991_1995.csv'

meds_df.to_csv(output_folder + meds_csv_name)

# Flag duplicate rows
meds_df_edr = meds_df.copy()

meds_df_edr['Duplicate_row'] = meds_df.duplicated()

meds_df_edr_name = meds_csv_name.replace('.', '_dr.')

meds_df_edr.to_csv(output_folder + meds_df_edr_name)
