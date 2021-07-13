# Export all oxygen climatology data to common csv format
# Sources: IOS, NODC, MEDS

from xarray import open_dataset
from pandas import to_datetime, DataFrame, concat
import numpy as np
import glob
from os.path import basename

# Find all oxygen data
ios_path = '/home/hourstonh/Documents/climatology/data/IOS_CIOOS/'
ios_files = glob.glob(ios_path + '*Oxy*.nc', recursive=False)
ios_files.sort()

nodc_nocad_path = '/home/hourstonh/Documents/climatology/data/WOD_extracts/Oxy_WOD_May2021_extracts/'
nodc_oxy_files = glob.glob(nodc_nocad_path + '*.nc', recursive=False)

nodc_cad_path = '/home/hourstonh/Documents/climatology/data/WOD_extracts/WOD_July_CDN_nonIOS_extracts/'
nodc_cad_files = glob.glob(nodc_cad_path + 'Oxy*.nc', recursive=False)

meds_path = '/home/hourstonh/Documents/climatology/data_explore/MEDS/raw_extracts/'
meds_files = glob.glob(meds_path + '*DOXY*.csv', recursive=False)


# DataFrame columns
df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
           "Instrument_type", "Date_string", "Latitude",
           "Longitude", "Quality_control_flag"]


ios_df = DataFrame(columns=df_cols)


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
    ios_time_strings = to_datetime(ios_data.time.data).strftime('%Y%m%d%H%M%S%z')
    
    # QC flags: good data by default, according to Germaine
    ios_flags = np.ones(len(indices))
    
    # Take transpose of arrays?
    ios_df_add = DataFrame(
        data=np.array([ios_fname_array,
                       ios_institute_array,
                       ios_data.mission_id.data[indices],
                       ios_instrument_type_array,
                       ios_time_strings[indices],
                       ios_data.longitude.data[indices],
                       ios_data.latitude.data[indices],
                       ios_flags]).transpose(), columns=df_cols)
    
    ios_df = concat([ios_df, ios_df_add])


ios_df.shape

ios_df_name = '/home/hourstonh/Documents/climatology/data_extracts/IOS_Profiles_Oxy_1991_2020.csv'
ios_df.to_csv(ios_df_name)

# Flag duplicates in the dataframe
ios_df_edr = ios_df.copy()
ios_df_edr['Duplicate_row'] = ios_df.duplicated()

# df stands for duplicates flagged
ios_df_edr_name = '/home/hourstonh/Documents/climatology/data_extracts/IOS_Profiles_Oxy_1991_2020_df.csv'
ios_df_edr.to_csv(ios_df_edr_name)

# NODC WOD
