"""
Script for checking for duplicate data within assembled data for NEP climatology.

Check within each data source separately first
Then check across different data sources next

Check for:
1. Stations with exact position/date/time values
2. Offsets in position/date/time

Also do statistical checks here (e.g., gradient, standard deviation)?
"""

import glob
import os
from tqdm import trange
from pandas import read_csv #, to_datetime
from xarray import open_dataset, Dataset
import numpy as np


def meds_check_dupl(meds_data):
    # Check a MEDS pandas dataframe for duplicate rows
    # Return a 1D dataframe (1 column) of booleans
    # indicating which rows are duplicates
    
    # Cut out first column
    cols2check = meds_data.columns.values.tolist()[1:]
    duplicates = meds_data.duplicated(subset=cols2check)
    
    return duplicates


def similar_lat(r1, r2, offset):
    # r1 and r2 are pandas dataframe rows
    return abs(r1['Lat'] - r2['Lat']) < offset


def similar_lon(r1, r2, offset=0.2):
    # r1 and r2 are pandas dataframe rows
    return abs(r1['Lon'] - r2['Lon']) < offset


def similar_date(r1, r2, offset=1):
    # r1 and r2 are pandas dataframe rows
    # Check if the days are close together
    # offset an integer of hours
    same_year = r1['Year'] == r2['Year']
    same_month = r1['Month'] == r2['Month']
    similar_day = abs(r1['Day'] - r2['Day']) < offset
    return same_year and same_month and similar_day


def similar_time(r1, r2, offset):
    # r1 and r2 are pandas dataframe rows
    return abs(r1['Time'] - r2['Time']) < offset


def meds_check_inexact_dupl(mdf):
    # mdf: pandas dataframe object
    
    # Check for inexact duplicates in the MEDS data
    # Check for offsets in position/date/time
    position_offset = 0.2 # 0.2 degrees latitude or longitude
    date_offset = None
    time_offset = 200 # two hours; time in format HHMM
    
    # Iterate through all the rows in the dataframe
    for r1 in range(len(mdf)):
        for r2 in range(r1, len(mdf)):
            # Check position
            if similar_lat(mdf.iloc[r1], mdf.iloc[r2], position_offset
                           ) and similar_lon(mdf.iloc[r1], mdf.iloc[r2], position_offset):
                # Flag in df
                pass
            # Check date
            if similar_date(mdf.iloc[r1], mdf.iloc[r2], date_offset):
                # Flag in df
                pass
            # Check time
            if similar_time(mdf.iloc[r1], mdf.iloc[r2], time_offset):
                # Flag in df
                pass
    
    return


def meds_run_check():
    # Check MEDS data
    # Find all files
    # Change path based on what instrument extracts: bo, xb or cd
    meds_fpath = '/home/hourstonh/Documents/climatology/data_explore/MEDS/cd_extracts/'
    meds_files = glob.glob(meds_fpath + '*_qc1.csv', recursive=False)
    meds_files.sort()
    
    # # Open files and read through them
    # df = read_csv(meds_files[1], skiprows=None, chunksize=None)
    # meds_check_dupl(df)
    
    # Iterate through the rows in the dataframe
    # Check for exact duplicates
    for j in trange(len(meds_files)):
        # print(meds_files[i])
        df = read_csv(meds_files[j], skiprows=None, chunksize=None)
        # dup_df_list.append(meds_check_dupl(df))
        # Remove duplicates in a new copy of the df
        # "edr" stands for "exact duplicates removed"
        # The "-" sign means "not-duplicates"
        df_edr = df[-meds_check_dupl(df)]
        # Export the new df in csv format
        outbasename = os.path.basename(meds_files[j])[:-4]
        outname = meds_fpath + outbasename + '_edr.csv'
        print(outname)
        df_edr.to_csv(outname, index=False)
        
    return
    

meds_run_check()

# Check for almost-exact duplicates
meds_fpath = '/home/hourstonh/Documents/climatology/data_explore/oxygen/MEDS/bo_profile_extracts/'
meds_files = glob.glob(meds_fpath + '*.csv', recursive=False)
meds_files.sort()

dup_df_list = []
# Iterate through files
for i in trange(len(meds_files)):
    # print(meds_files[i])
    df = read_csv(meds_files[i], skiprows=None, chunksize=None)
    dup_df_list.append(meds_check_inexact_dupl(df))


def wod_check_dupl(ncdata):
    # Check a netCDF file for duplicates
    
    # Use numpy.unique(array_object, axis=0) to get unique rows in a 2D array
    
    return


# Check WOD data
# Find all files
osddir = '/home/hourstonh/Documents/climatology/data/oxy_clim/WOD_extracts/Oxy_WOD_May2021_extracts/'
osdlist = glob.glob(osddir + '*OSD.nc')
osdlist.sort()

wod_cdn_dir = '/home/hourstonh/Documents/climatology/data/WOD_extracts/WOD_July_CDN_nonIOS_extracts/'
wod_cdn_list = glob.glob(wod_cdn_dir + 'Oxy*.nc')
wod_cdn_list.sort()

# Open netCDF file
odat = open_dataset(osdlist[0])

# Find duplicates
# Filter for exact position/date/time values
_, unique_indices = np.unique(np.array(odat.lat.data, odat.lon.data,
                                       odat.time.data), return_index=True)


def ios_check_dupl(ncfile, output_dir):
    # Check a netCDF file for duplicates
    # Flag duplicates then remove
    # Flag could just be a Python array not added to xarray object
    
    # Retrieve the base name of the input netCDF file
    in_nc_name = os.path.basename(ncfile)
    
    # Open the netCDF file with xarray
    ncdata = open_dataset(ncfile)
    
    # Initialize an array of flags for duplicates
    dup_flags = np.zeros(len(ncdata.time.data), dtype='int')
    
    # Index 1 to get the indices of all the unique values
    # Concern about inexactness, extra decimals that astype() might introduce
    # axis=1 to take unique columns
    # Number of columns = len(time)
    unique_tlld_ind = np.unique(
        np.array([ncdata.time.data.astype('float'), ncdata.latitude.data,
                  ncdata.longitude.data, ncdata.depth.data]),
        axis=1, return_index=True)[1]
    
    print(len(ncdata.time.data))
    print('time lat lon depth', len(unique_tlld_ind))
    
    print(ncdata.row.data)
    print(unique_tlld_ind)
    
    # Take set difference to find duplicate indices
    dup_flags[np.setdiff1d(ncdata.row.data, unique_tlld_ind)] = 1
    print('1', len(dup_flags[dup_flags == 1]))
    print('0', len(dup_flags[dup_flags == 0]))
    
    # Make new xarray object with duplicates removed to export as netcdf?
    outname = ios_make_nc(ncdata, dup_flags, in_nc_name, output_dir)
    
    return outname


def ios_make_nc(ncdata, flags, in_nc_name, output_dir):
    # One option is to remove all the variables and dimensions in a copy of ncdata
    
    # Length of the data_vars of the output file
    out_len = len(flags[flags == 0])
    
    # Remove the dimensions and associated variables
    # in a copy of the ncdata dataset
    outnc = ncdata.drop_dims(drop_dims=['row'])
    
    # Add back the dims and vars with duplicates removed
    outnc = outnc.expand_dims(dim={'row': np.arange(out_len, dtype='int')})
    
    # Drop 'row' as a variable
    outnc = outnc.drop_vars('row')
    
    # Need to add other _strlen dims?
    
    for varname, da in ncdata.data_vars.items():
        # Use double-asterisk kwargs syntax
        outnc = outnc.assign(**{varname: (('row'), da.data[flags == 0])})
        
    # Add attributes to each variable from the input netCDF dataset
    # Is there a way to do it with less code?
    # xr.Variable(dims, data, attrs=None)
    
    # Export the outnc object as a netCDF file
    # "edr" stands for "exact duplicates removed"
    # Alternative to file name change is to output in a new directory
    out_nc_name = output_dir + in_nc_name.replace('.nc', '_edr.nc')
    outnc.to_netcdf(out_nc_name, mode='w', format='NETCDF4')
    outnc.close()
    
    return out_nc_name


def ios_make_nc_v2(ncdata, flags, in_nc_name, output_dir):
    # The longer approach
    
    out = Dataset(coords={'row': ncdata.row.data[flags == 0],
                          'profile_strlen': len(ncdata.profile.data[0]),
                          'filename_strlen': len(ncdata.filename.data[0]),
                          'country_strlen': len(ncdata.country.data[0]),
                          'mission_id_strlen': len(ncdata.mission_id.data[0]),
                          'scientist_strlen': len(ncdata.scientist.data[0]),
                          'project_strlen': len(ncdata.project.data[0]),
                          'agency_strlen': len(ncdata.agency.data[0]),
                          'platform_strlen': len(ncdata.platform.data[0]),
                          'instrument_type_strlen': len(ncdata.instrument_type.data[0]),
                          'instrument_model_strlen': len(ncdata.instrument_model.data[0]),
                          'instrument_serial_number_strlen': len(ncdata.instrument_serial_number.data[0]),
                          'geographic_area_strlen': len(ncdata.geographic_area.data[0]),
                          'event_number_strlen': len(ncdata.event_number.data[0])},
                  data_vars={'profile': (['row', 'profile_strlen'], ncdata.profile.data[flags == 0]),
                             'filename': (['row', 'filename_strlen'], ncdata.filename.data[flags == 0]),
                             'country': (['row', 'country_strlen'], ncdata.country.data[flags == 0]),
                             'mission_id': (['row', 'mission_id_strlen'], ncdata.mission_id.data[flags == 0]),
                             'scientist': (['row', 'scientist_strlen'], ncdata.scientist.data[flags == 0]),
                             'project': (['row', 'project_strlen'], ncdata.project.data[flags == 0]),
                             'agency': (['row', 'agency_strlen'], ncdata.agency.data[flags == 0]),
                             'platform': (['row', 'platform_strlen'], ncdata.platform.data[flags == 0]),
                             'instrument_type': (['row', 'instrument_strlen'], ncdata.instrument_type.data[flags == 0]),
                             'instrument_model': (['row', 'instrument_model_strlen'], ncdata.instrument_model.data[flags == 0]),
                             'instrument_serial_number': (['row', 'instrument_serial_number_strlen'], ncdata.instrument_serial_number.data[flags == 0]),
                             'geographic_area': (['row', 'geographic_area_strlen'], ncdata.geographic_area.data[flags == 0]),
                             'event_number': (['row', 'event_number_strlen'], ncdata.event_number.data[flags == 0]),
                             }
                  )
    
    # Assign variable-specific attributes to variables
    # Need to reset "actual_range" attrs....
    for varname in outnc.data_vars.keys():
        for key, value in ncdata[varname].attrs:
            outnc[varname].attrs[key] = value
    
    # Assign global attributes from input netCDF data
    for key, value in ncdata.attrs:
        outnc.attrs[key] = value
        
    # Export to netCDF file
    out_nc_name = output_dir + in_nc_name.replace('.nc', '_edr.nc')
    out.to_netcdf(out_nc_name, mode='w', format='NETCDF4')
    out.close()
    
    return out_nc_name


# TESTING
# Check IOS data
dest_dir = '/home/hourstonh/Documents/climatology/data_explore/IOS/BOT/edr/'

# Find all files
fname = '/home/hourstonh/Documents/climatology/data/IOS_CIOOS/IOS_BOT_Profiles_Oxy_19910101_20201231.nc'

ios_check_dupl(fname, dest_dir)


# CTD OXY BATCH RUN
in_dir = '/home/hourstonh/Documents/climatology/data/IOS_CIOOS/'
dest_dir = '/home/hourstonh/Documents/climatology/data_explore/IOS/PCTD/edr/'
bot_oxy = glob.glob(in_dir + 'IOS_CTD_Profiles_Oxy*.nc', recursive=False)
bot_oxy.sort()

for i in trange(len(bot_oxy)):
    ios_check_dupl(bot_oxy[i], dest_dir)


nc = open_dataset(fname)
u, ind = np.unique(np.array([nc.time.data.astype('float'), nc.latitude.data,
                             nc.longitude.data, nc.depth.data]),
                   axis=1, return_index=True)

outnc = nc.copy(deep=False, data=None)
# outnc.dims
outnc = outnc.drop_dims(drop_dims='row') # vars and dims are now empty

outnc.expand_dims(dim={'profile_strlen2': 22})

# Can't simply replace variable data where new data is shorter
# Get ValueError: replacement data must match the Variable's shape.
# ValueError: Cannot assign to the .data attribute of dimension
# coordinate a.k.a IndexVariable 'row'. Please use
# DataArray.assign_coords, Dataset.assign_coords or Dataset.assign
# as appropriate.

# now add back variables with duplicates removed
# Use expand_dims() instead of assign_coords
# nc doesn't have coords, only dims
outnc.expand_dims(dim={'row': np.arange(100000, dtype='int')})
outnc.assign(time=(('row'), nc.time.data[:100000]))

outnc.assign(**{nc.latitude.name: (('row'), nc.latitude.data[:100000])})


# Dynamic keywords
def f(**kwargs):
    print(kwargs.keys())
    return


f(a=2, b="b")
f(**{'d'+'e': 1}) # -> ['de']
