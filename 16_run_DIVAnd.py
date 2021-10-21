"""Run the DIVAnd Python tool"""

import pandas as pd
import numpy as np
import os
import DIVAnd
from clim_helpers import get_standard_levels
import matplotlib.pyplot as plt
import xarray as xr


# Access command prompt
# os.system('cmd /c "set PYTHONPATH=%PYTHONPATH%;C:\\Users\\HourstonH\\DIVAnd.py\\DIVAnd\\"')

var = 'Oxy'
year = 1991
szn = 'JFM'
standard_depth = 0

# Get standard levels file
sl_filename = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
          'WOA_Standard_Depths.txt'

sl_arr = get_standard_levels(sl_filename)
print(sl_arr)

# Get mask file
mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
mask_filename = os.path.join(mask_dir + 'landsea_04_nep.msk')

mask_df = pd.read_csv(mask_filename)
print(mask_df.columns)

# Create 2d grid of lat and lon points by reshaping the mask_df columns
# Find dims to reshape columns to
unique, counts = np.unique(mask_df.Longitude, return_counts=True)
print(len(counts))
print(counts)

Lon = np.array(mask_df.Longitude).reshape((counts[0], len(counts)))
Lat = np.array(mask_df.Latitude).reshape((counts[0], len(counts)))
# DO NOt Have to reverse order of Lat so that Latitude is decreasing each row

# Create boolean mask based on standard level of the input obs
sl_index = np.where(sl_arr == standard_depth)[0][0]
# Indexing of sl_arr starts @ 0, while standard level counting starts at 1
mask = mask_df.Bottom_Standard_level >= (sl_arr[sl_index] + 1)
# Reshape mask to be same shape as Lat and Lon
mask = np.array(mask).reshape((counts[0], len(counts)))

# ---------Test DIVA-provided bathymetry----------------
# Is my problem with the WOA bathymetry, or with the data, or the input
# parameters?

# # bathymetry
# bath_dir = 'C:\\Users\\HourstonH\\DIVAnd.py\\examples\\'
# bath_name = os.path.join(bath_dir + "diva_bath.nc")
#
# # nc = netCDF4.Dataset(fname)
# nc = xr.open_dataset(bath_name)
#
# # b = nc.variables["bat"][:, :]
# # lon = nc.variables["lon"][:]
# # lat = nc.variables["lat"][:]
# b = nc.bat.data
# lon = nc.lon.data
# lat = nc.lat.data
#
# print(b.shape, lon.shape, lat.shape)
# # (160, 361) (361,) (160,)
#
# Lon, Lat = np.meshgrid(lon, lat)
# mask = b < 0


# Calculate input parameters
pm, pn = DIVAnd.metric(Lon, Lat)
print(pm, pn, sep='\n')

# Load data file
data_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
           "value_vs_depth\\14_sep_by_sl_and_year\\"
data_filename = os.path.join(data_dir + '{}_{}m_{}_{}.csv'.format(
    var, standard_depth, year, szn))

data = pd.read_csv(data_filename)

# Convert observations from df columns to numpy arrays
xobs = np.array(data.Longitude)
yobs = np.array(data.Latitude)
vobs = np.array(data.SL_value)

# Set more input parameters
lenx = 800e3  # in meters
leny = 800e3  # in meters
epsilon2 = 1.

# Compute anomalies (i.e., subtract mean)
vmean = np.mean(vobs)
vanom = vobs - vmean

print(min(vanom), max(vanom), np.mean(vanom), np.median(vanom))

print('mask', mask.dtype)
print('pm', pm.dtype)
print('Lat', Lat.dtype)
print('xobs', xobs.dtype)

# Execute the analysis
va = DIVAnd.DIVAnd(mask, (pm, pn), (Lon, Lat), (xobs, yobs), vobs, (lenx, leny), epsilon2)

# va = D.DIVAndrunfi(....
# the same as DIVAndrun, but just return the field fi
# If zero is not a valid first guess for your variable (as it is the case for
#   e.g. ocean temperature), you have to subtract the first guess from the
#   observations before calling DIVAnd and then add the first guess back in.

# JULIA (error): knot-vectors must be unique and sorted in increasing order

print(va.shape)

# Add the field to the mean
vout = va + vmean

# Plot the results
plt.pcolor(Lon, Lat, vout, shading='auto', cmap='jet')
plt.colorbar()
plt_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\outputs\\"
plt_filename = os.path.join(plt_dir + "{}_{}m_{}_{}_analysis2d.png".format(
    var, standard_depth, year, szn))
plt.savefig(plt_filename, dpi=400)

# Export the results as a netCDF file
ncout = xr.Dataset(coords={'Latitude': Lat[:, 0], 'Longitude': Lon[0, :]},
                   data_vars={'analysis': (('Latitude', 'Longitude'), va)})

ncout_filename = os.path.join(plt_dir + "{}_{}m_{}_{}_analysis2d.nc".format(
    var, standard_depth, year, szn))

ncout.to_netcdf(ncout_filename)

ncout.close()
