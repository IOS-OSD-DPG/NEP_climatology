"""Run the DIVAnd Python tool"""

import pandas as pd
import numpy as np
import os
from clim_helpers import get_standard_levels
import matplotlib.pyplot as plt
import xarray as xr
import DIVAnd


# Access command prompt
# os.system('cmd /c "set PYTHONPATH=%PYTHONPATH%;C:\\Users\\HourstonH\\DIVAnd.py\\DIVAnd\\"')

var = 'Oxy'
var_units = 'umol/kg'
year = 2010
szn = 'OND'
standard_depth = 10

# Get standard levels file
sl_filename = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
          'WOA_Standard_Depths.txt'

sl_arr = get_standard_levels(sl_filename)
# print(len(sl_arr))

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

# Get mask file
mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
mask_filename = os.path.join(mask_dir + 'landsea_04_nep.msk')

mask_df = pd.read_csv(mask_filename)
print(mask_df.columns)
print(min(mask_df.Bottom_Standard_level), max(mask_df.Bottom_Standard_level))

# Create 2d grid of lat and lon points by reshaping the mask_df columns
# Find dims to reshape columns to
unique, counts = np.unique(mask_df.Longitude, return_counts=True)
print('counts length', len(counts))
# print(counts)

Lon = np.array(mask_df.Longitude).reshape((counts[0], len(counts)))
Lat = np.array(mask_df.Latitude).reshape((counts[0], len(counts)))
# DO NOt Have to reverse order of Lat so that Latitude is decreasing each row

# Create boolean mask based on standard level of the input obs
sl_index = np.where(sl_arr == standard_depth)[0][0]
# Indexing of sl_arr starts @ 0, while standard level counting starts at 2!
# Bottom_Standard_level starts at 1, which is land, so ocean starts at 2
mask = mask_df.Bottom_Standard_level >= (sl_index + 2)
# Reshape mask to be same shape as Lat and Lon
mask = np.array(mask).reshape((counts[0], len(counts)))

# Further limit mask according to sampling locations
# Determine radius around sampling points to limit mask to? 10 deg maybe?
# Need arcpy for this?
radius = 10  # degrees


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
# print(pm, pn, sep='\n')

# For first guess correlation length, can use a value between 1/10 domain size and
# the domain size
# Also can try optimization on correlation length
domain_size_deg = -115-(-160)
deg2m = 111e3
domain_size_m = domain_size_deg * deg2m
# print(domain_size_m/10)

# Decreasing the correlation length decreases the "smoothness"
lenx = 500e3  # 800e3  # in meters
leny = 500e3  # 800e3  # in meters

# error variance of the observations (normalized by the error variance of
# the background field)
# If epsilon2 is a scalar, it is thus the inverse of the signal-to-noise ratio
epsilon2 = 1.

# Compute anomalies (i.e., subtract mean)
vmean = np.mean(vobs)
vanom = vobs - vmean

print('vanom stats:', min(vanom), max(vanom), np.mean(vanom), np.median(vanom))

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
print(np.min(va), np.max(va))

# Add the field to the mean
vout = va + vmean

# Plot the results
plt.pcolor(Lon, Lat, vout, shading='auto', cmap='jet')
plt.colorbar(label=var_units)

# Scatter plot the observation points
plt.scatter(xobs, yobs, c='k', s=5)

# Set limits
plt.xlim((-160., -115.))
plt.ylim((30., 60.))

plt_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\outputs\\"
plt_filename = os.path.join(plt_dir + "{}_{}m_{}_{}_analysis2d_500e5.png".format(
    var, standard_depth, year, szn))
plt.savefig(plt_filename, dpi=400)

# ---------------Export the results as a netCDF file-------------------------

# Create Dataset object
ncout = xr.Dataset(coords={'Latitude': Lat[:, 0], 'Longitude': Lon[0, :]},
                   data_vars={'analysis': (('Latitude', 'Longitude'), va),
                              'pre_analysis_obs_mean': ((), vmean)})

ncout_filename = os.path.join(plt_dir + "{}_{}m_{}_{}_analysis2d_500e5.nc".format(
    var, standard_depth, year, szn))

ncout.to_netcdf(ncout_filename)

ncout.close()


# -------masks-------------------
# woa_mask_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\WOA_clim_code\\' \
#                 'landsea_04.msk'
#
# woa_mask = pd.read_csv(woa_mask_file, skiprows=1)
#
# # There are 138 standard levels in the mask, but 137 in the WOA documents
# # So standard level == 1 used for LAND and ocean starts at standard level == 2
# print(min(woa_mask.Bottom_Standard_level), max(woa_mask.Bottom_Standard_level))
#
# # Create 2d grid of lat and lon points by reshaping the mask_df columns
# # Find dims to reshape columns to
# unique, counts = np.unique(woa_mask.Longitude, return_counts=True)
# print(len(counts))
# print(counts)
#
# Lon = np.array(woa_mask.Longitude).reshape((counts[0], len(counts)))
# Lat = np.array(woa_mask.Latitude).reshape((counts[0], len(counts)))
# SL = np.array(woa_mask.Bottom_Standard_level).reshape((counts[0], len(counts)))
#
# plt.pcolor(Lon, Lat, SL, shading='auto', cmap='jet')
# plt.colorbar()
# woa_plot_filename = os.path.join(plt_dir + 'WOA_mask_04_standard_levels.png')
# plt.savefig(woa_plot_filename, dpi=400)
# plt.close()
#
# land_finder = np.where(woa_mask.Bottom_Standard_level == 1)[0]
# land = woa_mask.loc[land_finder, ['Longitude', 'Latitude']]
