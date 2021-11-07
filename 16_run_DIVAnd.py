"""Run the DIVAnd Python tool
I have only been able to run this script in terminal, not Pycharm.
To run, follow these steps:

Command prompt:
>conda activate clim38
>set PYTHONPATH=%PYTHONPATH%;C:\\Users\\HourstonH\\DIVAnd.py\\DIVAnd\\
cd to the directory that 16_run_DIVAnd.py is in, then:
>python 16_run_DIVAnd.py

Must set PYTHONPATH each session.
"""

import pandas as pd
import numpy as np
import os
from clim_helpers import get_standard_levels, deg2km
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import trange
import haversine as hs
import time
from DIVAnd_cv import DIVAnd_cv
# from mpl_toolkits.basemap import Basemap
import DIVAnd


# Access command prompt
# os.system('cmd /c "set PYTHONPATH=%PYTHONPATH%;C:\\Users\\HourstonH\\DIVAnd.py\\DIVAnd\\"')

# ---------------------------------Set general parameters------------------------------------
var_name = 'Oxy'
var_units = r'$\mu$' + 'mol/kg'  # Micromol per kilogram

# Test files with diverging (not stabilizing to a point) estimates of corlen and epsilon2
file_info = (0, 1991, 'AMJ')  # (0, 1991, 'JFM')
standard_depth = file_info[0]
year = file_info[1]
szn = file_info[2]

radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)  # degrees length
var_cmap = 'Blues'  # matplotlib colormap to use for plotting the DIVAnd field

# Use 'Blues' for Sal and 'Oranges' or 'Reds' for Temp

# Mask subsampling interval for cross-validation
mask_subsamp_int = 10
# -------------------------------------------------------------------------------------------

# Get standard levels file
sl_filename = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
              'WOA_Standard_Depths.txt'

sl_arr = get_standard_levels(sl_filename)
# print(len(sl_arr))

# Load data file
data_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
           "value_vs_depth\\14_sep_by_sl_and_year\\"
data_filename = os.path.join(data_dir + '{}_{}m_{}_{}.csv'.format(
    var_name, standard_depth, year, szn))

print(os.path.basename(data_filename))

data = pd.read_csv(data_filename)

# Convert observations from df columns to numpy arrays
xobs = np.array(data.Longitude)
yobs = np.array(data.Latitude)
vobs = np.array(data.SL_value)

# ---------------------------Get mask file-------------------------------------------------

# mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
# mask_filename = os.path.join(mask_dir + 'landsea_04_nep.msk')
#
# mask_df = pd.read_csv(mask_filename)
# # print(mask_df.columns)
# print('Mask range:', min(mask_df.Bottom_Standard_level), max(mask_df.Bottom_Standard_level))
#
# # Create 2d grid of lat and lon points by reshaping the mask_df columns
# # Find dims to reshape columns to
# unique, counts = np.unique(mask_df.Longitude, return_counts=True)
# print('counts length:', len(counts))
# # print(counts)
#
# Lon = np.array(mask_df.Longitude).reshape((counts[0], len(counts)))
# Lat = np.array(mask_df.Latitude).reshape((counts[0], len(counts)))
# # DO NOt Have to reverse order of Lat so that Latitude is decreasing each row
#
# # Create boolean mask based on standard level of the input obs
# sl_index = np.where(sl_arr == standard_depth)[0][0]
# # Indexing of sl_arr starts @ 0, while standard level counting starts at 2!
# # Bottom_Standard_level starts at 1, which is land, so ocean starts at 2
# mask = mask_df.Bottom_Standard_level >= (sl_index + 2)
# # Reshape mask to be same shape as Lat and Lon
# mask = np.array(mask).reshape((counts[0], len(counts)))
#
# # Further limit mask according to sampling locations
# # Determine radius around sampling points to limit mask to? 10 deg maybe?
# # Need arcpy for this?
# radius_km = deg2km(radius_deg)  # degrees length
#
# mask_v2 = np.zeros(shape=mask.shape)
# mask_v2[mask] = 1
#
# print(len(mask_v2[mask_v2 == 1]), len(mask_v2[mask_v2 == 0]))
#
# print('Recalculating mask...')
#
# for i in trange(len(vobs)):
#     # Create tuple of the lon/lat of each standard level observation point
#     obs_loc = (xobs[i], yobs[i])
#     for j in range(len(Lat)):
#         for k in range(len(Lon[0])):
#             # Check if mask is True, otherwise pass
#             # Also pass if mask_v2[j, k] == 2, so it's already been checked
#             if mask_v2[j, k] == 1:
#                 grid_loc = (Lon[j, k], Lat[j, k])
#                 dist = hs.haversine(obs_loc, grid_loc)
#                 # print(dist)
#                 if dist < radius_km:
#                     mask_v2[j, k] = 2
#
# print(len(mask_v2[mask_v2 == 2]), len(mask_v2[mask_v2 == 1]))
#
# # Create boolean mask version
# mask_v3 = np.empty(shape=mask_v2.shape, dtype=bool)
# mask_v3[mask_v2 == 2] = True
# mask_v3[mask_v2 != 2] = False

# --------------------Test out GEBCO bathymetry-----------------------------------------


def update_gebco_bath(lon_obs, lat_obs, search_radius):
    gebco_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\' \
                'GEBCO_28_Oct_2021_16f8a0236741\\'

    gebco_filename = os.path.join(gebco_dir + 'gebco_2021_n60_s30_w-160_e-115.nc')

    gebco_bath = xr.open_dataset(gebco_filename)

    # print(np.diff(gebco_bath.lat.data))

    # Create 2d grid of lat and lon
    Lon2d, Lat2d = np.meshgrid(gebco_bath.lon.data, gebco_bath.lat.data)

    grid_shape = Lon2d.shape
    print(Lon2d.shape)
    # print(Lon)
    # print(Lat)

    # Find limits for range of standard level observations
    lon_min, lon_max, lat_min, lat_max = [np.nanmin(lon_obs), np.nanmax(lon_obs),
                                          np.nanmin(lat_obs), np.nanmax(lat_obs)]

    # -1 to convert elevation above sea level to depth below sea level
    # Subset out obviously out lat/lon
    mask = (-gebco_bath.elevation.data >= standard_depth) & (Lon2d >= lon_min - radius_deg) & \
           (Lon2d <= lon_max + radius_deg) & (Lat2d >= lat_min - radius_deg) & \
           (Lat2d <= lat_max + radius_deg)

    # mask_v2 = np.zeros(mask.shape, dtype='int')
    # mask_v2[mask] = 1
    #
    # print(len(mask_v2[mask_v2 == 1]), len(mask_v2[mask_v2 == 0]))

    print('Recalculating mask...')

    # Flatten the boolean mask
    mask_flat = mask.flatten()
    mask_v2_flat = np.zeros(len(mask_flat), dtype=int)
    mask_v2_flat[mask_flat] = 1

    for i in trange(len(vobs)):
        # Create tuple of the lon/lat of each standard level observation point
        obs_loc = (lon_obs[i], lat_obs[i])

        # print(i, 'Creating dist_arr...')
        dist_arr = np.repeat(np.nan, len(Lon2d.flatten()))
        dist_arr[mask_flat] = np.array(list(map(
            lambda x, y: hs.haversine(obs_loc, (x, y)), Lon2d[mask], Lat2d[mask])))
        # print(i, 'Dist time: %s seconds' % (time.time() - start_dist))

        mask_v2_flat[dist_arr < search_radius] = 2

    # Reshape flattened mask back to 2d
    mask_v2 = mask_v2_flat.reshape(grid_shape)
    mask_v3 = np.repeat(False, mask_v2_flat.shape).reshape(grid_shape)
    mask_v3[mask_v2 == 2] = True

    return Lon2d, Lat2d, mask_v3


# Compute subsetter mask if not already exists
# Lon, Lat, bool_mask = update_gebco_bath(xobs, yobs, radius_km)

# GEBCO 6 minute mask
mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
           '16_diva_analysis\\masks\\'

mask_filename = os.path.join(mask_dir + '{}_{}m_{}_{}_mask_6min.nc'.format(
    var_name, standard_depth, year, szn))

mask_data = xr.open_dataset(mask_filename)
Lon, Lat = np.meshgrid(mask_data.lon.data, mask_data.lat.data)
bool_mask = mask_data.mask.data

# -------------------------------Calculate input parameters-----------------------------

# Scale factor of the grid
pm, pn = DIVAnd.metric(Lon, Lat)
# print(pm, pn, sep='\n')

# # For first guess correlation length, can use a value between 1/10 domain size and
# # the domain size
# # Also can try optimization on correlation length
# domain_size_deg = -115-(-160)
# deg2m = 111e3  # This is an approximation
# domain_size_m = domain_size_deg * deg2m  # about 5e6 metres

# Decreasing the correlation length decreases the "smoothness"
lenx = 500e3  # 800e3  # in meters
leny = 500e3  # 800e3  # in meters

# error variance of the observations (normalized by the error variance of
# the background field)
# If epsilon2 is a scalar, it is thus the inverse of the signal-to-noise ratio
signal_to_noise_ratio = 50.  # Default from Lu ODV session
epsilon2 = 1 / signal_to_noise_ratio  # 1.

# Compute anomalies (i.e., subtract mean)
vmean = np.mean(vobs)
vanom = vobs - vmean

print('vanom stats:', min(vanom), max(vanom), np.mean(vanom), np.median(vanom))

# Choose number of testing points around the current value of L (corlen)
nl_cv = 1

# Choose number of testing points around the current value of epsilon2
ne_cv = 1

# Choose cross-validation method
# 1: full CV; 2: sampled CV; 3: GCV; 0: automatic choice between the three
method_cv = 3


def run_cv(subsamp_interval, mask, lon1d, lat1d, lenx_guess, leny_guess, epsilon2_guess,
           nl, ne, method):
    # Subsample lat and lon of the mask
    # Subset mask to speed up computations and avoid Cholesky factorization failure??
    mask_lon_subsetter = np.where(
        (lon1d >= np.min(xobs) - 2) & (lon1d <= np.max(xobs) + 2))[0]
    mask_lat_subsetter = np.where(
        (lat1d >= np.min(yobs) - 2) & (lat1d <= np.max(yobs) + 2))[0]

    mask_lon_subsetter2 = mask_lon_subsetter[::subsamp_interval]
    mask_lat_subsetter2 = mask_lat_subsetter[::subsamp_interval]

    Lon_subset, Lat_subset = np.meshgrid(lon1d[mask_lon_subsetter2], lat1d[mask_lat_subsetter2])

    pm_subset, pn_subset = DIVAnd.metric(Lon_subset, Lat_subset)
    mask_subset = mask[mask_lat_subsetter2][:, mask_lon_subsetter2]

    # Cross-validation to estimate
    bestfactorl, bestfactore, cvval, cvvalues, x2Ddata, y2Ddata, cvinter, xi2D, yi2D = DIVAnd_cv(
        mask_subset, (pm_subset, pn_subset), (Lon_subset, Lat_subset), (xobs, yobs), vanom,
        (lenx_guess, leny_guess), epsilon2_guess, nl, ne, method)

    # Update the correlation length and epsilon2 to the estimates from cross-validation
    lenx_est = bestfactorl * lenx
    leny_est = bestfactorl * leny
    epsilon2_est = bestfactore * epsilon2

    print(lenx_est, leny_est, epsilon2_est)

    return lenx_est, leny_est, epsilon2_est


# lenx_cv, leny_cv, epsilon2_cv = run_cv(mask_subsamp_int, bool_mask, mask_data.lon.data,
#                                        mask_data.lat.data, lenx, leny, epsilon2, nl_cv,
#                                        ne_cv, method_cv)

lenx_cv, leny_cv, epsilon2_cv = [500e3, 500e3, 1/50.]

# nl=ne=1, Oxy 0m 2010 OND
# lenx_cv, leny_cv, epsilon2_cv = [1377114.3516690833, 1377114.3516690833, 0.06600873717044665]

# -----------------------------------Run the analysis---------------------------------------

# Execute the analysis
va = DIVAnd.DIVAnd(bool_mask, (pm, pn), (Lon, Lat), (xobs, yobs), vanom,
                   (lenx_cv, leny_cv), epsilon2_cv)

# va = D.DIVAndrunfi(....
# the same as DIVAndrun, but just return the field fi
# If zero is not a valid first guess for your variable (as it is the case for
#   e.g. ocean temperature), you have to subtract the first guess from the
#   observations before calling DIVAnd and then add the first guess back in.

# JULIA (error): knot-vectors must be unique and sorted in increasing order
# Fixed by ordering lat in increasing order

print(va.shape)
print(np.min(va), np.max(va))

# Add the field to the mean
vout = va + vmean

# -------------------------Plot the results------------------------------------

# # Add in basemap coast
# # Set up Lambert conformal map
# left_lon = -162.
# bot_lat = 22.
# right_lon = -100.
# top_lat = 62.
#
# m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
#             urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
#             resolution='h', lat_0=0.5 * (bot_lat + top_lat),
#             lon_0=0.5 * (left_lon + right_lon))
#
# # Initialize figure
# fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
# # ax = plt.subplot(1, 1, 1) # need in order to add points to plot iteratively?
# m.drawcoastlines(linewidth=0.2)
# m.drawmapboundary(fill_color='white')
# m.fillcontinents(color='0.8')
#
# # Plot the locations of the profiles
# xobs, yobs = m(lon_subset, lat_subset)
# m.scatter(xobs, yobs, marker='o', color='r', s=0.5)


# Make simple 2d pcolor map

# For color bar
# vmin = 0
# vmax = 15
#
# qcs = ax.contourf(
#       X, Y, data,
#       vmin=vmin, vmax=vmax
# )

plt.pcolormesh(Lon, Lat, vout, shading='auto', cmap=var_cmap)  #, vmin=150, vmax=400)
plt.colorbar(label='{} [{}]'.format(var_name, var_units))  # ticks=range(150, 400 + 1, 50)

# Scatter plot the observation points
plt.scatter(xobs, yobs, c='k', s=0.1)
plt.title('nl = ne = {}'.format(nl_cv))

# Set limits
plt.xlim((-160., -102.))
plt.ylim((25., 62.))

plt_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
          "16_diva_analysis\\"
plt_filename = os.path.join(plt_dir + "{}_{}m_{}_{}_analysis2d_gebco_nle{}.png".format(
    var_name, standard_depth, year, szn, nl_cv))
plt.savefig(plt_filename, dpi=400)

plt.close()

# ---------------Export the results as a netCDF file-------------------------

# Create Dataset object
ncout = xr.Dataset(coords={'Latitude': Lat[:, 0], 'Longitude': Lon[0, :]},
                   data_vars={'analysis': (('Latitude', 'Longitude'), vout)})
#                  data_vars={'analysis': (('Latitude', 'Longitude'), va),
#                             'pre_analysis_obs_mean': ((), vmean)})

ncout_filename = os.path.join(plt_dir + "{}_{}m_{}_{}_analysis2d_gebco_nle{}.nc".format(
    var_name, standard_depth, year, szn, nl_cv))

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
