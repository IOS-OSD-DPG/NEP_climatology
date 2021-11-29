import numpy as np
import os
import xarray as xr
import pandas as pd
from tqdm import trange
# import time
# import haversine as hs
from clim_helpers import deg2km, get_standard_levels
# from numba import jit
import dask


# @jit(nopython=True)
def haversine(point1, point2):
    # Copied from https://github.com/mapado/haversine/blob/main/haversine/haversine.py
    # to use numpy instead of math module
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    earth_radius_km = 6371.0088
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = np.radians(lat1)
    lng1 = np.radians(lng1)
    lat2 = np.radians(lat2)
    lng2 = np.radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    # Return distance in km units
    return 2 * earth_radius_km * np.arcsin(np.sqrt(d))


# @jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def generate_gebco_mask(lon_obs, lat_obs, elevation, Lon2d, Lat2d, depth):
    # Mask out land area of gebco bathymetry

    # sldata = pd.read_csv(obs_filename)

    # Find limits for range of standard level observations
    lon_min, lon_max, lat_min, lat_max = [np.min(lon_obs), np.max(lon_obs),
                                          np.min(lat_obs), np.max(lat_obs)]

    # -1 to convert elevation above sea level to depth below sea level
    # Subset out obviously out lat/lon
    mask = (-elevation >= depth) & (Lon2d >= lon_min - radius_deg) & \
           (Lon2d <= lon_max + radius_deg) & (Lat2d >= lat_min - radius_deg) & \
           (Lat2d <= lat_max + radius_deg)

    # Flatten the boolean mask
    mask_flat = mask.flatten()
    mask_v2_flat = np.repeat(0, len(mask_flat))
    mask_v2_flat[mask_flat] = 1

    # start_time = time.time()
    for i in trange(len(lon_obs)):
        # Create tuple of the lon/lat of each standard level observation point
        obs_loc = (lon_obs[i], lat_obs[i])

        # print(i, 'Creating dist_arr...')
        # start_dist = time.time()
        dist_arr = np.repeat(np.nan, len(Lon2d.flatten()))

        # start_time = time.time()  # Time the lambda function
        # dist_arr[mask_flat] = np.array(list(map(
        #     lambda x, y: hs.haversine(obs_loc, (x, y)), Lon2d[mask], Lat2d[mask])))

        # Fancy indexing of numpy arrays not supported by numba
        dist_arr[mask_flat] = np.array(
            [haversine(obs_loc, (x, y)) for (x, y) in zip(Lon2d[mask], Lat2d[mask])])
        # print(i, 'Execution time: %s seconds' % (time.time() - start_time))

        # for j in range(len(mask_flat)):

        mask_v2_flat[dist_arr < radius_km] = 2

    # print("--- %s seconds ---" % (time.time() - start_time))

    # Reshape flattened mask back to 2d
    mask_v2 = mask_v2_flat.reshape(Lon.shape)
    mask_v3 = np.repeat(False, mask_v2_flat.shape).reshape(Lon.shape)
    mask_v3[mask_v2 == 2] = True

    return mask_v3


def generate_gebco_mask_dask(lon_obs, lat_obs, elevation, Lon2d, Lat2d, depth, year, season,
                             ncout_dir):
    # Mask out land area of gebco bathymetry
    # Use dask to improve execution speed...

    # Find limits for range of standard level observations
    lon_min, lon_max, lat_min, lat_max = [np.min(lon_obs), np.max(lon_obs),
                                          np.min(lat_obs), np.max(lat_obs)]

    # -1 to convert elevation above sea level to depth below sea level
    # Subset out obviously out lat/lon
    mask = (-elevation >= depth) & (Lon2d >= lon_min - radius_deg) & \
           (Lon2d <= lon_max + radius_deg) & (Lat2d >= lat_min - radius_deg) & \
           (Lat2d <= lat_max + radius_deg)

    # Flatten the boolean mask
    mask_flat = mask.flatten()
    mask_v2_flat = np.repeat(0, len(mask_flat))
    mask_v2_flat[mask_flat] = 1

    Lon2d_flat = Lon2d.flatten()
    Lat2d_flat = Lat2d.flatten()

    for i in trange(len(lon_obs)):
        # Create tuple of the lon/lat of each standard level observation point
        obs_loc = (lon_obs[i], lat_obs[i])

        # Compute haversine distance between the observation point and each
        # point in the mask coordinates
        # Dask delays the computation of the haversine distance
        dist_arr = dask.delayed(
            lambda x, y: haversine(obs_loc, (x, y)))(Lon2d_flat, Lat2d_flat)

        # Returns a tuple; index its first element which is the result
        Dist_arr = dask.compute(dist_arr)[0]
        # print(len(Dist_arr))

        # If distance less than search radius
        mask_v2_flat[Dist_arr < radius_km] = 2

    # Reshape flattened mask back to 2d
    mask_v2 = mask_v2_flat.reshape(Lon.shape)
    # Create final version of boolean mask
    mask_v3 = np.repeat(False, mask_v2_flat.shape).reshape(Lon.shape)
    mask_v3[mask_v2 == 2] = True

    # Export boolean mask to netCDF file
    ncout = xr.Dataset(coords={'lon': Lon2d[0], 'lat': Lat2d[:, 0]},
                       data_vars={'mask': (('lat', 'lon'), mask_v3)})

    ncout_filename = os.path.join(ncout_dir + '{}_{}m_{}_{}_mask_6min.nc'.format(
        var_name, depth, year, season))

    ncout.to_netcdf(ncout_filename)

    ncout.close()  # Close the dataset

    return ncout_filename


# -----------------------------Choose data file----------------------------------
var_name = 'Oxy'
years = np.arange(1991, 2021)  # [1995, 2005]
szns = ['JFM', 'AMJ', 'JAS', 'OND']

# standard_depths = np.arange(1500, 500, -50)  # np.arange(3900, 2900, -100)
# standard_depths = [0]
# Already made all 0m and 5m masks so skip to 10m
standard_depths = get_standard_levels(
    'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\WOA_Standard_Depths.txt')[2:]

radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)  # degrees length

# Standard level observations directory
obs_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
          '14_sep_by_sl_and_year\\'

# Directory for netCDF boolean masks to be output into
out_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
          '16_diva_analysis\\masks\\'
# ---------------------------------------------------------------------------------

# Use GEBCO 2021 6'x6' bathymetry file to create masks by depth

# Read in elevation file
gebco_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\' \
            'GEBCO_28_Oct_2021_16f8a0236741\\'

gebco_filename = os.path.join(gebco_dir + 'gebco_2021_n60_s30_w-160_e-115.nc')

gebco_data = xr.open_dataset(gebco_filename)

# Create 2d grid of lat and lon
Lon, Lat = np.meshgrid(gebco_data.lon.data, gebco_data.lat.data)

# Iterate through all requested files
for dep in standard_depths:
    print()
    print('--------------------Depth: {}m--------------------'.format(dep))
    for yr in years:
        print(yr)
        for szn in szns:
            print(szn)
            # Skip making mask if it already exists
            if os.path.exists(out_dir + '{}_{}m_{}_{}_mask_6min.nc'.format(
                    var_name, dep, yr, szn)):
                print('Mask already exists for this file -- skipping')
                continue

            # Read in standard level data file
            obs_filename = os.path.join(obs_dir + '{}_{}m_{}_{}.csv'.format(
                var_name, dep, yr, szn))

            # Read into pandas dataframe
            sldata = pd.read_csv(obs_filename)

            if sldata.empty:
                print('Dataframe empty -- skipping')
                continue

            mask_out = generate_gebco_mask_dask(np.array(sldata['Longitude']),
                                                np.array(sldata['Latitude']),
                                                gebco_data.elevation.data, Lon, Lat,
                                                dep, yr, szn, out_dir)

# ----------------------------------------------------------------------------------
# # RAM calculations
# # https://blogs.sas.com/content/iml/2014/04/28/how-much-ram-do-i-need-to-store-that-matrix.html
# r = Lon.shape[0]
# c = Lon.shape[1]
# gb_RAM_needed = r * c * 8 / 1e9
# print(gb_RAM_needed)
