from xarray import open_dataset
import os
import numpy as np
import pandas as pd

# gebco_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\' \
#                'GEBCO_28_Oct_2021_16f8a0236741\\'
#
# gebco_filename = os.path.join(gebco_folder, 'gebco_2021_n60_s30_w-160_e-115.nc')
#
# gebco_ds = open_dataset(gebco_filename)
#
# lon = gebco_ds.lon.data
# lat = gebco_ds.lat.data
# bat = -gebco_ds.elevation.data
#
# lon2d, lat2d = np.meshgrid(lon, lat)
#
# lon_txt = lon2d.flatten()
# lat_txt = lat2d.flatten()
#
# df_out = pd.DataFrame()
# df_out['Longitude [degrees East]'] = lon_txt + 360
# df_out['Latitude [degrees North]'] = lat_txt
#
# df_filename = os.path.join(gebco_folder, 'nep_latlon_gebco_2021_6_minute_grid.txt')
# df_out.to_csv(df_filename, sep='\t')

# ---------------------------------------------------------------------------------
# import dask.dataframe as dd
# df_dd = dd.read_csv(df_filename)
# ---------------------------------------------------------------------------------

# Subsample the resolution
# Every 24th point -- 24 * 6 minutes = 144 minute res = 2.4 degree resolution
# Every 30th point -- 30 * 6 minutes = 180 minute res = 3 degree resolution


def subsample_grid_res(interval, include_bath=False):
    # Try interval = 24 and interval = 30
    # Original resolution of gebco elevation data is 6 minutes
    new_resolution = interval * 6  # minutes

    gebco_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                   'GEBCO_28_Oct_2021_16f8a0236741\\'

    gebco_filename = os.path.join(gebco_folder, 'gebco_2021_n60_s30_w-160_e-115.nc')

    gebco_ds = open_dataset(gebco_filename)

    lon = gebco_ds.lon.data
    lat = gebco_ds.lat.data
    bat = -gebco_ds.elevation.data

    lon_sub = lon[::interval]
    lat_sub = lat[::interval]

    lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    df_out = pd.DataFrame()
    df_out['Longitude [degrees East]'] = lon_flat + 360
    df_out['Latitude [degrees North]'] = lat_flat

    if include_bath:
        bat_sub = bat[::interval, ::interval]
        bat_flat = bat_sub.flatten()
        df_out['Bathymetry [m below sea level]'] = bat_flat

    if include_bath:
        df_filename = os.path.join(
            gebco_folder, 'nep_latlon_gebco_2021_{}_min_grid_w_bath.txt'.format(
                new_resolution))
    else:
        df_filename = os.path.join(
            gebco_folder, 'nep_latlon_gebco_2021_{}_min_grid.txt'.format(
                new_resolution))

    df_out.to_csv(df_filename, sep='\t', index=False)

    return


subsample_grid_res(interval=24, include_bath=False)
print('Done 24 w/o bath')
subsample_grid_res(interval=24, include_bath=True)
print('Done 24 w bath')
subsample_grid_res(interval=30, include_bath=False)
print('Done 30 w/o bath')
subsample_grid_res(interval=30, include_bath=True)
print('Done 30 w bath')
