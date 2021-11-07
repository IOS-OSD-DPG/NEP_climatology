import numpy as np
import os
import xarray as xr
import pandas as pd
from tqdm import trange
import haversine as hs
from clim_helpers import deg2km


# -----------------------------Choose data file----------------------------------
var_name = 'Oxy'
years = np.arange(1991, 2021)  # [1995, 2005]
szns = ['JFM', 'AMJ', 'JAS', 'OND']
standard_depth = 5100
radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)  # degrees length
# -------------------------------------------------------------------------------

# Use GEBCO 2021 6'x6' bathymetry file to create masks by depth

# Read in elevation file
gebco_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\' \
            'GEBCO_28_Oct_2021_16f8a0236741\\'

gebco_filename = os.path.join(gebco_dir + 'gebco_2021_n60_s30_w-160_e-115.nc')

gebco_data = xr.open_dataset(gebco_filename)

# print(np.diff(gebco_bath.lat.data))

# Create 2d grid of lat and lon
Lon, Lat = np.meshgrid(gebco_data.lon.data, gebco_data.lat.data)
# print(Lon.shape)
# print(Lon)
# print(Lat)

print('Depth: {}m'.format(standard_depth))
for y in years:
    print(y)
    for s in szns:
        print(s)
        # Read in standard level data file
        sl_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
                 '14_sep_by_sl_and_year\\'
        sl_filename = os.path.join(sl_dir + '{}_{}m_{}_{}.csv'.format(
            var_name, standard_depth, y, s))

        sldata = pd.read_csv(sl_filename)

        if sldata.empty:
            print('Dataframe empty -- skipping')
            continue

        xobs = np.array(sldata.Longitude)
        yobs = np.array(sldata.Latitude)

        # Find limits for range of standard level observations
        lon_min, lon_max, lat_min, lat_max = [np.nanmin(xobs), np.nanmax(xobs),
                                              np.nanmin(yobs), np.nanmax(yobs)]

        # -1 to convert elevation above sea level to depth below sea level
        # Subset out obviously out lat/lon
        mask = (-gebco_data.elevation.data >= standard_depth) & (Lon >= lon_min - radius_deg) & \
               (Lon <= lon_max + radius_deg) & (Lat >= lat_min - radius_deg) & \
               (Lat <= lat_max + radius_deg)

        # Flatten the boolean mask
        mask_flat = mask.flatten()
        mask_v2_flat = np.zeros(len(mask_flat), dtype=int)
        mask_v2_flat[mask_flat] = 1

        # start_time = time.time()
        for i in trange(len(xobs)):
            # Create tuple of the lon/lat of each standard level observation point
            obs_loc = (xobs[i], yobs[i])

            # print(i, 'Creating dist_arr...')
            # start_dist = time.time()
            dist_arr = np.repeat(np.nan, len(Lon.flatten()))
            dist_arr[mask_flat] = np.array(list(map(
                lambda x, y: hs.haversine(obs_loc, (x, y)), Lon[mask], Lat[mask])))
            # print(i, 'Dist time: %s seconds' % (time.time() - start_dist))

            mask_v2_flat[dist_arr < radius_km] = 2

        # print("--- %s seconds ---" % (time.time() - start_time))

        # Reshape flattened mask back to 2d
        mask_v2 = mask_v2_flat.reshape(Lon.shape)
        mask_v3 = np.repeat(False, mask_v2_flat.shape).reshape(Lon.shape)
        mask_v3[mask_v2 == 2] = True

        # Export boolean mask to netCDF file
        ncout = xr.Dataset(coords={'lon': gebco_data.lon.data, 'lat': gebco_data.lat.data},
                           data_vars={'mask': (('lat', 'lon'), mask_v3)})

        ncout_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
                    '16_diva_analysis\\masks\\'

        ncout_filename = os.path.join(ncout_dir + '{}_{}m_{}_{}_mask_6min.nc'.format(
            var_name, standard_depth, y, s))

        ncout.to_netcdf(ncout_filename)

        ncout.close()
