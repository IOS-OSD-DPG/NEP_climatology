import pandas as pd
import numpy as np
import os
from clim_helpers import deg2km, get_standard_levels
from xarray import open_dataset
import DIVAnd


# Test out the Julia cross-validation function for estimating correlation length
# and signal-to-noise ratio

# -----------------------Choose data file and params-----------------------------
var_name = 'Oxy'
year = 2010
szn = 'OND'
standard_depth = 0
radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)  # degrees length
subsamp_interval_list = [1]  # 3 is minimum possible
# -------------------------------------------------------------------------------

# Get standard levels file
sl_filename = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
              'WOA_Standard_Depths.txt'

sl_arr = get_standard_levels(sl_filename)

# Read in standard level data file
sl_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
         '14_sep_by_sl_and_year\\'
sl_filename = os.path.join(sl_dir + '{}_{}m_{}_{}.csv'.format(
    var_name, standard_depth, year, szn))

sldata = pd.read_csv(sl_filename)

# Convert sldata's contents into numpy arrays
xobs = np.array(sldata.Longitude)
yobs = np.array(sldata.Latitude)
vobs = np.array(sldata.SL_value)

# print(min(xobs), max(xobs))
# print(min(yobs), max(yobs))

# ---------------------------------------mask------------------------------------------

# GEBCO 6 minute mask
mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
           '16_diva_analysis\\masks\\'

mask_filename = os.path.join(mask_dir + '{}_{}m_{}_{}_mask_6min.nc'.format(
    var_name, standard_depth, year, szn))

mask_data = open_dataset(mask_filename)

# Subset mask to speed up computations and avoid Cholesky factorization failure??
mask_lon_subsetter = np.where(
    (mask_data.lon.data >= np.min(xobs) - 2) & (mask_data.lon.data <= np.max(xobs) + 2))[0]
mask_lat_subsetter = np.where(
    (mask_data.lat.data >= np.min(yobs) - 2) & (mask_data.lat.data <= np.max(yobs) + 2))[0]


for subsamp_interval in subsamp_interval_list:
    print()
    print('Subsampling interval:', subsamp_interval)

    # Reduce the subsetter for testing Cholesky factorization
    mask_lon_subsetter2 = mask_lon_subsetter[::subsamp_interval]
    mask_lat_subsetter2 = mask_lat_subsetter[::subsamp_interval]

    mask = mask_data.mask.data[mask_lat_subsetter2][:, mask_lon_subsetter2]
    print(mask.shape)
    Lon, Lat = np.meshgrid(mask_data.lon.data[mask_lon_subsetter2],
                           mask_data.lat.data[mask_lat_subsetter2])  # Create 2d arrays

    pm, pn = DIVAnd.metric(Lon, Lat)

    print('pm:')
    print('min:', np.min(pm), 'max:', np.max(pm))
    print(pm)
    print('pn:')
    print('min:', np.min(pn), 'max:', np.max(pn))
    print(pn)

