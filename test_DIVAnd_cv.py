from DIVAnd_cv import DIVAnd_cv
import pandas as pd
import numpy as np
import os
from xarray import open_dataset
import time
import DIVAnd


# Test out the Julia cross-validation function for estimating correlation length
# and signal-to-noise ratio

# -----------------------Choose data file and params-----------------------------
var_name = 'Oxy'
# standard_depth = 10
# year = 2010
# szn = 'OND'
subsamp_interval_list = [50, 40, 30, 20, 10, 5]  # 3 or 5 is minimum possible

# files = [(10, 2013, 'JAS'), (5200, 1993, 'OND'), (20, 1991, 'AMJ')]
files = [(0, 1991, 'JFM'), (0, 1991, 'AMJ'), (0, 1991, 'JAS'), (0, 1991, 'OND'),
         (0, 2000, 'JFM'), (0, 2000, 'AMJ'), (0, 2000, 'JAS'), (0, 2000, 'OND'),
         (50, 1995, 'JFM'), (50, 1995, 'AMJ'), (50, 1995, 'JAS'), (50, 1995, 'OND')]

# GEBCO 6 minute mask
mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
           '16_diva_analysis\\masks\\'

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\' \
             'cross_validation\\'

# --------------------------------------------------------------------------------

# Iterate through file list
for f in files:
    standard_depth = f[0]
    year = f[1]
    szn = f[2]

    # Initialize object to store new parameter estimates in
    corlen_list = []
    epsilon2_list = []

    # Read in standard level data file
    obs_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
              '14_sep_by_sl_and_year\\'
    obs_filename = os.path.join(obs_dir + '{}_{}m_{}_{}.csv'.format(
        var_name, standard_depth, year, szn))

    print(os.path.basename(obs_filename))

    sldata = pd.read_csv(obs_filename)

    if sldata.empty:
        print('Warning: Dataframe empty')
        continue

    # Convert sldata's contents into numpy arrays
    xobs = np.array(sldata.Longitude)
    yobs = np.array(sldata.Latitude)
    vobs = np.array(sldata.SL_value)

    # print(min(xobs), max(xobs))
    # print(min(yobs), max(yobs))

    # Get boolean mask
    mask_filename = os.path.join(mask_dir + '{}_{}m_{}_{}_mask_6min.nc'.format(
        var_name, standard_depth, year, szn))

    mask_data = open_dataset(mask_filename)

    # Subset mask to speed up computations and avoid Cholesky factorization failure??
    mask_lon_subsetter = np.where(
        (mask_data.lon.data >= np.min(xobs) - 2) &
        (mask_data.lon.data <= np.max(xobs) + 2))[0]
    mask_lat_subsetter = np.where(
        (mask_data.lat.data >= np.min(yobs) - 2) &
        (mask_data.lat.data <= np.max(yobs) + 2))[0]

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

        # # WOA 1/4 deg mask
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
        # #
        # # # Further limit mask according to sampling locations
        # #
        # # mask_v2 = np.zeros(shape=mask.shape)
        # # mask_v2[mask] = 1
        # #
        # # print(len(mask_v2[mask_v2 == 1]), len(mask_v2[mask_v2 == 0]))

        # --------------------Calculate input parameters-----------------------

        # Scale factor of the grid
        pm, pn = DIVAnd.metric(Lon, Lat)
        # print(pm, pn, sep='\n')

        # # For first guess correlation length, can use a value between 1/10 domain size and
        # # the domain size
        # # Also can try optimization on correlation length
        # domain_size_deg = -115-(-160)
        # deg2m = 111e3  # This is an approximation
        # domain_size_m = domain_size_deg * deg2m
        # print(domain_size_m/10)

        # Decreasing the correlation length decreases the "smoothness"
        lenx = 500e3  # 800e3  # in meters
        leny = 500e3  # 800e3  # in meters

        # error variance of the observations (normalized by the error variance of
        # the background field)
        # If epsilon2 is a scalar, it is thus the inverse of the signal-to-noise ratio
        signal_to_noise_ratio = 50.  # Default from Lu ODV session
        epsilon2 = 1/signal_to_noise_ratio  # 1.

        # Compute anomalies (i.e., subtract mean)
        vmean = np.mean(vobs)
        vanom = vobs - vmean

        # Choose number of testing points around the current value of L (corlen)
        nl = 1

        # Choose number of testing points around the current value of epsilon2
        ne = 1

        # Choose cross-validation method
        # 1: full CV; 2: sampled CV; 3: GCV; 0: automatic choice between the three
        method = 3

        # ------------------------Run cross-validation---------------------------------

        # Time the execution
        start_time = time.time()
        print('Start time: {}s'.format(start_time))

        bestfactorl, bestfactore, cvval, cvvalues, x2Ddata, y2Ddata, cvinter, xi2D, yi2D = DIVAnd_cv(
            mask, (pm, pn), (Lon, Lat), (xobs, yobs), vanom, (lenx, leny), epsilon2, nl, ne, method)

        execution_time = time.time() - start_time
        print('Execution time: {} sec'.format(execution_time))

        print('Bestfactorl:', bestfactorl)
        print('bestfactore:', bestfactore)
        # print('cvval:', cvval)
        # print('cvvalues:', cvvalues)
        # print('x2Ddata:', x2Ddata)
        # print('y2dData:', y2Ddata)
        # print('cvinter:', cvinter)
        # print('xi2D:', xi2D)
        # print('yi2D:', yi2D)

        print('New corlen:', lenx * bestfactorl)
        print('New epsilon2:', epsilon2 * bestfactore)

        corlen_list.append(lenx * bestfactorl)
        epsilon2_list.append(epsilon2 * bestfactore)

    # Make lists into a dataframe?
    param_df = pd.DataFrame(
        data=np.array([subsamp_interval_list, corlen_list, epsilon2_list]).transpose(),
        columns=['interval_size', 'lenx', 'epsilon2'])

    param_df_filename = os.path.join(output_dir + 'cv_{}_{}m_{}_{}_nle_{}.csv'.format(
        var_name, standard_depth, year, szn, nl))

    param_df.to_csv(param_df_filename, index=False)
