import os
import numpy as np
import pandas as pd
from clim_helpers import szn_str2int, plot_linterp_tg_data
import xarray as xr


def avg_tg_data(input_dir, output_dir, var_name, depth, szn):
    """
    Average triangle grid data over 1991-2020 and output averaged data in netcdf file
    :param input_dir:
    :param output_dir:
    :param var_name:
    :param depth:
    :param szn:
    :return: full filename of output netcdf file
    """
    years = np.arange(1991, 2021)
    num_of_nodes = 97959

    # Initialize matrix for computing the means on
    means_mtx = np.repeat(np.nan, num_of_nodes * len(years)).reshape(
        (num_of_nodes, len(years)))

    for i in range(len(years)):
        ncin_file = os.path.join(input_dir + '{}_{}m_{}_{}_tg.nc'.format(
            var_name, depth, years[i], szn))
        if os.path.exists(ncin_file):
            # Need to check if file exists because not all years have observations
            ncin_ds = xr.open_dataset(ncin_file)
            means_mtx[:, i] = ncin_ds.SL_value.data

    SL_value_means = np.nanmean(means_mtx, axis=1)
    # print(SL_value_means.shape)

    ncout_ds = xr.Dataset(
        coords={'node': ncin_ds.node.data},
        data_vars={'Season': (('node'), np.repeat(szn_str2int(szn), num_of_nodes)),
                   'longitude': (('node'), ncin_ds.longitude.data),
                   'latitude': (('node'), ncin_ds.latitude.data),
                   'SL_value_30yr_avg': (('node'), SL_value_means)})

    # Add attrs to ncout
    ncout_ds.longitude.attrs['units'] = 'degrees East'
    ncout_ds.latitude.attrs['units'] = 'degrees North'
    ncout_ds.SL_value_30yr_avg.attrs['units'] = ncin_ds.SL_value.attrs['units']

    ncout_filename = os.path.join(output_dir + '{}_{}m_{}_tg_30yr_avg.nc'.format(
        var_name, depth, szn))

    ncout_ds.to_netcdf(ncout_filename)

    ncin_ds.close()
    ncout_ds.close()
    return ncout_filename


# --------------------------------------------------------------------------------

# Set parameters
variable_name = 'Oxy'
variable_units_math = r'$\mu$' + 'mol/kg'  # Micromol per kilogram
variable_units = 'micromoles per kilogram'
variable_cmap = 'Blues'
standard_depth = 0
years = np.arange(1991, 2021)
season = 'OND'
# seasons = ['OND']  # ['JFM', 'AMJ', 'JAS', 'OND']

num_of_nodes = 97959

# Use the files that have been linearly interpolated to the triangle grid
# Have the files be in .npy format or .csv format?
# input_folder = '/home/hourstonh/Documents/climatology/data/value_vs_depth/17_lin_interp_to_trigrid/'
# output_folder = '/home/hourstonh/Documents/climatology/data/value_vs_depth/18_30yr_avg/'

# Windows paths
input_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
               'value_vs_depth\\17_lin_interp_to_trigrid\\'
output_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                'value_vs_depth\\18_30yr_avg\\'

# For plotting
mforeman_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\MForeman\\'

# ------------------------------run averaging---------------------------------------

ncout_name = avg_tg_data(input_folder, output_folder, variable_name, standard_depth,
                         season)

# ------------------------------inspect data----------------------------------------

ncout_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\18_30yr_avg\\Oxy_0m_OND_tg_30yr_avg.nc'

ncdata = xr.open_dataset(ncout_name)

print(np.nanmin(ncdata.SL_value_30yr_avg.data),
      np.nanmax(ncdata.SL_value_30yr_avg.data),
      np.nanmean(ncdata.SL_value_30yr_avg.data))

png_30yr_avg = plot_linterp_tg_data(
    ncout_name, mforeman_folder, output_folder, variable_name, variable_units_math,
    variable_cmap, standard_depth, '1991_2020', season, avged=True)

# -----------------------------------old--------------------------------------------


def avg_30yr_old(input_dir, output_dir):
    # Initialize dataframe to hold data
    df_mean_all = pd.DataFrame()

    # Convert szn string to corresponding integer
    szn_int = szn_str2int(season)

    # Initialize dataframe for containing 30-year means
    df_szn_mean = pd.DataFrame(
        {"Season": np.repeat(szn_int, num_of_nodes),
         "Longitude [degrees East]": np.repeat(np.nan, num_of_nodes),
         "Latitude [degrees North]": np.repeat(np.nan, num_of_nodes),
         "SL_value_30yr_avg": np.repeat(np.nan, num_of_nodes)})

    # Initialize matrix to hold data values
    means_mtx = np.repeat(np.nan, num_of_nodes * len(years)).reshape((num_of_nodes, len(years)))

    # Iterate through all the years
    for i in range(len(years)):
        data_filename = os.path.join(input_dir + '{}_{}m_{}_tg.nc')
        if os.path.exists(data_filename):
            data_df = pd.read_csv(data_filename)
            means_mtx[:, i] = np.array(data_df['SL_value'])

    df_szn_mean.loc[:, "Longitude [degrees East]"] = np.array(data_df.loc[:, "lon"])
    df_szn_mean.loc[:, "Latitude [degrees North]"] = np.array(data_df.loc[:, "lon"])
    df_szn_mean.loc[:, "SL_value_30yr_avg"] = np.nanmean(means_mtx, axis=1)

    df_mean_all = pd.concat([df_mean_all, df_szn_mean])

    # Export the dataframe of 30 year means as a csv file (to agree with ODV)
    df_mean_filename = os.path.join(output_dir + '{}_{}m_tg_30yr_avg.csv'.format(
        variable_name, standard_depth))

    df_mean_all.to_csv(df_mean_filename, index=False)
    return

