from clim_helpers import plot_divand_analysis
import os
from xarray import open_dataset
import numpy as np
from pandas import read_csv

colourmap = "Blues"
variable = "Oxy"
variable_units = r"$\mu$" + "mol/kg"
standard_depth = 0
years = [1991]  # np.arange(1992, 2021)
season = "JFM"
# nle_value = 1
lenxy_method = "fithorzlen"  # correlation length estimation method

plt_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
          "16_diva_analysis\\analysis\\fithorzlen\\"
field_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
            "16_diva_analysis\\analysis\\fithorzlen\\"
obs_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
          "14_sep_by_sl_and_year\\"


for y in years:
    field_filename = os.path.join(field_dir + "{}_{}m_{}_{}_analysis2d.nc".format(
        variable, standard_depth, y, season))

    # Skip any files that don't exist
    if not os.path.exists(field_filename):
        continue    

    print(os.path.basename(field_filename))
    field_ds = open_dataset(field_filename)

    obs_filename = os.path.join(obs_dir + "{}_{}m_{}_{}.csv".format(
        variable, standard_depth, y, season))
    obs_df = read_csv(obs_filename)

    Lon2d, Lat2d = np.meshgrid(field_ds.longitude.data, field_ds.latitude.data)
    var_field = field_ds.vout.data

    print("Min, max, mean:", np.nanmin(var_field), np.nanmax(var_field), np.nanmean(var_field))

    # def plot_divand_analysis(output_dir, lon2d, lat2d, var_field, var_cmap, var_name, var_units,
    #                          lon_obs, lat_obs, depth, yr, szn, nle_val):

    plot_filename = plot_divand_analysis(plt_dir, Lon2d, Lat2d, var_field,
                                         colourmap, variable, variable_units,
                                         np.array(obs_df.Longitude), np.array(obs_df.Latitude),
                                         standard_depth, y, season, lenxy_method)

    field_ds.close()
