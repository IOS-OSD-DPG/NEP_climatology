from clim_helpers import plot_divand_analysis
import os
from xarray import open_dataset

plt_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
          "16_diva_analysis\\"
field_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
            "16_diva_analysis\\analysis\\"

field_filename = os.path.join(field_dir + "Oxy_0m_2010_OND_analysis2d_guess.nc")

field_ds = open_dataset(field_filename)

# def plot_divand_analysis(output_dir, lon2d, lat2d, var_field, var_cmap, var_name, var_units,
#                          lon_obs, lat_obs, depth, yr, szn, nle_val):

