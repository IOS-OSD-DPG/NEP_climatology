import pandas as pd
import numpy as np
import os
from clim_helpers import get_standard_levels
from DIVAnd_fithorzlen import fithorzlen
import time
from clim_helpers import date_string_to_datetime
# import DIVAnd

# ------------------------------------------------
var_name = 'Oxy'
year = 2010
szn = 'OND'
# ------------------------------------------------

# Get array of standard levels
sl_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\WOA_Standard_Depths.txt'
sl_arr = get_standard_levels(sl_dir)

obs_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
          '9_gradient_check\\'

obs_filename = os.path.join(obs_dir + 'Oxy_1991_2020_value_vs_depth_grad_check_done.csv')

obs_df = pd.read_csv(obs_filename)

# Subset dataframe by year and season
# Add pandas datetime column to dataframe
obs_dft = date_string_to_datetime(obs_df)

x = (np.array(obs_df.Longitude[obs_df.Profile_number == 15]),
     np.array(obs_df.Latitude[obs_df.Profile_number == 15]),
     np.array(obs_df.Depth_m[obs_df.Profile_number == 15]))
value = np.array(obs_df.Value[obs_df.Profile_number == 15])

# Subset the standard level array according to the range of the input data
sl_subsetter = np.where((sl_arr >= np.min(x[2])) & (sl_arr <= np.max(x[2])))
sl_arr_subset = sl_arr[sl_subsetter]

# xjulia = tuple([np.transpose(_) for _ in x])

start_time = time.time()

# Does sl_arr need to be within the range of x's depth?
lenxy, infoxy = fithorzlen(x, value, sl_arr_subset)

execution_time = time.time() - start_time
print('Execution time:', execution_time, 's')

print('len', lenxy)
# print(infoxy)

# obs_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
#           '14_sep_by_sl_and_year\\'
# obs_list = list(map(lambda x: os.path.join(obs_dir + '{}_{}m_{}_{}.csv'.format(
#     var_name, x, year, szn)), sl_arr))
#
# print(obs_list[0])
#
# # Initialize dataframe to hold x=tuple(longitude, latitude, depth)
# df_x = pd.DataFrame()
#
# for fname in obs_list:
#     # Extract the standard depth of the data from each file name
#     standard_depth = int(os.path.basename(fname)[4:-14])
#     # Read observational data into pandas dataframe
#     df_obs = pd.read_csv(fname)
#     # Add column of standard depth to df_obs
#     df_obs['Standard_depth'] = np.repeat(standard_depth, len(df_obs))
#     # Concatenate the observations with the x dataframe
#     df_x = pd.concat([df_x, df_obs])
#
# # Extract dataframe columns to a tuple of arrays
# x = (np.array(df_x.Longitude), np.array(df_x.Latitude), np.array(df_x.Standard_depth))
# value = np.array(df_x.SL_value)

