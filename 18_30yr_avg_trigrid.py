import os
import numpy as np
import pandas as pd
from clim_helpers import szn_str2int

variable_name = 'Oxy'
standard_depth = 0
years = np.arange(1991, 2021)
# season = 'JFM'
seasons = ['JFM', 'AMJ', 'JAS', 'OND']

num_of_nodes = 97959

# Use the files that have been linearly interpolated to the triangle grid
# Have the files be in .npy format or .csv format?
input_dir = '/home/hourstonh/Documents/climatology/data/value_vs_depth/17_lin_interp_to_trigrid/'
output_dir = '/home/hourstonh/Documents/climatology/data/value_vs_depth/18_30yr_avg/'

# Initialize dataframe to hold all season's data
df_mean_all = pd.DataFrame()

for s in seasons:
	# Convert szn string to corresponding integer
    szn_int = szn_str2int(s)

    # Initialize dataframe for containing 30-year means
    df_szn_mean = pd.DataFrame(
		{"Season": np.repeat(szn_int, num_of_nodes),
         "Longitude [degrees East]": np.repeat(np.nan, total_nodes),
         "Latitude [degrees North]": np.repeat(np.nan, total_nodes),
         "SL_value_30yr_avg": np.repeat(np.nan, total_nodes)})
	
	# Initialize matrix to hold data values
	means_mtx = np.repeat(np.nan, num_of_nodes * len(years)).reshape((num_of_nodes, len(years)))

	# Iterate through all the years
	for i in range(len(years)):
		data_filename = os.path.join(input_dir + '{}_{}m_{}.csv')
		data_df = pd.read_csv(data_filename)
		means_mtx[:, i] = np.array(data_df['SL_value'])

	df_szn_mean.loc[:, "Longitude [degrees East]"] = np.array(data_df.loc[:, "lon"])
	df_szn_mean.loc[:, "Latitude [degrees North]"] = np.array(data_df.loc[:, "lon"])
	df_szn_mean.loc[:, "SL_value_30yr_avg"] = np.nanmean(means_mtx, axis=1)

	df_mean_all = pd.concat([df_mean_all, df_szn_mean])

# Export the dataframe of 30 year means as a csv file (to agree with ODV)
df_mean_filename = os.path.join(output_dir + '{}_{}m_30yr_avg.csv'.format(
	variable_name, standard_depth))

df_mean_all.to_csv(df_mean_filename, index=False)


	

