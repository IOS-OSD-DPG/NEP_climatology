import os
import pandas as pd
import numpy as np
# import glob
from clim_helpers import szn_str2int
from tqdm import trange

# Do 30-year averaging on ODV outputs for Lu

# For the odv output .txt files for each depth & each seacon, you will need
# to read 30 files, and calculate the mean for each location. Then generate
# a file in the same format but with mean values.

var_name = "Temp"  # ['Temp', 'Sal', 'Oxy']
# standard_depth = 10
years = range(1991, 2021)
szns = np.array(["JFM", "AMJ", "JAS", "OND"])

odv_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\"

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\' \
             'avg_30yr\\'

# --------------------------Correct season in file names-----------------------------

# # Change file names for OND since season is wrong (shouldn't be JFM)
# odv_files = glob.glob(odv_dir + "{}_{}m_*_est.txt".format(var_name, standard_depth),
#                       recursive=False)
#
# print(odv_files)
# print(len(odv_files))
#
# for f in odv_files:
#     new_name = os.path.basename(f).replace("JAS", "AMJ")
#     os.rename(f, odv_dir + new_name)

# -----------------------------------------------------------------------------------

# Set parameters for reading text files
header = 0
total_nodes = 97959

for standard_depth in [0]:  # 80, 150,  trange(1200, 2500, 200):  # trange(260, 420, 20):
    # Initialize dataframe to hold all season's data
    df_mean_all = pd.DataFrame()

    for szn in szns:
        # odv_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\" \
        #           "{}\\{}_{}m_{}\\".format(szn, var_name, standard_depth, szn)

        # Convert szn string to corresponding integer
        szn_int = szn_str2int(szn)

        data_dir = os.path.join(odv_dir, "{}_{}".format(szn_int, szn),
                                "{}_{}m_{}".format(var_name, standard_depth, szn))

        # odv_df = pd.read_csv(odv_filename, sep="\t")

        # Initialize dataframe for containing 30-year means
        df_szn_mean = pd.DataFrame({"Season": np.repeat(szn_int, total_nodes),
                                    "Longitude [degrees East]": np.repeat(np.nan, total_nodes),
                                    "Latitude [degrees North]": np.repeat(np.nan, total_nodes),
                                    "SL_value_30yr_avg": np.repeat(np.nan, total_nodes)})

        # Initialize matrix for storing values to take 30-year mean of
        output_mtx = np.repeat(
            np.nan, total_nodes * len(years)).reshape((total_nodes, len(years)))

        for i in range(len(years)):
            # Read in ODV data
            # File names are not consistent so need to check which name to use
            odv_filename = os.path.join(data_dir, "{}_{}m_{}_TG_{}_est.txt".format(
                var_name, standard_depth, szn, years[i]))
            # Check if filename is correct
            if not os.path.exists(odv_filename):
                odv_filename = os.path.join(data_dir, "{}_{}_TG_{}_est.txt".format(
                    var_name, szn, years[i]))
            # Check if filename is correct
            if not os.path.exists(odv_filename):
                odv_filename = os.path.join(data_dir, "{}_{}_TG_mean_{}_est.txt".format(
                    var_name, szn, years[i]))
            # Check if filename is correct
            if not os.path.exists(odv_filename):
                odv_filename = os.path.join(data_dir, "{}_{}_TG_{}.txt".format(
                    var_name, szn, years[i]))

            # Check if file exists
            if os.path.exists(odv_filename):
                print(odv_filename)
                # Read in data
                odv_df = pd.read_csv(odv_filename, sep="\t", header=header)

                output_mtx[:, i] = np.array(
                    odv_df.loc[:, "SL_value @ YR={}.00".format(years[i])])
            else:
                print("Warning: File for depth={} szn={} year={} does not exist".format(
                    standard_depth, szn, years[i]))

        # Populate lat and lon in dataframe
        df_szn_mean.loc[:, 'Longitude [degrees East]'] = odv_df.Longitude
        df_szn_mean.loc[:, 'Latitude [degrees North]'] = odv_df.Latitude

        # Calculate means
        df_szn_mean.loc[:, "SL_value_30yr_avg"] = np.nanmean(output_mtx, axis=1)

        df_mean_all = pd.concat([df_mean_all, df_szn_mean])

    # Remove rows/nodes with mean==NaN
    df_mean_all.dropna(subset=["SL_value_30yr_avg"], inplace=True)

    # Save the dataframe as a tab-delimited txt file
    df_mean_filename = os.path.join(
        output_dir, "{}_{}m_30yr_avg.csv".format(var_name, standard_depth))

    print(df_mean_filename)

    df_mean_all.to_csv(df_mean_filename, index=False)

# Check for nans in dataframe
print(np.where(pd.isna(df_mean_all)))

# ---------------------------------------------------------------------------------

# Average DIVAnd files that have already been projected to the triangle grid

f2008 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\1_JFM\\' \
        'Oxy_80m_JFM\\Oxy_JFM_TG_2008_est.txt'

f2009 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\1_JFM\\' \
        'Oxy_80m_JFM\\Oxy_JFM_TG_2009_est.txt'

df2008 = pd.read_csv(f2008, sep='\t')
df2009 = pd.read_csv(f2009, sep='\t')

df2008.loc[~pd.isnan(df2008.loc[:,"Value"])]