import os
import pandas as pd
import numpy as np
import glob
from clim_helpers import szn_str2int

# Do 30-year averaging on ODV outputs for Lu

# For the odv output .txt files for each depth & each seacon, you will need
# to read 30 files, and calculate the mean for each location. Then generate
# a file in the same format but with mean values.

var_name = "Oxy"
standard_depth = 0
years = range(1991, 2021)
szns = np.array(["JFM", "AMJ", "JAS", "OND"])

odv_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\"

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

# Initialize dataframe to hold all season's data
df_mean_all = pd.DataFrame()

for szn in szns:
    # odv_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\" \
    #           "{}\\{}_{}m_{}\\".format(szn, var_name, standard_depth, szn)

    data_dir = odv_dir + "{}\\{}m\\".format(szn, standard_depth)

    # Read in ODV data
    odv_filename = os.path.join(data_dir + "{}_{}m_{}_TG_{}_est.txt".format(
        var_name, standard_depth, szn, years[0]))

    odv_df = pd.read_csv(odv_filename, sep="\t")

    # Convert szn string to corresponding integer
    szn_int = szn_str2int(szn)

    # Initialize dataframe for containing 30-year means
    df_szn_mean = pd.DataFrame({"Season": np.repeat(szn_int, total_nodes),
                                "Longitude [degrees East]": odv_df.Longitude,
                                "Latitude [degrees North]": odv_df.Latitude,
                                "SL_value_30yr_avg": np.repeat(np.nan, total_nodes)})

    # Initialize matrix for storing values to take 30-year mean of
    output_mtx = np.repeat(
        np.nan, total_nodes * len(years)).reshape((total_nodes, len(years)))

    for i in range(len(years)):
        odv_filename = os.path.join(data_dir + "{}_{}m_{}_TG_{}_est.txt".format(
            var_name, standard_depth, szn, years[i]))

        # Check if file exists
        if os.path.exists(odv_filename):
            # Read in data
            odv_df = pd.read_csv(odv_filename, sep="\t", header=header)

            output_mtx[:, i] = np.array(
                odv_df.loc[:, "SL_value @ YR={}.00".format(years[i])])
        else:
            print("Warning: File for year={} does not exist".format(years[i]))

    # Calculate means
    df_szn_mean.loc[:, "SL_value_30yr_avg"] = np.nanmean(output_mtx, axis=1)

    df_mean_all = pd.concat([df_mean_all, df_szn_mean])

# Remove rows/nodes with mean==NaN
df_mean_all.dropna(subset=["SL_value_30yr_avg"], inplace=True)

# Save the dataframe as a tab-delimited txt file
df_mean_filename = odv_dir + "{}_{}m_30yr_avg.csv".format(var_name, standard_depth)

print(df_mean_filename)

df_mean_all.to_csv(df_mean_filename, index=False)


