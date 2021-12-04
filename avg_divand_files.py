import numpy as np
import pandas as pd
import os
from clim_helpers import szn_str2int

# Perform 30-year averaging on DIVAnd files (files from DIVAnd analysis)

# Set parameters
variable_name = "Oxy"
years = np.arange(1991, 2021)
season_abbrev = "JAS"  # ["JFM", "AMJ", "JAS", "OND"] 'spr'
season_abbrevs = ["JFM", "AMJ", "JAS", "OND"]
standard_depth = '0'
nnodes = 97959

data_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
           "value_vs_depth\\17_lin_interp_to_trigrid\\"
output_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
             "value_vs_depth\\18_30yr_avg\\"

# Initialize df to hold means output
df_mean_all = pd.DataFrame()

# Iterate through seasons

for sa in season_abbrevs:
    sa_int = szn_str2int(sa)

    # Initialize dataframe for containing 30-year means
    df_mean_szn = pd.DataFrame({"Season": np.repeat(sa_int, nnodes),
                                "Longitude [degrees East]": odv_df.Longitude,
                                "Latitude [degrees North]": odv_df.Latitude,
                                "SL_value_30yr_avg": np.repeat(np.nan, nnodes)})

    # Initialize matrix (node_num x year) dims
    data_mtx = np.repeat(
        np.nan, nnodes * len(years)).reshape(nnodes, len(years))

    for i in range(len(years)):
        # This is a made-up filename structure
        data_filename = os.path.join(data_dir + "{}_{}m_{}_{}_TG.csv".format(
            variable_name, standard_depth, years[i], sa))
        data_df = pd.read_csv(data_filename)

        data_mtx[i, :] = np.array(data_df.SL_value)

    df_mean_szn.loc[:, "SL_value_30yr_avg"] = np.nanmean(data_mtx, axis=1)

    df_mean_all = pd.concat([df_mean_all, df_mean_szn])

df_mean_all.dropna(subset=["SL_value_30yr_avg"], inplace=True)

df_mean_filename = ""
df_mean_all.to_csv(df_mean_filename, index=False)