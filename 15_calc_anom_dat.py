"""
Reformat the standard level data files to the data.dat format required
by Diva PythonTools
"""

import pandas as pd
import glob
import numpy as np
from os.path import basename
from tqdm import trange

indir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
        "value_vs_depth\\14_sep_by_sl_and_year\\"

infiles = glob.glob(indir + '*.csv', recursive=False)
infiles.sort()
print(len(infiles))

outdir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
         "value_vs_depth\\15_minus_mean_dat\\"

# Initialize dataframe to contain the mean of each file
df_means = pd.DataFrame(columns=['File_mean'],
                        index=list(map(lambda f: basename(f), infiles)))

# Do I need to separate by month, not just by season?
# And take the mean of each month of each year?

# Iterate through the files
for i in trange(len(infiles)):
    dfin = pd.read_csv(infiles[i])

    # Check if dfin is empty
    if dfin.empty:
        sl_mean = np.NaN
    else:
        sl_mean = np.mean(dfin.SL_value)
        # Calculate the anomaly by subtracting the mean
        dfin['SL_anom'] = dfin.SL_value - sl_mean
        # Arrange into dat format
        arrout = np.array([np.array(dfin.Longitude), np.array(dfin.Latitude),
                           np.array(dfin.SL_anom)]).transpose()
        # Export as a text file
        outname = basename(infiles[i]).replace('.csv', '_anom.dat')
        # Set delimiter to be a space
        np.savetxt(outdir + outname, arrout, delimiter=' ')

    # Save filename and mean to a dataframe
    df_means.loc[basename(infiles[i]), 'File_mean'] = sl_mean

    # continue

df_means_name = 'Oxy_1991_2020_anom_means.csv'
df_means.to_csv(outdir + df_means_name, index=True)

# Next, need to compute variance of each background field (varbak)...
