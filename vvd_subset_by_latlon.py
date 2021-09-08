import pandas as pd
import numpy as np
from os.path import basename
import glob


def vvd_subset_latlon(fpath, out_dir):
    lat_min = 30.
    lat_max = 60.
    lon_min = -160.
    lon_max = -115.

    df = pd.read_csv(fpath)

    subsetter = np.where((df.Latitude >= lat_min) &
                         (df.Latitude <= lat_max) &
                         (df.Longitude >= lon_min) &
                         (df.Longitude <= lon_max))[0]

    df_out = df.loc[subsetter]

    df_out_name = basename(fpath)

    df_out.to_csv(out_dir + df_out_name, index=False)

    print(df_out_name)
    print(len(df))
    print(len(df_out))

    return out_dir + df_out_name


indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        'value_vs_depth\\11_stats_check\\'

infiles = glob.glob(indir + '*done.csv')

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\12_latlon_check\\'

for f in infiles:
    vvd_subset_latlon(f, outdir)


