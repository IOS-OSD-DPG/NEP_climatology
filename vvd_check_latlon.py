import pandas as pd
import numpy as np
from os.path import basename


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

    df_out_name = basename(fpath).replace('dup_rm.csv', 'latlon.csv')

    df_out.to_csv(out_dir + df_out_name, index=False)

    print(df_out_name)
    print('In df length', len(df))
    print('Out df length', len(df_out))

    return out_dir + df_out_name

