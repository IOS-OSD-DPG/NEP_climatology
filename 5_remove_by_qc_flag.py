import glob
from os.path import basename
import pandas as pd
from copy import deepcopy
import numpy as np

# 4: Now for the qc flags

variable_name = 'Temp'  # Sal Oxy

# IOS data ==1 means good quality
# NODC data ==0 means good quality; ==1 means range_out, so bad quality
# MEDS data: 1=data is good, 3=suspicious, 4=bad

# nodc_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
#             'WOD_extracts\\Oxy_WOD_May2021_extracts\\Oxy_1991_2020_AMJ_OSD.nc'
#
# nc = open_dataset(nodc_file)

input_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\4_latlon_check\\'

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\5_filtered_for_quality_flag\\'

# files = glob.glob(input_dir + 'WOD_PFL_Oxy*.csv')

for var in ['Temp', 'Sal']:
    files = glob.glob(input_dir + '*{}*.csv'.format(var))
    print(len(files))

    for f in files:
        print(basename(f))
        df = pd.read_csv(f)

        # Drop by quality flag
        if 'WOD' in basename(f):
            # keep source flag == 0, drop the rest
            subsetter = np.where((df.Source_flag.astype(int) == 0) &
                                 (df.Depth_flag.astype(int) == 0))[0]
        elif 'MEDS' in basename(f):
            # Remove XBT since instrument models unknown and would need that
            # info to correct temperature bias
            subsetter = np.where((df.Source_flag == 1.) &
                                 (df.Depth_flag == 1.) &
                                 (df.Instrument_type != 'XBT'))[0]
        else:
            # keep source flag == 1, drop the rest
            subsetter = np.where((df.Source_flag.astype(int) == 1) &
                                 (df.Depth_flag.astype(int)) == 1)[0]

        print(len(subsetter))
        df_new = deepcopy(df.loc[subsetter])

        df_new.drop(columns=['Depth_flag', 'Source_flag'], inplace=True)

        outname = basename(f).replace('latlon', 'qc')
        df_new.to_csv(output_dir + outname, index=False)


