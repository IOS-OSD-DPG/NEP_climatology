# Filter the value vs depth tables to only include good non-duplicate data

import pandas as pd
import glob
from os.path import basename
import numpy as np


vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\2_added_dup_flags\\exact_duplicate_double_check\\'

vvd_files = glob.glob(vvd_dir + 'WOD_PFL_Oxy*.csv')
print(len(vvd_files))
vvd_files.sort()

output_dir1 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\3_filtered_for_duplicates\\'

output_dir2 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\4_filtered_for_quality_flag\\'

for f in vvd_files:
    print(basename(f))
    df = pd.read_csv(f)

    # Remove by duplicate flag
    subsetter = np.where((df.Exact_duplicate_flag == True) |
                         (df.CTD_BOT_duplicate_flag == True) |
                         (df.Inexact_duplicate_flag == True))[0]

    print(len(subsetter))

    df.drop(index=subsetter, inplace=True)

    df.drop(columns=['Exact_duplicate_flag', 'CTD_BOT_duplicate_flag',
                     'Inexact_duplicate_flag'], inplace=True)

    # Save df
    outname = basename(f).replace('_dup', '_dup_rm')
    df.to_csv(output_dir1 + outname, index=False)

