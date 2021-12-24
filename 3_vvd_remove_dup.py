# Filter the value vs depth tables to only include good non-duplicate data

import pandas as pd
import glob
from os.path import basename
import numpy as np


vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\2_added_dup_flags\\exact_duplicate_double_check\\'

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\3_filtered_for_duplicates\\'

for var in ['Temp', 'Sal']:  # ['Oxy']
    # Gather value vs depth files
    vvd_files = glob.glob(vvd_dir + '*{}*.csv'.format(var))

    print(len(vvd_files))
    vvd_files.sort()

    for f in vvd_files:
        # Only use glider data that has had the extra inexact duplicate check
        if 'gld' not in f and 'GLD' in f:
            continue
        else:  # else
            print(basename(f))
            df = pd.read_csv(f)

            # Correct OSD to BOT in WOD files
            OSD_subsetter = np.where(df.Instrument_type == 'OSD')[0]
            df.loc[OSD_subsetter, 'Instrument_type'] = 'BOT'

            # Remove by duplicate flag
            if 'gld' in f:
                subsetter = np.where((df.loc[:, 'Exact_duplicate_flag'] == True) |
                                     (df.loc[:, 'CTD_BOT_duplicate_flag'] == True) |
                                     (df.loc[:, 'Inexact_duplicate_flag'] == True) |
                                     (df.loc[:, 'GLD_Inexact_dup_check'] == True))[0]

                print(len(subsetter))

                df.drop(index=subsetter, inplace=True)

                df.drop(columns=['Exact_duplicate_flag', 'CTD_BOT_duplicate_flag',
                                 'Inexact_duplicate_flag', 'GLD_Inexact_dup_check'],
                        inplace=True)
            else:
                subsetter = np.where((df.loc[:, 'Exact_duplicate_flag'] == True) |
                                     (df.loc[:, 'CTD_BOT_duplicate_flag'] == True) |
                                     (df.loc[:, 'Inexact_duplicate_flag'] == True))[0]

                print(len(subsetter))

                df.drop(index=subsetter, inplace=True)

                df.drop(columns=['Exact_duplicate_flag', 'CTD_BOT_duplicate_flag',
                                 'Inexact_duplicate_flag'], inplace=True)

            # Save df
            outname = basename(f).replace('_dup', '_dup_rm')
            df.to_csv(output_dir + outname, index=False)

