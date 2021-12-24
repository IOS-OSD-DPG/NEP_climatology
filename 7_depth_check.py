# To perform on value vs depth data

import pandas as pd
import numpy as np
from tqdm import trange
from clim_helpers import vvd_apply_value_flag
import glob
from os.path import basename


def vvd_depth_check(vvd):
    # vvd: value vs depth dataframe

    # Now do depth checks: inversions and duplicates
    # 0: check passed; 1: range check failed; 2: inversion check failed;
    # 3: range check and inversion check failed; 4: duplicate check failed;
    # 5: range check and duplicate check both failed
    # 6: inversion check and duplicate check both failed
    # 7: range check, inversion check and duplicate check all failed
    vvd['Depth_check_flag'] = np.zeros(len(vvd), dtype=int)

    # Iterate through each profile in nested for loops?
    prof_start_ind = np.unique(vvd.Profile_number, return_index=True)[1]

    for i in trange(len(prof_start_ind)):  # len(prof_start_ind) 10
        if i == len(prof_start_ind) - 1:
            end_ind = len(vvd)
        else:
            # Indexing in pandas includes the end so we need the -1
            end_ind = prof_start_ind[i + 1] - 1

        # Get the depth measurements of the profile
        depths = vvd.loc[prof_start_ind[i]:end_ind, 'Depth_m']

        # print(depths)

        # Check for depths out of range
        # Out of range: above the surface or below 10,000 m
        fail_depth_range = np.where((depths < 0) | (depths > 1e4))[0]

        # Check for depth inversions and copies
        # len(diffs) = len(depths)-1
        # diffs could be an empty array
        diffs = np.diff(depths)
        # pass_check = np.where(diffs > 0)[0]
        fail_inverse = np.where(diffs < 0)[0]
        fail_copy = np.where(diffs == 0)[0]

        # Assign flags to df accordingly
        # +1 to account for len(diffs) = len(depths)-1
        if len(fail_depth_range) > 0:
            vvd.loc[prof_start_ind[i] + 1 + fail_depth_range, 'Depth_check_flag'] += 1
        if len(fail_inverse) > 0:
            vvd.loc[prof_start_ind[i] + 1 + fail_inverse, 'Depth_check_flag'] += 2
        if len(fail_copy) > 0:
            vvd.loc[prof_start_ind[i] + 1 + fail_copy, 'Depth_check_flag'] += 4

        # continue

    print('Number of depth values out of range:',
          len(vvd.loc[vvd.Depth_check_flag == 1, 'Depth_check_flag']))
    print('Number of depth inversions:',
          len(vvd.loc[vvd.Depth_check_flag == 2, 'Depth_check_flag']))
    print('Number of depth duplicates',
          len(vvd.loc[vvd.Depth_check_flag == 4, 'Depth_check_flag']))

    return vvd


# ---------------------------STEP 1: Depth check----------------------------

# nan_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#           'value_vs_depth\\6_filtered_for_nans\\'

nan_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\6_filtered_for_nans\\sep_by_origin\\'

df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                    'value_vs_depth\\7_depth_check\\'

for var in ["Temp", 'Sal']:
    nan_files = glob.glob(nan_dir + '*{}*.csv'.format(var))
    print(len(nan_files))
    # nan_files.remove(nan_dir + 'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv')
    # print(len(nan_files))

    # df = pd.read_csv(nan_files[0])

    # nan_vvd = 'Oxy_1991_2020_value_vs_depth_nan_rm.csv'
    # nan_vvd = 'WOD_PFL_Oxy_1991_2020_value_vs_depth_nan_rm.csv'

    for f in nan_files:
        print(basename(f))
        df_in = pd.read_csv(f)

        df_out = vvd_depth_check(df_in)

        df_outname = df_outdir + basename(f).replace('nan_rm', 'dep_check')

        df_out.to_csv(df_outname, index=False)

        # Apply the flags in a separate df
        # df_in_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        #              'value_vs_depth\\6_depth_check\\' \
        #              'ALL_Oxy_1991_2020_value_vs_depth_dep_check.csv'

        df_in_name = df_outname
        df_in = pd.read_csv(df_in_name)

        df_out = vvd_apply_value_flag(df_in, 'Depth_check_flag')
        df_outname2 = df_in_name.replace('dep_check', 'dep_check_done')
        df_out.to_csv(df_outname2, index=False)

print('done')

