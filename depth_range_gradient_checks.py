# To perform on value vs depth data

import pandas as pd
import numpy as np
from tqdm import trange
import glob


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


def vvd_range_check(vvd, range_df):
    vvd['Range_check_flag'] = np.zeros(len(vvd), dtype=int)

    for i in trange(len(vvd)):  # len(df) 10
        # Want to find the last depth in the range_df that the i-th depth is greater than?
        # cond = np.where(range_df.loc['Depth_m'] > df.loc[i, 'Depth_m'])[0]

        for j in range(len(range_df)):
            depth_cond = range_df.loc[j, 'Depth_min'] <= vvd.loc[
                i, 'Depth_m'] <= range_df.loc[j, 'Depth_max']
            range_cond = range_df.loc[j, 'N_Pacific_min'] <= vvd.loc[
                i, 'Value'] <= range_df.loc[j, 'N_Pacific_max']

            if depth_cond and not range_cond:
                # Flag the df row if value is out of range
                vvd.loc[i, 'Range_check_flag'] = 1

    return vvd


def vvd_gradient_check(df, grad_df, verbose=False):
    # Value vs depth gradient check
    # Check for gradients, inversions and zero sensitivity
    # df: value vs depth dataframe
    # grad_df: dataframe from WOA18 containing maximum gradient, inversion,
    #          and zero sensitivity index values to check vvd data against

    df['Gradient_check_flag'] = np.zeros(len(df), dtype=int)

    prof_start_ind = np.unique(df.Profile_number, return_index=True)[1]

    # Iterate through all of the profiles
    for i in trange(len(prof_start_ind)):  # len(prof_start_ind) 20
        # print(prof_start_ind[i])

        # Set profile end index
        if i == len(prof_start_ind) - 1:
            end_ind = len(df)
        else:
            # Pandas indexing is inclusive so need the -1
            end_ind = prof_start_ind[i + 1]

        # Get profile data; np.arange not inclusive of end which we want here
        indices = np.arange(prof_start_ind[i], end_ind)
        depths = df.loc[indices, 'Depth_m']
        values = df.loc[indices, 'Value']

        if verbose:
            print('Got values')

        # Try to speed up computations by skipping profiles with only 1 measurement
        if len(depths) <= 1:
            continue
        else:
            # gradients = np.zeros(len(depths), dtype=float)
            # for j in range(len(depths) - 1):
            # gradients[i] = (values[i + 1] - values[i]) / (depths[i + 1] - depths[i])

            # Use numpy built-in gradient method (uses 2nd order central differences)
            # Need fix for divide by zero
            gradients = np.gradient(values, depths)

            # Find the rate of change of gradient
            d_gradients = np.diff(gradients)

            # Create flags accordingly
            # If depth <= 400m and gradient < -max, apply one set of criteria
            # If depth > 400m and gradient < -max, apply other set of criteria...
            subsetter_MGV_lt_400 = np.where((depths <= 400) &
                                            (gradients < -grad_df.loc['Oxygen', 'MGV_Z_lt_400m']))[0]
            subsetter_MGV_gt_400 = np.where((depths > 400) &
                                            (gradients < -grad_df.loc['Oxygen', 'MGV_Z_gt_400m']))[0]
            subsetter_MIV_lt_400 = np.where((depths <= 400) &
                                            (gradients > grad_df.loc['Oxygen', 'MIV_Z_lt_400m']))[0]
            subsetter_MIV_gt_400 = np.where((depths > 400) &
                                            (gradients > grad_df.loc['Oxygen', 'MIV_Z_gt_400m']))[0]

            if verbose:
                print('Created MGV/MIV subsetters')

            # Zero sensitivity check
            # Only flag observations with Value = 0
            # If there are zero-as-missing-values at the very surface, then
            # the ZSI check wouldn't find them because it needs the gradient
            subsetter_ZSI_lt_400 = np.where(
                (depths[1:] <= 400) &
                (d_gradients < -grad_df.loc[
                    'Oxygen', 'MGV_Z_lt_400m'] * grad_df.loc['Oxygen', 'ZSI']) &
                (values[1:] == 0.))[0]
            subsetter_ZSI_gt_400 = np.where(
                (depths[1:] > 400) &
                (d_gradients < -grad_df.loc[
                    'Oxygen', 'MGV_Z_gt_400m'] * grad_df.loc['Oxygen', 'ZSI']) &
                (values[1:] == 0.))[0]

            if verbose:
                print('Created ZSI subsetters')

            # Flag the observations that failed the checks
            # "indices" span prof_start_ind[i] to the end of the profile
            df.loc[indices[np.union1d(subsetter_MGV_lt_400, subsetter_MGV_gt_400)],
                   'Gradient_check_flag'] = 1
            df.loc[indices[np.union1d(subsetter_MIV_lt_400, subsetter_MIV_gt_400)],
                   'Gradient_check_flag'] = 2

            # Flag = 3 for ZSI check failed
            # Flag = 4, for ZSI check and gradient check failed
            # Flag = 5 for ZSI check and inversion check failed
            df.loc[indices[np.union1d(subsetter_ZSI_lt_400, subsetter_ZSI_gt_400)],
                   'Gradient_check_flag'] += 3

    return df


def vvd_apply_value_flag(df, flag_name):
    # Apply flag and generate new copy of df
    # Flag=0 means data passed the check so want to keep that

    df = df.loc[df[flag_name] == 0]

    df_return = df.drop(columns=flag_name)

    return df_return


##### STEP 1: Depth check
nan_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\5_filtered_for_nans\\'

nan_files = glob.glob(nan_dir + '*.csv')
nan_files.remove(nan_dir + 'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv')

# df = pd.read_csv(nan_files[0])

nan_vvd = 'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv'

df_in = pd.read_csv(nan_dir + nan_vvd)

df_out = vvd_depth_check(df_in)
df_outname = nan_vvd.replace('nan_rm', 'dep_check')
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\6_depth_check\\'

df_out.to_csv(df_outdir + df_outname, index=False)

# Apply the flags in a separate df
df_in_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\6_depth_check\\' \
             'ALL_Oxy_1991_2020_value_vs_depth_dep_check.csv'

df_in = pd.read_csv(df_in_name)

df_out = vvd_apply_value_flag(df_in, 'Depth_check_flag')

df_out_name = df_in_name.replace('dep_check', 'dep_check_done')
df_out.to_csv(df_out_name, index=False)


##### STEP 2: Range check
# Now do range checks: flag=0 if check passed; flag=1 if check failed
# Use preset ranges from WOD
range_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\literature\\' \
             'WOA docs\\wod18_users_manual_tables\\wod18_ranges_DOXY_N_Pac.csv'
df_range = pd.read_csv(range_file)

# What is "Coastal Pacific" defined as?
# "depending on the number of one-degree by one-degree
# latitude-longitude grid boxes in the five-degree box which were land areas"

# Use North Pacific not Coastal North Pacific values for now

df_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\6_depth_check\\'

df_file = 'ALL_Oxy_1991_2020_value_vs_depth_dep_check_done.csv'

df_in = pd.read_csv(df_dir + df_file)

# Range check function
df_out = vvd_range_check(df_in, df_range)

print(len(df_out.loc[df_out.Range_check_flag == 1, 'Range_check_flag']))
indices_rng = np.where(df_out.Range_check_flag == 1)[0]
print(df_out.loc[indices_rng, ['Depth_m', 'Value']])
# 24174 rows that were flagged during this check: should limits be expanded??

# df_out.drop(columns='Depth_check_flag', inplace=True)

df_outname = df_file.replace('dep_check_done', 'rng_check')
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\7_range_check\\'
df_out.to_csv(df_outdir + df_outname, index=False)

df_rng_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\7_range_check\\' \
              'ALL_Oxy_1991_2020_value_vs_depth_rng_check.csv'

df_in_fm = pd.read_csv(df_rng_name)

df_out_fm = vvd_apply_value_flag(df_in_fm, 'Range_check_flag')

df_out_name_fm = df_rng_name.replace('rng_check', 'rng_check_done')
df_out_fm.to_csv(df_out_name_fm, index=False)


##### STEP 3: Gradient checks
# Now do gradient checks: flag=1 if check failed; flag=0 if check passed

df_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\7_range_check\\'

df_file = 'ALL_Oxy_1991_2020_value_vs_depth_rng_check_done.csv'

df_in = pd.read_csv(df_dir + df_file)

# Read in table of WOD18 maximum gradients and inversions
grad_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\literature\\' \
            'WOA docs\\wod18_users_manual_tables\\wod18_max_gradient_inversion.csv'
df_grad = pd.read_csv(grad_file, index_col='Variable')

# Run gradient check
df_out = vvd_gradient_check(df_in, df_grad)

print('Done')

# Print summary statistics
print(len(df_out.loc[df_out.Gradient_check_flag == 1, 'Gradient_check_flag']))  # gradient
print(len(df_out.loc[df_out.Gradient_check_flag == 2, 'Gradient_check_flag']))  # inversion
print(len(df_out.loc[df_out.Gradient_check_flag == 3, 'Gradient_check_flag']))  # ZSI
print(len(df_out.loc[df_out.Gradient_check_flag == 4, 'Gradient_check_flag']))  # ZSI and gradient
print(len(df_out.loc[df_out.Gradient_check_flag == 5, 'Gradient_check_flag']))  # ZSI and inversion

df_outname = df_file.replace('rng_check_done', 'grad_check')
print(df_outname)
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\8_gradient_check\\'

df_out.to_csv(df_outdir + df_outname, index=False)

df_out2 = vvd_apply_value_flag(df_out, 'Gradient_check_flag')

df_out2_name = df_outname.replace('grad_check', 'grad_check_done')
df_out2.to_csv(df_outdir + df_out2_name, index=False)
