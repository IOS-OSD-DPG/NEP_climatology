# To perform on value vs depth data


import pandas as pd
import numpy as np
from tqdm import trange
import glob


def vvd_depth_check(vvd):
    # vvd: value vs depth dataframe

    # Now do depth checks: inversions and duplicates
    # 0: check passed; 1: inversion check failed; 2: duplicate check failed;
    # 3: inversion check and duplicate check both failed
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

        if len(depths) == 1:
            # Skip the rest because it's redundant
            continue
        else:
            # print(depths)

            # len(diffs) = len(depths)-1
            # diffs could be an empty array
            diffs = np.diff(depths)
            # pass_check = np.where(diffs > 0)[0]
            fail_inverse = np.where(diffs < 0)[0]
            fail_copy = np.where(diffs == 0)[0]

            # Assign flags to df accordingly
            # +1 to account for len(diffs) = len(depths)-1
            if len(fail_inverse) > 0:
                vvd.loc[prof_start_ind[i] + 1 + fail_inverse, 'Depth_check_flag'] += 1
            if len(fail_copy) > 0:
                vvd.loc[prof_start_ind[i] + 1 + fail_copy, 'Depth_check_flag'] += 2

            # continue

    print('Number of depth inversions:',
          len(vvd.loc[vvd.Depth_check_flag == 1, 'Depth_check_flag']))
    print('Number of depth duplicates',
          len(vvd.loc[vvd.Depth_check_flag == 2, 'Depth_check_flag']))

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


def vvd_gradient_check(df, grad_df):
    df['Gradient_check_flag'] = np.zeros(len(df), dtype=int)

    prof_start_ind = np.unique(df.Profile_number, return_index=True)[1]

    for i in trange(len(prof_start_ind)):  # len(prof_start_ind) 20
        # Set profile end index
        if i == len(prof_start_ind) - 1:
            end_ind = len(df)
        else:
            # Pandas indexing is inclusive so need the -1
            end_ind = prof_start_ind[i + 1] - 1

        # Need the +1 because indices would otherwise not go up to and include second last elem?
        indices = np.arange(prof_start_ind[i], end_ind + 1)
        depths = df.loc[indices, 'Depth_m']
        values = df.loc[indices, 'Value']

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
            # What to do about Z = 400 m??
            # If depth < 400m and gradient < -max, apply one set of criteria
            # If depth > 400m and gradient < -max, apply other set of criteria
            subsetter_MGV_lt_400 = np.where((depths < 400) &
                                            (gradients < -grad_df.loc['Oxygen', 'MGV_Z_lt_400m']))[0]
            subsetter_MGV_gt_400 = np.where((depths > 400) &
                                            (gradients < -grad_df.loc['Oxygen', 'MGV_Z_gt_400m']))[0]
            subsetter_MIV_lt_400 = np.where((depths < 400) &
                                            (gradients > grad_df.loc['Oxygen', 'MIV_Z_lt_400m']))[0]
            subsetter_MIV_gt_400 = np.where((depths > 400) &
                                            (gradients > grad_df.loc['Oxygen', 'MIV_Z_gt_400m']))[0]

            # Zero sensitivity check
            # Only flag observations with Value = 0
            # If there are zero-as-missing-values at the very surface, then
            # the ZSI check wouldn't find them because it needs the gradient
            subsetter_ZSI_lt_400 = np.where(
                (depths[1:] < 400) &
                (d_gradients < -grad_df.loc[
                    'Oxygen', 'MGV_Z_lt_400m'] * grad_df.loc['Oxygen', 'ZSI']) &
                (values[1:] == 0.))[0]
            subsetter_ZSI_gt_400 = np.where(
                (depths[1:] > 400) &
                (d_gradients < -grad_df.loc[
                    'Oxygen', 'MGV_Z_gt_400m'] * grad_df.loc['Oxygen', 'ZSI']) &
                (values[1:] == 0.))[0]

            # Flag the observations that failed the checks
            df.loc[indices[subsetter_MGV_lt_400 | subsetter_MGV_gt_400], 'Gradient_check_flag'] = 1
            df.loc[indices[subsetter_MIV_lt_400 | subsetter_MIV_gt_400], 'Gradient_check_flag'] = 2

            # Flag = 3 for ZSI check failed
            # Flag = 4, for ZSI check and gradient check failed
            # Flag = 5 for ZSI check and inversion check failed
            df.loc[indices[subsetter_ZSI_lt_400 | subsetter_ZSI_gt_400], 'Gradient_check_flag'] += 3

    return df


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

# Export the updated dataframe
# When to apply these flags?


##### STEP 2
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

df_file = 'ALL_Oxy_1991_2020_value_vs_depth_dep_check.csv'

df = pd.read_csv(df_dir + df_file)

# Range check function
df_out = vvd_range_check(df, df_range)

print(len(df_out.loc[df_out.Range_check_flag == 1, 'Range_check_flag']))
indices_rng = np.where(df_out.Range_check_flag == 1)[0]
print(df_out.loc[indices_rng, ['Depth_m', 'Value']])
# 24176 rows that were flagged during this check: should limits be expanded??

# Export the updated dataframe
# When to apply these flags?

df_outname = df_file.replace('dep_check', 'rng_check')
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\7_range_check\\'
df.to_csv(df_outdir + df_outname, index=False)


### STEP 3
# Now do gradient checks: flag=1 if check failed; flag=0 if check passed

df_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\7_range_check\\'

df_file = 'ALL_Oxy_1991_2020_value_vs_depth_rng_check.csv'

df_in = pd.read_csv(df_dir + df_file)

# Read in table of WOD18 maximum gradients and inversions
grad_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\literature\\' \
            'WOA docs\\wod18_users_manual_tables\\wod18_max_gradient_inversion.csv'
df_grad = pd.read_csv(grad_file, index_col='Variable')

# 0: good; 1: excessive gradient (negative gradient over depth);
# 2: excessive inversion (positive gradient over depth)
# 3: Zero sensitivity check failed ????????

df_out = vvd_gradient_check(df_in, df_grad)

print(len(df_out.loc[df_out.Gradient_check_flag == 1, 'Gradient_check_flag']))  # gradient
print(len(df_out.loc[df_out.Gradient_check_flag == 2, 'Gradient_check_flag']))  # inversion
print(len(df_out.loc[df_out.Gradient_check_flag == 3, 'Gradient_check_flag']))  # ZSI
print(len(df_out.loc[df_out.Gradient_check_flag == 4, 'Gradient_check_flag']))  # ZSI and gradient
print(len(df_out.loc[df_out.Gradient_check_flag == 5, 'Gradient_check_flag']))  # ZSI and inversion

df_outname = df_file.replace('rng_check', 'grad_check')
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\8_gradient_check\\'

df_out.to_csv(df_outdir + df_outname, index=False)

# Export the updated dataframe
# When to apply these flags?
