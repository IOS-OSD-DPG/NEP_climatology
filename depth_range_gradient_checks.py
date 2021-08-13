# To perform on value vs depth data


import pandas as pd
import numpy as np
import glob


nan_vvd = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\5_filtered_for_nans\\' \
          'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv'

df = pd.read_csv(nan_vvd)

# Maybe make a new Profile_number column?
df['Profile_number_new'] = np.zeros(len(df), dtype=int)

counter = 0
for i in range(1, len(df)):
    if df.loc[i, 'Profile_number'] != df.loc[i-1, 'Profile_number']:
        counter += 1
        df.loc[i, 'Profile_number_new'] = counter


# Now do depth checks: inversions and duplicates
# 0: check passed; 1: inversion check failed; 2: duplicate check failed;
# 3: inversion check and duplicate check both failed
df['Depth_check_flag'] = np.zeros(len(df), dtype=int)

# Iterate through each profile in nested for loops?
prof_start_ind = np.unique(df.Profile_number_new, return_index=True)[1]

for i in range(len(prof_start_ind)):
    if i == len(prof_start_ind) - 1:
        end_ind = len(prof_start_ind)
    else:
        end_ind = prof_start_ind[i + 1]

    # Get the depth measurements of the profile
    depths = df.loc[prof_start_ind[i]:end_ind, 'Depth_m']

    # len(diffs) = len(depths)-1
    diffs = np.diff(depths)
    pass_check = np.where(diffs > 0)[0]
    fail_inverse = np.where(diffs < 0)[0]
    fail_copy = np.where(diffs == 0)[0]

    # Assign flags to df accordingly
    df[prof_start_ind + 1 + fail_inverse, 'Depth_check_flag'] += 1
    df[prof_start_ind + 1 + fail_copy, 'Depth_check_flag'] += 2

    continue

# Export the updated dataframe
# When to apply these flags?

# Now do range checks: flag=0 if check passed; flag=1 if check failed
# Use preset ranges from WOD
df['Range_check_flag'] = np.zeros(len(df), dtype=int)

# Export the updated dataframe
# When to apply these flags?

# Now do gradient checks: flag=1 if check failed; flag=0 if check passed
df['Gradient_check_flag'] = np.zeros(len(df), dtype=int)

# Define maximum gradient amount
max_grad = 99

for i in range(len(prof_start_ind)):
    if i == len(prof_start_ind) - 1:
        end_ind = len(prof_start_ind)
    else:
        end_ind = prof_start_ind[i + 1]

    # Get the depth measurements of the profile
    depths = df.loc[prof_start_ind[i]:end_ind, 'Depth_m']

    # len(diffs) = len(depths)-1
    diffs = np.diff(depths)

    pass_check = np.where(abs(diffs) < max_grad)[0]
    fail_check = np.where(abs(diffs) >= max_grad)[0]

    df.loc[prof_start_ind[i] + 1 + fail_check, 'Gradient_check_flag'] = 1

    continue

# Export the updated dataframe
# When to apply these flags?
