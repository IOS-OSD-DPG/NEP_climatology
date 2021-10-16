"""
Print out the profiles from 8_gradient_check that got dropped
during vertical interpolation
"""
import pandas as pd
import numpy as np

interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\9_vertical_interp\\'

series_drop = pd.read_csv(interp_dir + 'Oxy_1991_2020_rr_prof_drop.csv')

grad_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\8_gradient_check\\' \
            'ALL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

df_grad = pd.read_csv(grad_file)

list_len_gt_1 = []  # Profile length greater than 1

# Iterate through the df of profile numbers of profiles dropped during
# vertical interpolation
for i in range(len(series_drop)):  # 20
    # Find the profile in the observed level data
    prof_ind = np.where(
        df_grad.Profile_number == series_drop.loc[i, 'Profile_number'])[0]
    prof = df_grad.loc[prof_ind, ['Depth_m', 'Value']]
    # print(series_drop.iloc[i])
    # print(prof)
    if len(prof) > 1:
        list_len_gt_1.append(series_drop.loc[i, 'Profile_number'])

print(len(series_drop))  # 185 profiles v1; 177 profiles v2
print(len(list_len_gt_1))
# Number of dropped profiles with length > 1 == 11 profiles v1; 3 profiles v2

for num in list_len_gt_1:
    # Find the profile in the observed level data
    prof_ind = np.where(
        df_grad.Profile_number == num)[0]
    prof = df_grad.loc[prof_ind,
                       ['Cruise_number', 'Instrument_type', 'Depth_m', 'Value']]
    print('Profile number:', num)
    print(prof)
    print()

# Some of these seem to be upcasts, which would be why my interpolation
# code wouldn't use them
# Other profiles are composed of only 2 measurements, so maybe add linear
# interpolation (np.interp or scipy.interpolate). Would need to take the
# acceptable interpolation point distances from WOA13/18 into account,
# which would still leave profiles with one obs at 4.9 m and the other at
# > 300 m

# numpy.interp1d(): One-dimensional linear interpolation for monotonically
# increasing sample points.

# scipy.interpolate.interp1d(): Interpolate a 1-D function.
# Returns an interpolating function
# (doesn't say the sample points have to be monotonically increasing)
