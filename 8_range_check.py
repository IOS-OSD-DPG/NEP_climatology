import pandas as pd
import numpy as np
from tqdm import trange
from clim_helpers import vvd_apply_value_flag


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


# ---------------------------STEP 2: Range check----------------------------

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
         'value_vs_depth\\7_depth_check\\'
df_file = 'Oxy_1991_2020_value_vs_depth_dep_check_done.csv'
# df_file = 'WOD_PFL_Oxy_1991_2020_value_vs_depth_dep_check_done.csv'

df_in = pd.read_csv(df_dir + df_file)

# Range check function
df_out = vvd_range_check(df_in, df_range)

# Checks
print(len(df_out.loc[df_out.Range_check_flag == 1, 'Range_check_flag']))
indices_rng = np.where(df_out.Range_check_flag == 1)[0]
print(df_out.loc[indices_rng, ['Depth_m', 'Value']])
# 24174 rows that were flagged during this check for Oxy: should limits be expanded??

# df_out.drop(columns='Depth_check_flag', inplace=True)

df_outname = df_file.replace('dep_check_done', 'rng_check')
df_outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
            'value_vs_depth\\8_range_check\\'
df_out.to_csv(df_outdir + df_outname, index=False)

df_rng_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
              'value_vs_depth\\8_range_check\\' \
              'Oxy_1991_2020_value_vs_depth_rng_check.csv'
# df_rng_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#               'value_vs_depth\\7_range_check\\' \
#               'WOD_PFL_Oxy_1991_2020_value_vs_depth_rng_check.csv'

df_in_fm = pd.read_csv(df_rng_name)

df_out_fm = vvd_apply_value_flag(df_in_fm, 'Range_check_flag')

df_out_name_fm = df_rng_name.replace('rng_check', 'rng_check_done')
df_out_fm.to_csv(df_out_name_fm, index=False)