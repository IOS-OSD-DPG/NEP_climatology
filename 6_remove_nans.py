import glob
from os.path import basename
import pandas as pd
from clim_helpers import concat_vvd_files

# Remove rows with variable == nan? (lots of IOS 1990's data)
nan_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'value_vs_depth\\6_filtered_for_nans\\sep_by_origin\\'

qc_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\5_filtered_for_quality_flag\\'

qc_files = glob.glob(qc_dir + '*.csv')
print(len(qc_files))

for f in qc_files:
    print(basename(f))
    df = pd.read_csv(f)
    print('Starting df length:', len(df))
    # Drop all rows that have df.Value == NaN
    df.dropna(axis='index', subset=['Depth_m', 'Value'], inplace=True)
    print('Ending df length:', len(df))
    # Export df
    outname = basename(f).replace('qc', 'nan_rm')
    df.to_csv(nan_dir + outname, index=False)


# Put all nan vvds into one file? (except WOD PFL)
all_name = 'Oxy_1991_2020_value_vs_depth_nan_rm.csv'
nan_files = glob.glob(nan_dir + '*.csv')
nan_files.remove(nan_dir + 'WOD_PFL_Oxy_1991_2020_value_vs_depth_nan_rm.csv')
fname = concat_vvd_files(nan_files, nan_dir, all_name)

# # WOD PFL Oxy data: did this at step 2!!
# indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#         'value_vs_depth\\2_added_dup_flags\\'
# dup_files = glob.glob(indir + 'WOD_PFL_Oxy*.csv')
# print(len(dup_files))
# fname = 'WOD_PFL_Oxy_1991_2020_value_vs_depth_dup.csv'
#
# concat_vvd_files(dup_files, indir, fname)
