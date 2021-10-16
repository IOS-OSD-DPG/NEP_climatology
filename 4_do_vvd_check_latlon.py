# Run the lat/lon check
# Make sure values within lat and lon study area

from vvd_check_latlon import vvd_subset_latlon
import glob

# indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#         'value_vs_depth\\11_stats_check\\'
indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        'value_vs_depth\\3_filtered_for_duplicates\\'

infiles = glob.glob(indir + '*.csv')

# Remove PFL file
# infiles.remove(indir + 'WOD_PFL_Oxy_1991_2020_value_vs_depth_dup_rm.csv')

# fname = 'ALL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

# Check WOD argo data
# fname = 'WOD_PFL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'
# fpath = indir + fname

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\4_latlon_check\\'

# Saves updated df to a new file
for f in infiles:
    vvd_subset_latlon(f, outdir)
