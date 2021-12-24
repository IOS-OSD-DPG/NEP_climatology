# Run the lat/lon check
# Make sure values within lat and lon study area

# The limits are hard-coded into the vvd_subset_latlon function
#     lat_min = 30.
#     lat_max = 60.
#     lon_min = -160.
#     lon_max = -115.

from vvd_check_latlon import vvd_subset_latlon
import glob
from os.path import basename

variable_name = 'Temp'  # Sal Oxy

# indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#         'value_vs_depth\\11_stats_check\\'
indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        'value_vs_depth\\3_filtered_for_duplicates\\'

# Remove PFL file
# infiles.remove(indir + 'WOD_PFL_Oxy_1991_2020_value_vs_depth_dup_rm.csv')

# fname = 'ALL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

# Check WOD argo data
# fname = 'WOD_PFL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'
# fpath = indir + fname

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\4_latlon_check\\'

for var in ['Temp', 'Sal']:
    infiles = glob.glob(indir + '*{}*value_vs_depth*.csv'.format(var))
    # Saves updated df to a new file
    for f in infiles:
        print(basename(f))
        vvd_subset_latlon(f, outdir)
