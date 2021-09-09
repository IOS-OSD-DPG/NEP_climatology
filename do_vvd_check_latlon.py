# Run the lat/lon check
# Make sure values within lat and lon study area

import glob
from vvd_check_latlon import vvd_subset_latlon

indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        'value_vs_depth\\11_stats_check\\'

infiles = glob.glob(indir + '*done.csv')

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\12_latlon_check\\'

for f in infiles:
    vvd_subset_latlon(f, outdir)
