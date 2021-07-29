""" July 29, 2021
Accounting statistics for the ncdata.Oxygen_Method data in NODC WOD OSD files.

Results from np.unique(ncdata.Oxygen_Method.data, return_counts=True):

Oxy_1991_2020_AMJ_OSD.nc
[b''
 b'Winkler automated oxygen titration: amperometric end-detection (Culberson 1991)'
 b'Winkler automated oxygen titration; whole bottle method (Carpenter 1965)']
[ 909    2 1498]

Oxy_1991_2020_JAS_OSD.nc
[b''
 b'Winkler automated oxygen titration; whole bottle method (Carpenter 1965)'
 b'Winkler method (unknown)']
[ 289 1589   15]
Oxy_1991_2020_JFM_OSD.nc
[b''
 b'Winkler automated oxygen titration; whole bottle method (Carpenter 1965)']
[ 639 1862]
Oxy_1991_2020_OND_OSD.nc
[b''
 b'Winkler automated oxygen titration: amperometric end-detection (Culberson 1991)'
 b'Winkler automated oxygen titration: whole-bottle method; amperometric end-detection (Culberson and Huang 1987)'
 b'Winkler automated oxygen titration; whole bottle method (Carpenter 1965)'
 b'Winkler method (unknown)']
[ 593    4   76 1681    1]

"""

import glob
from xarray import open_dataset
import numpy as np
from os.path import basename

osd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'source_format\\WOD_extracts\\Oxy_WOD_May2021_extracts\\'

osd_files = glob.glob(osd_dir + 'Oxy_*_OSD.nc')
osd_files.sort()

for f in osd_files:
    print(basename(f))
    data = open_dataset(f)
    unique, counts = np.unique(data.Oxygen_Method.data, return_counts=True)
    print(unique)
    print(counts)

# Count the number of blank entries ==2430
blanks = 909 + 289 + 639 + 593

# Number of non-blank entries around 6500
