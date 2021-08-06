# Check IOS WP CTD data for oxygen data
# If no oxygen data available, move to a subfolder?

import glob
from xarray import open_dataset
import shutil
from os.path import basename

ctd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
          'source_format\\SHuntington\\WP_unique_CTD_forHana\\'

ctd_files = glob.glob(ctd_dir + '*.ctd.nc', recursive=False)

counter_no_DOXM = 0
counter_no_DOXY = 0

no_oxy_dir = ctd_dir + 'no_oxygen\\'

for f in ctd_files:
    # print(f)
    data = open_dataset(f)

    flag = 0
    try:
        oxygen = data.DOXMZZ01.data
    except AttributeError:
        counter_no_DOXM += 1
        flag += 1

    try:
        oxygen = data.DOXYZZ01.data
    except AttributeError:
        counter_no_DOXY += 1
        flag += 1

    # Close the file so we can move it if needed
    data.close()

    if flag == 2:
        # Move the file to a subdir
        shutil.move(f, no_oxy_dir + basename(f))

print(counter_no_DOXM)  # 170 files
print(counter_no_DOXY)  # 156 files




