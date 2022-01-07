import os
import pandas as pd
import numpy as np
import glob
from clim_helpers import date_string_to_datetime

# Check value vs depth values for Temperature

variable_name = 'Temp'

# vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\data\\value_vs_depth\\' \
#           '6_filtered_for_nans\\sep_by_origin\\'
# vvd_files = glob.glob(vvd_dir + '*{}*value_vs_depth_nan_rm.csv'.format(variable_name))

vvd_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\data\\value_vs_depth\\' \
          '9_gradient_check\\'
vvd_files = glob.glob(vvd_dir + '*{}*done.csv'.format(variable_name))
print(len(vvd_files))

# Go through PFL data
for f in vvd_files[-4:]:
    print(os.path.basename(f))
    df_vvd = pd.read_csv(f)
    values = np.array(df_vvd.Value[df_vvd.Depth_m < 5])
    lats = np.array(df_vvd.Latitude[df_vvd.Depth_m < 5])
    # lons = np.array(df_vvd.Longitude[df_vvd.Depth_m < 5])
    print('Number of observations above 5m depth:', len(values))
    print('Min of latitude of obs:', np.nanmin(lats))
    print('Max of latitude of obs:', np.nanmax(lats))
    print('Mean of latitude of obs:', np.nanmean(lats))
    print('Median of latitude of obs:', np.nanmedian(lats))
    # print('Min of longitude of obs:', np.nanmin(lons))
    # print('Max of longitude of obs:', np.nanmax(lons))
    print('Min of observations:', np.nanmin(values))
    print('Max of observations:', np.nanmax(values))
    print('Mean of observations:', np.nanmean(values))
    print('Median of observations:', np.nanmedian(values))
    print()

# Check WOD CTD data by season
wod_ctd_file = vvd_files[-7]
df_wod_ctd = pd.read_csv(wod_ctd_file)
df_wc = date_string_to_datetime(df_wod_ctd)

for szn, name in zip(range(4), ['JFM', 'AMJ', 'JAS', 'OND']):
    szn_months = (3 * szn + 1, 3 * szn + 3)  # range
    print(name, szn_months)
    # Surface
    szn_subsetter = np.where((df_wc.Time_pd.dt.month >= szn_months[0]) &
                             (df_wc.Time_pd.dt.month <= szn_months[1]) &
                             (df_wc.Depth_m < 5))[0]
    values = np.array(df_wc.Value[szn_subsetter])
    print('Number of observations above 5m depth:', len(values))
    print('Min of observations:', np.nanmin(values))
    print('Max of observations:', np.nanmax(values))
    print('Mean of observations:', np.nanmean(values))
    print()

ios_ctd_files = vvd_files[:6]

df_ios_ctd = pd.DataFrame()
for f in ios_ctd_files:
    df_in = pd.read_csv(f)
    df_ios_ctd = pd.concat([df_ios_ctd, df_in])

df_ios_ctd.reset_index(inplace=True)
df_ic = date_string_to_datetime(df_ios_ctd)

for szn, name in zip(range(4), ['JFM', 'AMJ', 'JAS', 'OND']):
    szn_months = (3 * szn + 1, 3 * szn + 3)  # range
    print(name, szn_months)
    # Surface
    szn_subsetter = np.where((df_ic.Time_pd.dt.month >= szn_months[0]) &
                             (df_ic.Time_pd.dt.month <= szn_months[1]) &
                             (df_ic.Depth_m < 5))[0]
    values = np.array(df_ic.loc[szn_subsetter, 'Value'])
    lats = np.array(df_ic.loc[szn_subsetter, 'Latitude'])
    # lons = np.array(df_ic.loc[szn_subsetter, 'Longitude'])
    print('Number of observations above 5m depth:', len(values))
    print('Min of latitude of obs:', np.nanmin(lats))
    print('Max of latitude of obs:', np.nanmax(lats))
    print('Mean of latitude of obs:', np.nanmean(lats))
    print('Median of latitude of obs:', np.nanmedian(lats))
    # print('Min of longitude of obs:', np.nanmin(lons))
    # print('Max of longitude of obs:', np.nanmax(lons))
    print('Min of observations:', np.nanmin(values))
    print('Max of observations:', np.nanmax(values))
    print('Mean of observations:', np.nanmean(values))
    print('Median of observations:', np.nanmedian(values))
    print()
