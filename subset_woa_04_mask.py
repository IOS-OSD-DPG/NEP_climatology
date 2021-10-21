import pandas as pd
import numpy as np
from os.path import join

mask_filename = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\WOA_clim_code\\' \
                'landsea_04.msk'

mask_data = pd.read_csv(mask_filename, skiprows=1)

lon_min = -160
lon_max = -115
lat_min = 30
lat_max = 60

# Subset the Northeast Pacific ocean
subsetter = np.where(
    (mask_data.Longitude >= lon_min) & (mask_data.Longitude <= lon_max) &
    (mask_data.Latitude >= lat_min) & (mask_data.Latitude <= lat_max))[0]

print(subsetter.shape)
print(mask_data.shape)

# Also subset out other ocean bits that we don't need?

mask_out = mask_data.loc[subsetter]

print(min(mask_out.Bottom_Standard_level), max(mask_out.Bottom_Standard_level))
print(min(mask_data.Bottom_Standard_level), max(mask_data.Bottom_Standard_level))

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
mask_out_filename = join(outdir + 'landsea_04_nep.msk')

mask_out.to_csv(mask_out_filename, index=False)
