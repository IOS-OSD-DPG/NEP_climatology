import pandas as pd
import numpy as np
from os.path import join
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os

# Read in WOA 0.25-degree standard level bathymetry
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

# -----------------------plot the result-----------------------------------------

mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
mask_filename = os.path.join(mask_dir + 'landsea_04_nep.msk')

mask_nep = pd.read_csv(mask_filename)

# Create the plot
left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]

# m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
#             urcrnrlon=right_lon, urcrnrlat=top_lat,
#             projection='lcc',  # width=40000, height=40000, #lambert conformal project
#             resolution='h', lat_0=0.5 * (bot_lat + top_lat),
#             lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
# m.drawcoastlines(linewidth=0.2)
# m.drawmapboundary(fill_color='white')
# m.fillcontinents(color='0.8')

lat = np.array(mask_nep.Latitude)
lon = np.array(mask_nep.Longitude)
bot_sl = np.array(mask_nep.Bottom_Standard_level)

unique_lat = np.unique(lat)
print(len(unique_lat))

unique_lon = np.unique(lon)
print(len(unique_lon))

lat2d = lat.reshape((len(unique_lat), len(unique_lon)))
print(lat2d)
lon2d = lon.reshape((len(unique_lat), len(unique_lon)))
print(lon2d)
bot_sl2d = bot_sl.reshape((len(unique_lat), len(unique_lon)))
print(bot_sl2d)

plt.pcolor(lon2d[0], lat2d[:, 0], bot_sl2d, shading='auto', cmap='jet')
plt.colorbar()

png_name = os.path.join(mask_dir + 'landsea_04_nep_msk.png')
plt.savefig(png_name)
plt.close(fig)
