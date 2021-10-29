import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import xarray as xr


standard_depth = 0

gebco_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\' \
            'GEBCO_28_Oct_2021_16f8a0236741\\'

gebco_filename = os.path.join(gebco_dir + 'gebco_2021_n60_s30_w-160_e-115.nc')

gebco_data = xr.open_dataset(gebco_filename)

# print(np.diff(gebco_bath.lat.data))

# Create 2d grid of lat and lon
Lon, Lat = np.meshgrid(gebco_data.lon.data, gebco_data.lat.data)
print(Lon.shape)
print(Lon)
print(Lat)

# -1 to convert elevation above sea level to depth below sea level
mask = -gebco_data.elevation.data >= standard_depth

# Create bathymetry to plot
bath = -gebco_data.elevation.data.astype(float)
bath[~mask] = np.nan

# Plot mask?

mask_v2 = np.zeros(mask.shape, dtype='int')
mask_v2[mask] = 1

print(len(mask_v2[mask])/len(mask_v2.flatten()))

# Create the plot
left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]

m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',  # width=40000, height=40000, #lambert conformal project
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
m.drawcoastlines(linewidth=0.2)
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='0.8')

plt.pcolormesh(gebco_data.lon.data, gebco_data.lat.data, bath,
               shading='auto', cmap='jet')
# plt.colorbar(label='Depth [m]')

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
png_name = os.path.join(output_dir + 'gebco_2021_6sec_msk.png')
plt.savefig(png_name)
plt.close(fig)
