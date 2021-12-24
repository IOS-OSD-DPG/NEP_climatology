from scipy.ndimage import gaussian_filter  # ,median_filter
from scipy.signal import medfilt2d
from xarray import open_dataset
import os
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap

# Options for performing smoothing of regular gridded data

variable_name = 'Oxy'
standard_depth = 0
season_int = 1
# ---------------------Option 1: Do median filter of data-------------------------
# Need to project (linearly interpolate) to 6 minute regular grid first?

# nc file contains 2d regular gridded climatology dataset
data_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\ODV_outputs\\' \
              'mean_from_odv\\0m\\'
data_filename = os.path.join(data_folder, 'Oxy_0m_JFM_TG_mean_est.txt')
data_df = pd.read_csv(data_filename, sep='\t')

lon = np.array(data_df['Longitude'])
lat = np.array(data_df['Latitude'])
sl_data = np.array(data_df['Sl_value_30yr_avg @ Season={}.00'.format(season_int)])

xi = np.linspace(200, 245, 13500)  # ~ 333m ~ 0.003 degree
yi = np.linspace(30, 60, 9000)  # ~ 333m ~ 0.003 degree
x_lon_r, y_lat_r = np.meshgrid(xi, yi)  # create regular grid

# create basemap
left_lon, right_lon, bot_lat, top_lat = [200, 245, 25, 62]
m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',  # width=40000, height=40000, #lambert conformal project
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

xpr, ypr = m(x_lon_r, y_lat_r)  # convert lat/lon to x/y map projection coordinates in meters using basemap

# Apply median filter
data2d = None
medfilt_data = medfilt2d(data2d)


# ---------------------Option 2: Do Gaussian filter of data-----------------------

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter

# Sigma is a required parameter to choose
gaussfilt_data = gaussian_filter(data2d, sigma=1)


# -----------------------Gaussian filter, alternate A-------------------

# Copied directly from
# https://gis.stackexchange.com/questions/9431/what-raster-smoothing-generalization-tools-are-available

import numpy as np
from scipy.signal import fftconvolve


def gaussian_blur(in_array, size):
    # expand in_array to fit edge of kernel
    padded_array = np.pad(in_array, size, 'symmetric')
    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)
    # do the Gaussian blur
    return fftconvolve(padded_array, g, mode='valid')


# ----------------------Gaussian filter, alternate B--------------------

# https://gist.github.com/esisa/5849392 Gist

