from DIVAnd_cv import DIVAnd_cv
import pandas as pd
import numpy as np
import os
from clim_helpers import deg2km, get_standard_levels
import DIVAnd


# Test out the Julia cross-validation function for estimating correlation length
# and signal-to-noise ratio

# -----------------------------Choose data file----------------------------------
var_name = 'Oxy'
year = 2010
szn = 'OND'
standard_depth = 0
radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)  # degrees length
# -------------------------------------------------------------------------------

# Get standard levels file
sl_filename = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
              'WOA_Standard_Depths.txt'

sl_arr = get_standard_levels(sl_filename)

# Read in standard level data file
sl_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
         '14_sep_by_sl_and_year\\'
sl_filename = os.path.join(sl_dir + '{}_{}m_{}_{}.csv'.format(
    var_name, standard_depth, year, szn))

sldata = pd.read_csv(sl_filename)

# Convert sldata's contents into numpy arrays
xobs = np.array(sldata.Longitude)
yobs = np.array(sldata.Latitude)
vobs = np.array(sldata.SL_value)

# Read in mask
mask_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\'
mask_filename = os.path.join(mask_dir + 'landsea_04_nep.msk')

mask_df = pd.read_csv(mask_filename)
# print(mask_df.columns)
print('Mask range:', min(mask_df.Bottom_Standard_level), max(mask_df.Bottom_Standard_level))

# Create 2d grid of lat and lon points by reshaping the mask_df columns
# Find dims to reshape columns to
unique, counts = np.unique(mask_df.Longitude, return_counts=True)
print('counts length:', len(counts))
# print(counts)

Lon = np.array(mask_df.Longitude).reshape((counts[0], len(counts)))
Lat = np.array(mask_df.Latitude).reshape((counts[0], len(counts)))
# DO NOt Have to reverse order of Lat so that Latitude is decreasing each row

# Create boolean mask based on standard level of the input obs
sl_index = np.where(sl_arr == standard_depth)[0][0]
# Indexing of sl_arr starts @ 0, while standard level counting starts at 2!
# Bottom_Standard_level starts at 1, which is land, so ocean starts at 2
mask = mask_df.Bottom_Standard_level >= (sl_index + 2)
# Reshape mask to be same shape as Lat and Lon
mask = np.array(mask).reshape((counts[0], len(counts)))

# Further limit mask according to sampling locations

mask_v2 = np.zeros(shape=mask.shape)
mask_v2[mask] = 1

print(len(mask_v2[mask_v2 == 1]), len(mask_v2[mask_v2 == 0]))

# --------------------Calculate input parameters and run analysis-----------------------

# Scale factor of the grid
pm, pn = DIVAnd.metric(Lon, Lat)
# print(pm, pn, sep='\n')

# # For first guess correlation length, can use a value between 1/10 domain size and
# # the domain size
# # Also can try optimization on correlation length
# domain_size_deg = -115-(-160)
# deg2m = 111e3  # This is an approximation
# domain_size_m = domain_size_deg * deg2m
# print(domain_size_m/10)

# Decreasing the correlation length decreases the "smoothness"
lenx = 500e3  # 800e3  # in meters
leny = 500e3  # 800e3  # in meters

# error variance of the observations (normalized by the error variance of
# the background field)
# If epsilon2 is a scalar, it is thus the inverse of the signal-to-noise ratio
signal_to_noise_ratio = 50.  # Default from Lu ODV session
epsilon2 = 1/signal_to_noise_ratio  # 1.

# Compute anomalies (i.e., subtract mean)
vmean = np.mean(vobs)
vanom = vobs - vmean

# Choose number of testing points around the current value of L (corlen)
nl = 1

# Choose number of testing points around the current value of epsilon2
ne = 1

# Choose cross-validation method
# 1: full CV; 2: sampled CV; 3: GCV; 0: automatic choice between the three
method = 3

# ------------------------Run cross-validation---------------------------------

bestfactorl, bestfactore, cvval, cvvalues, x2Ddata, y2Ddata, cvinter, xi2D, yi2D = DIVAnd_cv(
    mask, (pm, pn), (Lon, Lat), (xobs, yobs), vanom, (lenx, leny), epsilon2, nl, ne, method)

print('Bestfactorl:', bestfactorl)
print('bestfactore:', bestfactore)
print('cvval:', cvval)
print('cvvalues:', cvvalues)
print('x2Ddata:', x2Ddata)
print('y2dData:', y2Ddata)
print('cvinter:', cvinter)
print('xi2D:', xi2D)
print('yi2D:', yi2D)

print('New corlen:', lenx * bestfactorl)
print('New epsilon2:', epsilon2 * bestfactore)
