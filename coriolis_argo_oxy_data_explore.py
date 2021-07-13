# Explore Coriolis Argo oxygen data
# Plot temporal and spatial distributions of this data

import xarray as xr
import glob

argo_dir = '/home/hourstonh/Documents/climatology/oxy_clim/Coriolis_Argo/'
argo_files = glob.glob(argo_dir + '*.nc')

argo = argo_files[0]

argodat = xr.open_dataset(argo)

# Some key variable names:
# LATITUDE, LONGITUDE, DOXY_ADJUSTED, N_PROF (number of profiles), JULD (Julian day in UTC)
# JULD same as JULD_LOCATION ??

# JULD in numpy.datetime64 format

# Plot spatial distribution

argodat.close()
