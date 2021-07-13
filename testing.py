import xarray as xr

inFile = '/home/hourstonh/Downloads/IOS_CTD_Profiles_dcfd_46ec_bcd4.nc'

ctdProf = xr.open_dataset(inFile)

# Close the dataset
ctdProf.close()
