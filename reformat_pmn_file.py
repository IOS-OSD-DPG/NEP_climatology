"""
Since the pmn netCDF file is really big with the 2d lat and lon, this script will
reduce lon and lat to 1d both. Float64 format for pm,pn,lon,lat is required to
maintain all decimal places
"""

from xarray import open_dataset, Dataset

pmn_dir = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\" \
          "16_diva_analysis\\pmn\\"

pmn_filename = pmn_dir + "divand_pmn_for_mask_6min_v2.nc"

pmn_ds = open_dataset(pmn_filename)

ncout = Dataset(coords={"lon": pmn_ds.lon2d.data[0, :], "lat": pmn_ds.lat2d.data[:, 0]},
                data_vars={"pm": (("lat", "lon"), pmn_ds.pm.data),
                           "pn": (("lat", "lon"), pmn_ds.pn.data)})

ncout.lon.attrs["longname"] = "Longitude"
ncout.lon.attrs["units"] = "degrees east"
ncout.lat.attrs["longname"] = "Latitude"
ncout.lat.attrs["units"] = "degrees north"
ncout.pm.attrs["longname"] = "Inverse of the local resolution in the x (longitude) " \
                             "dimension using the mean Earth radius"
ncout.pm.attrs["units"] = "meters"
ncout.pn.attrs["longname"] = "Inverse of the local resolution in the y (latitude) " \
                             "dimension using the mean Earth radius"
ncout.pn.attrs["units"] = "meters"

ncout_filename = pmn_dir + "divand_pmn_for_mask_6min_v3.nc"
ncout.to_netcdf(ncout_filename)
ncout.close()
