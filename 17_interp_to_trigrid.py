"""Oct. 25, 2021
Take the regular grid field from DIVAnd and linearly-interpolate it to
Mike Foreman's unstructured triangle grid
"""
import pandas as pd
from scipy.interpolate import interp2d, griddata
from xarray import open_dataset
import os
import numpy as np
# import geopandas
# from shapely.geometry import Polygon, Point
from haversine import haversine
from tqdm import trange
from clim_helpers import deg2km

# Set constants
var = 'Oxy'
var_units = r'$\mu$' + 'mol/kg'  # Micromol per kilogram
year = 2010
szn = 'OND'
standard_depth = 0
radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)


mforeman_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\MForeman\\'
trigrid_filename = os.path.join(mforeman_dir + 'nep35_reord_latlon_wgeo.ngh')

# Read in triangle grid data
trigrid_data = np.genfromtxt(
    trigrid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
    names=['node', 'lon', 'lat', 'type', 'depth', 's1', 's2', 's3', 's4', 's5', 's6'],
    delimiter="", skip_header=3)

# Read in analysis from DIVAnd
interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\outputs\\'
field_filename = os.path.join(
    interp_dir + '{}_{}m_{}_{}_analysis2d_5deg_vanom.nc'.format(
        var, standard_depth, year, szn))

reg_field = open_dataset(field_filename)

# Create 2d interpolation function
# Convert longitude to positive
lon_diva = reg_field.Longitude.data + 360.
lat_diva = reg_field.Latitude.data
vout = reg_field.analysis.data + reg_field.pre_analysis_obs_mean.data
func = interp2d(x=lon_diva, y=lat_diva,
                z=vout, kind='linear', bounds_error=False, fill_value=np.NaN)

# -------Subset grid_data to reg_field size to avoid runtime error------------

# Called "clipping" in ArcGIS lingo

# # Create dictionary of trigrid data to make into geodataframe
# trigrid_data_dict = {
#     'node': list(trigrid_data['node']),
#     'geometry': list(map(lambda x, y: Point(x, y), trigrid_data['lon'], trigrid_data['lat']))}
#
# gdf_trigrid = geopandas.GeoDataFrame(trigrid_data_dict, crs=None)
#
# # Create polygon from DIVAnd analysis field
# # Input only the edge points as a list into Polygon to create the Polygon object
# poly_diva = Polygon(zip(lon_diva, lat_diva))
# gdf_poly = geopandas.GeoDataFrame([1], geometry=[poly_diva], crs=gdf_trigrid.crs)
#
# # Create geodataframe of Points from DIVA data
# lon_diva2d, lat_diva2d = np.meshgrid(lon_diva, lat_diva)
# diva_data_dict = {'value': vout.flatten(),
#                   'geometry': list(map(lambda x, y: Point(x, y), lon_diva2d.flatten(),
#                                        lat_diva2d.flatten()))}
# gdf_diva = geopandas.GeoDataFrame(diva_data_dict, crs=gdf_trigrid.crs)
#
# # Remove nan values from gdf_diva
# gdf_diva.dropna(inplace=True)
#
# # Create polygon(s) from gdf_data
# # Need to use clustering on the diva points??
#
# # The object on which you call clip is the object that will be clipped.
# # The object you pass is the clip extent.
# # So the triangle grid gets clipped by the DIVAnd field
# gdf_point_clipped = gdf_trigrid.clip(gdf_poly)


# Create mask from standard level data to subset grid_data

sl_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\14_sep_by_sl_and_year\\'
sl_filename = os.path.join(sl_dir + '{}_{}m_{}_{}.csv'.format(
    var, standard_depth, year, szn))

sl_data = pd.read_csv(sl_filename)

mask_trigrid = np.repeat(False, len(trigrid_data['lon']))
print('Creating mask for triangle grid...')

for i in trange(len(sl_data)):
    # Create tuple of coords
    pt_sl = (sl_data.loc[i, 'Longitude'], sl_data.loc[i, 'Latitude'])
    for j in range(len(trigrid_data['lon'])):
        # Check if mask already True, proceed if False
        if not mask_trigrid[j]:
            # .x accesses longitude part of Point object; .y accesses latitude part
            pt_trigrid = (trigrid_data['lon'][j] - 360., trigrid_data['lat'][j])
            dist = haversine(pt_sl, pt_trigrid)
            if dist <= radius_km:
                mask_trigrid[j] = True

trigrid_lon_subset = trigrid_data['lon'][mask_trigrid]
trigrid_lat_subset = trigrid_data['lat'][mask_trigrid]

# # Don't clip and just use rectangular subset of NEP instead -- TOO LARGE STILL
# lon_diva2d, lat_diva2d = np.meshgrid(lon_diva, lat_diva)
# subsetter_nan = ~np.isnan(vout)
# lon_min, lon_max = [np.min(lon_diva2d[subsetter_nan]), np.max(lon_diva2d[subsetter_nan])]
# lat_min, lat_max = [np.min(lat_diva2d[subsetter_nan]), np.max(lat_diva2d[subsetter_nan])]
#
# grid_lonlat_subsetter = np.where(
#     (grid_data['lon'] >= lon_min) & (grid_data['lon'] <= lon_max) &
#     (grid_data['lat'] >= lat_min) & (grid_data['lat'] <= lat_max))[0]
#
# grid_lon_subset = grid_data['lon'][grid_lonlat_subsetter]
# grid_lat_subset = grid_data['lat'][grid_lonlat_subsetter]


# Call the linear interpolating function on the clipped coordinates of the
# unstructured triangle grid
tri_field = func(trigrid_lon_subset, trigrid_lat_subset)


# Use griddata?
grid_linear = griddata(points=(lon_diva, lat_diva), values=vout,
                       xi=(trigrid_lon_subset, trigrid_lat_subset), method='linear')
