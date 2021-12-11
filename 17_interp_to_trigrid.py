"""Oct. 25, 2021
Take the regular grid field from DIVAnd and linearly-interpolate it to
Mike Foreman's unstructured triangle grid
"""
import pandas as pd
from scipy.interpolate import griddata, interpn
from xarray import open_dataset, Dataset
import os
import numpy as np
# import geopandas
# from shapely.geometry import Polygon, Point
# from haversine import haversine
# from tqdm import trange
from clim_helpers import deg2km
# from scipy.spatial import Delaunay
# from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.tri as mtri
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# from dask import delayed


def linterp_to_TG(rg_dir, output_dir, tg_file, mask_file, var_name, depth, yr, szn):
    # Linear interpolation from 6 minute regular grid to unstructured triangle grid
    rg_file = os.path.join(rg_dir + '{}_{}m_{}_{}_analysis2d.nc')
    rg_ds = open_dataset(rg_file)
    
    # Open grid file containing coordinates of triangle knots
    tg_data = np.genfromtxt(
        tg_file, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
        names=['node', 'lon', 'lat', 'type', 'depth', 's1', 's2', 's3', 's4', 
               's5', 's6'],
        delimiter="", skip_header=3)
    
    # Create 2d matrix holding trigrid points
    tg_points = np.array(
        [tg_data['lon'].tolist(), tg_data['lat'].tolist()]).transpose()

    # Use nc masks to mask out land in tg_points?????????
    # But the mask points are regular grid not trigrid...
    mask_ds = open_dataset(mask_file)

    # Prepare the regular grid data from DIVAnd for linear interpolation
    x_rg = rg_ds.longitude.data
    y_rg = rg_ds.latitude.data
    var_data = rg_ds.vout.data
    
    # xi: The coordinates to sample the gridded data at
    tg_values = interpn(points=(x_rg, y_rg), values=var_data, xi=tg_points[0],
                        method='linear', bounds_error=False, fill_value=np.nan)
    
    # Export the values on the trigrid
    ncout = Dataset(
        coords={'longitude': tg_points[:, 0], 'latitude': tg_points[:, 1}, 
        data_vars={'data_values': tg_values})
    
    ncout_filename = os.path.join(output_dir + '{}_{}m_{}_{}_tg.nc'.format(
        var_name, depth, yr, szn))
    ncout.to_netcdf(ncout_filename)
    ncout.close()
    
    return ncout_filename


# ----------------------------Set constants---------------------------------------------
var_name = 'Oxy'
var_units = r'$\mu$' + 'mol/kg'  # Micromol per kilogram
year = 1991
szn = 'JFM'
standard_depth = 0
radius_deg = 2  # search radius
radius_km = deg2km(radius_deg)

# -----------------------Import data----------------------------------------------------

mforeman_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\MForeman\\'
grid_filename = os.path.join(mforeman_dir + 'nep35_reord_latlon_wgeo.ngh')

# Read in triangle grid data
grid_data = np.genfromtxt(
    grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
    names=['node', 'lon', 'lat', 'type', 'depth', 's1', 's2', 's3', 's4', 's5', 's6'],
    delimiter="", skip_header=3)

# Read in analysis from DIVAnd
interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
             '16_diva_analysis\\analysis\\fithorzlen\\'
field_filename = os.path.join(
    interp_dir + '{}_{}m_{}_{}_analysis2d.nc'.format(
        var_name, standard_depth, year, szn))

diva_data = open_dataset(field_filename)

# Convert longitude to positive
lon_diva = diva_data.longitude.data + 360.
lat_diva = diva_data.latitude.data
field_diva = diva_data.vout.data  # + diva_data.pre_analysis_obs_mean.data

print(field_diva.shape)

# # Create 2d interpolation function
# func = interp2d(x=lon_diva, y=lat_diva,
#                 z=vout, kind='linear', bounds_error=False, fill_value=np.NaN)

# -----------Subset grid_data to reg_field size to avoid runtime error---------------------

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


# --------------Create mask from standard level data to subset grid_data------------------

# sl_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#          'value_vs_depth\\14_sep_by_sl_and_year\\'
# sl_filename = os.path.join(sl_dir + '{}_{}m_{}_{}.csv'.format(
#     var_name, standard_depth, year, szn))
#
# sl_data = pd.read_csv(sl_filename)
#
# mask_trigrid = np.repeat(False, len(trigrid_data['lon']))
# print('Creating mask for triangle grid...')
#
# for i in trange(len(sl_data)):
#     # Create tuple of coords
#     pt_sl = (sl_data.loc[i, 'Longitude'], sl_data.loc[i, 'Latitude'])
#     for j in range(len(trigrid_data['lon'])):
#         # Check if mask already True, proceed if False
#         if not mask_trigrid[j]:
#             # .x accesses longitude part of Point object; .y accesses latitude part
#             pt_trigrid = (trigrid_data['lon'][j] - 360., trigrid_data['lat'][j])
#             dist = haversine(pt_sl, pt_trigrid)
#             if dist <= radius_km:
#                 mask_trigrid[j] = True
#
# trigrid_lon_subset = trigrid_data['lon'][mask_trigrid]
# trigrid_lat_subset = trigrid_data['lat'][mask_trigrid]
# trigrid_node_subset = trigrid_data['node'][mask_trigrid]
#
# print(len(trigrid_lon_subset), len(trigrid_lat_subset))

# --------------Don't clip and just use rectangular subset of NEP instead -- TOO LARGE STILL
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

# # Call the linear interpolating function on the clipped coordinates of the
# # unstructured triangle grid
# tri_field = func(trigrid_lon_subset, trigrid_lat_subset)

# ---------------Apply scipy griddata()--------------------------------------------------

lon_diva2d, lat_diva2d = np.meshgrid(lon_diva, lat_diva)
print(lon_diva2d, lat_diva2d, sep='\n')

# field_diva_flat = field_diva.flatten()
# lon_diva2d_flat = lon_diva2d.flatten()[~np.isnan(field_diva_flat)]
# lat_diva2d_flat = lat_diva2d.flatten()[~np.isnan(field_diva_flat)]
# field_diva_flat = field_diva_flat[~np.isnan(field_diva_flat)]
#
# var_tri_interp = griddata(points=(lon_diva2d_flat, lat_diva2d_flat), values=field_diva_flat,
#                           xi=(trigrid_lon_subset, trigrid_lat_subset), method='linear',
#                           fill_value=np.nan)
#
# print(var_tri_interp)
# print(var_tri_interp.shape)

# # Clean up nan values?
# result_tri_interp_qc = var_tri_interp[~np.isnan(var_tri_interp)]
# trigrid_lon_subset_qc = trigrid_lon_subset[~np.isnan(var_tri_interp)]
# trigrid_lat_subset_qc = trigrid_lat_subset[~np.isnan(var_tri_interp)]
# trigrid_node_subset_qc = trigrid_node_subset[~np.isnan(var_tri_interp)]
#
# print(result_tri_interp_qc)

# Do all points to simplify plotting???
field_scipy = griddata(points=(lon_diva2d.flatten(), lat_diva2d.flatten()),
                       values=field_diva.flatten(), xi=(grid_data['lon'], grid_data['lat']),
                       method='linear', fill_value=np.nan)

print(field_scipy.shape)
print(len(field_scipy[~np.isnan(field_scipy)]), len(field_scipy[~np.isnan(field_scipy)]) / len(field_scipy))

# ------------------Plot the linearly-interpolated data-----------------------------------
# Copied from Lu Guan T_climatology.py

# Read in the triangles that are listed by their 3 nodes
tri_filename = os.path.join(mforeman_dir + 'nep35_reord.tri')

tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3))-1

# # Initialize boolean mask
# tri_mask = np.repeat(True, len(tri_data))
#
# # Subset tri_data to fit the subsetted trigrid lon and lat
# for i in trange(len(tri_data)):
#     # Check whether each node in each row of tri_data is in trigrid lon and lat
#     # If not, flag the offending row to remove it later
#     node1, node2, node3 = [tri_data[i, 0], tri_data[i, 1], tri_data[i, 2]]
#     if (node1 not in trigrid_node_subset_qc or node2 not in trigrid_node_subset_qc
#             or node3 not in trigrid_node_subset_qc):
#         tri_mask[i] = False
#
# tri_data_subset = tri_data[tri_mask]
# print(len(tri_data_subset))
# print(len(tri_data), len(tri_data_subset)/len(tri_data))

# Create the plot
left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]

m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',  # width=40000, height=40000, #lambert conformal project
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

# lcc: Lambert Conformal Projection;
# cyl: Equidistant Cylindrical Projection
# merc: Mercator Projection

# xpt, ypt = m(trigrid_lon_subset_qc, trigrid_lat_subset_qc)
# tri_pt = mtri.Triangulation(xpt, ypt, tri_data_subset)

# All
xpt, ypt = m(grid_data['lon'], grid_data['lat'])
# tri_pt = mtri.Triangulation(xpt, ypt, tri_data)

# tri = mtri.Triangulation(trigrid_lon_subset_qc, trigrid_lat_subset_qc, tri_data_subset)  #, mask=tri_mask

# Try with un-subsetted
tri = mtri.Triangulation(grid_data['lon'], grid_data['lat'], tri_data)

triangles = tri.triangles

fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
m.drawcoastlines(linewidth=0.2)
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='0.8')

# set the nan to white on the map
color_map = plt.cm.get_cmap()
color_map.set_bad('w')

# VAR_TRI_INTERP
cax = plt.tripcolor(xpt, ypt, triangles, field_scipy, cmap='jet', edgecolors='none',
                    vmin=150, vmax=400)
#                   cmap='YlOrBr', vmin=np.nanmin(var_all), vmax=np.nanmax(var_all))

cbar = fig.colorbar(cax, shrink=0.7)  # set scale bar
cbar.set_label('{} [{}]'.format(var_name, var_units), size=14)  # scale label
# labels = [left,right,top,bottom]
parallels = np.arange(bot_lat, top_lat, 4.)
# parallels = np.arange(48., 54, 0.2); parallels = np.linspace(bot_lat, top_lat, 10)
m.drawparallels(parallels, labels=[True, False, False, False])  #draw parallel lat lines
meridians = np.arange(left_lon, -100.0, 15.)
# meridians = np.linspace(int(left_lon), right_lon, 5)
m.drawmeridians(meridians, labels=[False, False, True, True])
# plt.show()

lin_interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                 'value_vs_depth\\17_lin_interp_to_trigrid\\'
png_name = os.path.join(lin_interp_dir + '{}_{}m_{}_{}_tri_hasnan.png'.format(
    var_name, standard_depth, year, szn))
plt.savefig(png_name)

plt.close(fig)

# ------------------Save the linearly-interpolated data to csv??--------------------------

lin_interp_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                 'value_vs_depth\\17_lin_interp_to_trigrid\\'
csv_outname = os.path.join(lin_interp_dir + '{}_{}m_{}_{}_tri_hasnan.csv'.format(
    var_name, standard_depth, year, szn))

# df_lin = pd.DataFrame(
#     data=np.array([trigrid_node_subset_qc, trigrid_lon_subset_qc, trigrid_lat_subset_qc,
#                    result_tri_interp_qc]).transpose(),
#     columns=['Node', 'Longitude', 'Latitude', 'Value'])

df_lin = pd.DataFrame(
    data=np.array([grid_data['node'], grid_data['lon'], grid_data['lat'],
                   field_scipy]).transpose(),
    columns=['Node', 'Longitude', 'Latitude', 'Value'])

print(df_lin)

df_lin.Node = df_lin.Node.astype(int)

# Should I remove all NaN values?
print(len(df_lin.dropna()))
# See what percentage of values are not NaNs
print(len(df_lin.dropna()) / len(df_lin))

print(df_lin.dropna())

# df_lin.dropna(inplace=True)

print(df_lin)

df_lin.to_csv(csv_outname, index=False)
