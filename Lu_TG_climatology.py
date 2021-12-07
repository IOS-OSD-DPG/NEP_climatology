# globals().clear()
# clear all
# os.system("clear")

# -------------------  import packages ------------------------------------------------------------

# import sys
import os
import numpy as np
import pandas as pd
# from scipy.spatial import Delaunay
# from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from itertools import islice
from mpl_toolkits.basemap import Basemap
import fiona
import rasterio.mask
from clim_helpers import szn_str2int
import rasterio
# import pyproj
from rasterio.transform import Affine


# -----------------------------Read and reformat climatology data -------------------------

def reformat_array(file_path, season):
    text_name = os.path.join(file_path, 'nep35_tem_' + season + '_extrap2.dat')
    # convert the depth part in .dat file
    with open(text_name) as lines:
        array_d_1 = np.genfromtxt(islice(lines, 1, 9), dtype=int)
    with open(text_name) as lines:
        array_d_2 = np.genfromtxt(islice(lines, 9, 10), dtype=int)
    array_d_1 = array_d_1.flatten()
    array_d = np.concatenate((array_d_1, array_d_2), axis=None)  # as the start of new array

    array = array_d  # starting array with depth
    num_lines = sum(1 for line in open(text_name))  # get number of lines
    i = 10  # starting line number

    for i in range(10, num_lines, 4):
        with open(text_name) as lines:
            array_t_1 = np.genfromtxt(islice(lines, i, i + 3), dtype=float)
        with open(text_name) as lines:
            array_t_2 = np.genfromtxt(islice(lines, i + 3, i + 4), dtype=float)
        array_t_1 = array_t_1.flatten()
        array_t_3 = np.concatenate((array_t_1, array_t_2), axis=None)
        array = np.vstack((array, array_t_3))

    tem_reformat = os.path.join(file_path, 'nep35_tem_' + season + '_extrap2_reformat')
    np.savetxt(tem_reformat, array, delimiter=',', newline='\n')
    np.save(tem_reformat, array)
    return array


# ------------------ Read and plot climatology on triangle grid------------------------


def read_climatologies(file_path, output_folder, season):
    grid_filename = os.path.join(file_path, 'nep35_reord_latlon_wgeo.ngh')
    tri_filename = os.path.join(file_path, 'nep35_reord.tri')

    data = np.genfromtxt(grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                         names=['node', 'lon', 'lat', 'type', 'depth',
                                's1', 's2', 's3', 's4', 's5', 's6'],
                         delimiter="", skip_header=3)

    tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1  # python starts from 0

    array_filename = os.path.join(file_path, output_folder + '/nep35_tem_' + season + '_extrap2_reformat.npy')

    array = np.load(array_filename)  # Load the .npy file
    array_t = array[1:]  # Index only the values not the standard levels in row 0
    array_t = np.transpose(array_t)
    grid_depth = abs(array[0])  # Take abs value of elevation to get depth (+ down)
    array_t = np.vstack((array_t, data['depth']))
    for i in range(0, 51, 1):
        array_t[i] = np.where(array_t[52] < grid_depth[i], np.nan,
                              array_t[i])  # replace the value below bottom depth with nan

    # create a data dictionary, and write data into dictionary
    data_dict = dict()
    data_dict['node_number'] = data['node'] - 1  # use node_number as Key
    data_dict['depth_in_m'] = data['depth']
    data_dict['y_lat'] = data['lat']
    data_dict['x_lon'] = data['lon']
    data_dict['grid_depth'] = abs(array[0])

    # write index for each grid depth
    for i in range(0, 52, 1):
        var_name = 'grid_depth_' + str(int(abs(grid_depth[i]))) + 'm'
        data_dict[var_name] = array_t[i]

    tri = mtri.Triangulation(data_dict['x_lon'], data_dict['y_lat'],
                             tri_data)  # attributes: .mask, .triangles, .edges, .neighbors
    # min_circle_ratio = 0.1
    # mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
    # tri.set_mask(mask)
    data_dict['triangles'] = tri.triangles
    plt.triplot(tri, color='0.7', lw=0.2)  # check grid plot
    plt.show()

    return data_dict


def read_climatologies_v2(tri_dir, clim_data_file, season):
    grid_filename = os.path.join(tri_dir, 'nep35_reord_latlon_wgeo.ngh')
    tri_filename = os.path.join(tri_dir, 'nep35_reord.tri')

    grid_ds = np.genfromtxt(grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                            names=['node', 'lon', 'lat', 'type', 'depth',
                                   's1', 's2', 's3', 's4', 's5', 's6'],
                            delimiter="", skip_header=3)

    tri_ds = np.genfromtxt(
        tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1  # python starts from 0

    # create a data dictionary, and write data into dictionary
    data_dict = dict()
    data_dict['node_number'] = np.array(grid_ds['node']) - 1  # use node_number as Key
    data_dict['depth_in_m'] = np.array(grid_ds['depth'])
    data_dict['y_lat'] = np.array(grid_ds['lat'])
    data_dict['x_lon'] = np.array(grid_ds['lon'])
    # data_dict['grid_depth'] = abs(array[0])

    tri = mtri.Triangulation(data_dict['x_lon'], data_dict['y_lat'],
                             tri_ds)  # attributes: .mask, .triangles, .edges, .neighbors
    data_dict['triangles'] = tri.triangles

    if clim_data_file.endswith('.txt'):
        clim_df = pd.read_csv(clim_data_file, sep="\t")
        var_data = np.array(clim_df["SL_value_30yr_avg @ Season={}.00".format(
            szn_str2int(season))])
    elif clim_data_file.endswith('.npy'):
        clim_array = np.load(clim_data_file)
        var_data = clim_array[1:]

    data_dict['var_data'] = var_data

    return data_dict


# -----------------------------Plot Climatology with unstructured triangle grid----------------------------------------

# left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]


def plot_clim_triangle(data_dict, file_path, left_lon, right_lon, bot_lat, top_lat,
                       output_folder, season, depth):
    tri_filename = os.path.join(file_path, 'nep35_reord.tri')
    tri_data = np.genfromtxt(
        tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1

    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    # lcc: Lambert Conformal Projection;
    # cyl: Equidistant Cylindrical Projection
    # merc: Mercator Projection

    x_lon = data_dict['x_lon']
    y_lat = data_dict['y_lat']
    xpt, ypt = m(x_lon, y_lat)  # convert lat/lon to x/y map projection coordinates in meters
    tri_pt = mtri.Triangulation(xpt, ypt, tri_data)
    # min_circle_ratio = 0.1
    # mask = TriAnalyzer(tri_pt).get_flat_tri_mask(min_circle_ratio)
    # tri_pt.set_mask(mask)
    triangles = data_dict['triangles']

    # bottom_depth = np.array(data_dict['depth_in_m'])  # as single number array
    var_name = 'grid_depth_' + depth + 'm'
    var = np.array(data_dict[var_name])

    # Create plot with map components
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    # m.drawrivers()

    # Draw depth on the map using triangulation or gridded data
    # color_map = plt.cm.get_cmap('Blues_r')
    # color_map_r = color_map.reversed()
    # cax = plt.tripcolor(xpt, ypt, triangles, var, cmap='YlOrBr', edgecolors= 'none')
    cax = plt.tripcolor(xpt, ypt, triangles, var, cmap='YlOrBr', edgecolors='none',
                        vmin=np.nanmin(var), vmax=np.nanmax(var))

    # cax = plt.tripcolor(xpt, ypt, triangles, -depth, cmap='Blues_r', edgecolors=edge_color, vmin=-5000, vmax=0)

    # set the nan to white on the map
    # masked_array = np.ma.array(var, mask=np.isnan(var)) #mask the nan values
    color_map = plt.cm.get_cmap()
    color_map.set_bad('w')
    # cax = plt.tripcolor(xpt, ypt, triangles, masked_array, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var), vmax=np.nanmax(var))

    cbar = fig.colorbar(cax, shrink=0.7)  # set scale bar
    cbar.set_label('Temperature [°C]', size=14)  # scale label
    # labels = [left,right,top,bottom]
    parallels = np.arange(bot_lat, top_lat,
                          4.)  # parallels = np.arange(48., 54, 0.2); parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  # draw parallel lat lines
    meridians = np.arange(left_lon, -100.0, 15.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, True, True])
    # plt.show()
    png_name = os.path.join(file_path, output_folder + '/T_' + season + '_tri_' + depth + 'm.png')
    fig.savefig(png_name, dpi=400)
    plt.close(fig)
    return


def plot_clim_triangle_v2(tri_dir, clim_data_file, left_lon, right_lon, bot_lat, top_lat,
                          output_dir, var_name, var_units, var_cmap, season, depth):
    """
    Need to specify file type of clim_data_file...
    :param tri_dir: directory containing 'nep35_reord_latlon_wgeo.ngh' and 'nep35_reord.tri'
    :param clim_data_file: txt file containing climatology values at each trigrid node
    :param left_lon: plotting parameter
    :param right_lon: plotting parameter
    :param bot_lat: plotting parameter
    :param top_lat: plotting parameter
    :param output_dir: directory to export the plot to; string
    :param var_name: "Oxy" or "Temp" or "Sal"
    :param var_units: units of the above variable
    :param var_cmap: 'YlOrBr' for temperature, "Blues" for oxygen and salinity
    :param season: one of ["JFM", "AMJ", "JAS", "OND"]
    :param depth: string; units of meters
    :return: string of full name including path of output plot
    """

    grid_fullpath = os.path.join(tri_dir, 'nep35_reord_latlon_wgeo.ngh')
    tri_fullpath = os.path.join(tri_dir, 'nep35_reord.tri')

    # Read in grid data
    grid_ds = np.genfromtxt(grid_fullpath, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                            names=['node', 'lon', 'lat', 'type', 'depth',
                                   's1', 's2', 's3', 's4', 's5', 's6'],
                            delimiter="", skip_header=3)

    # Read in triangle data
    tri_ds = np.genfromtxt(
        tri_fullpath, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1

    tri = mtri.Triangulation(grid_ds['lon'], grid_ds['lat'],
                             tri_ds)  # attributes: .mask, .triangles, .edges, .neighbors

    triangles = tri.triangles

    # Open the climatology data file (Oxygen, Temperature or Salinity)
    clim_df = pd.read_csv(clim_data_file, sep="\t")
    var_data = np.array(clim_df["SL_value_30yr_avg @ Season={}.00".format(szn_str2int(season))])

    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    # lcc: Lambert Conformal Projection;
    # cyl: Equidistant Cylindrical Projection
    # merc: Mercator Projection

    xpt, ypt = m(grid_ds['lon'], grid_ds['lat'])  # convert lat/lon to x/y map projection coordinates in meters

    # Create plot with map components
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')

    cax = plt.tripcolor(xpt, ypt, triangles, var_data, cmap=var_cmap, edgecolors='none',
                        vmin=230, vmax=330)
                        # vmin=np.nanmin(var_data), vmax=np.nanmax(var_data))

    # set the nan to white on the map
    # color_map = plt.cm.get_cmap()
    color_map = plt.cm.get_cmap().copy()
    color_map.set_bad('w')

    cbar = fig.colorbar(cax, shrink=0.7)  # set scale bar
    cbar.set_label('{} [{}]'.format(var_name, var_units), size=14)  # scale label
    # labels = [left,right,top,bottom]
    parallels = np.arange(bot_lat, top_lat,
                          4.)  # parallels = np.arange(48., 54, 0.2); parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  # draw parallel lat lines
    meridians = np.arange(left_lon, -100.0, 15.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, True, True])

    plt.title("Season: {}".format(season))

    png_filename = output_dir + "{}_{}m_{}_TG_mean_est.png".format(var_name, depth, season)
    fig.savefig(png_filename, dpi=400)
    plt.close(fig)

    return png_filename


# -----------------------------------------------------------------------------------------------------------------------
# set boundary

# left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]   # NE Pacific
# left_lon, right_lon, bot_lat, top_lat = [-140, -120, 45, 56]   # EEZ

def triangle_to_regular(data_dict, file_path, left_lon, right_lon, bot_lat, top_lat,
                        output_folder, season, depth):
    tri_filename = os.path.join(file_path, 'nep35_reord.tri')
    tri_data = np.genfromtxt(
        tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1

    # build regular grid mesh and interpolate value on to the regular mesh
    # print(data_dict['y_lat'].max(), data_dict['y_lat'].min())
    # print(data_dict['x_lon'].max(), data_dict['x_lon'].min())
    # xi = np.linspace(clim_data['x_lon'].min(), clim_data['x_lon'].max(), 4422)   # ~ 1000m ~ 0.01 degree, for full NE Pacific
    # yi = np.linspace(clim_data['y_lat'].min(), clim_data['y_lat'].max(), 3151)   # ~ 1000m ~ 0.01 degree, for full NE Pacific
    xi = np.linspace(221, 239, 5400)  # ~ 333m ~ 0.003 degree
    yi = np.linspace(46, 55, 2700)  # ~ 333m ~ 0.003 degree
    x_lon_r, y_lat_r = np.meshgrid(xi, yi)  # create regular grid

    # create basemap
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    xpr, ypr = m(x_lon_r, y_lat_r)  # convert lat/lon to x/y map projection coordinates in meters using basemap

    # 2nd method to convert lat/lon to x/y
    # import pyproj
    # proj_basemap = pyproj.Proj(m.proj4string) # find out the basemap projection
    # t_lon, t_lat = proj_basemap(x_lon_g, y_lat_g)

    # get triangular mesh information
    x_lon = data_dict['x_lon']
    y_lat = data_dict['y_lat']
    xpt, ypt = m(x_lon, y_lat)  # convert lat/lon to x/y map projection coordinates in meters
    tri_pt = mtri.Triangulation(xpt, ypt, tri_data)
    trifinder = tri_pt.get_trifinder()  # trifinder= mtri.Triangulation.get_trifinder(tri_pt), return the default of this triangulation

    var_name = 'grid_depth_' + depth + 'm'
    var = np.array(data_dict[var_name])

    # interpolate from triangular to regular mesh
    interp_lin = mtri.LinearTriInterpolator(tri_pt, var,
                                            trifinder=None)  # conduct interpolation on lcc projection, not on lat/long
    var_r = interp_lin(xpr, ypr)
    var_r[var_r.mask] = np.nan  # set the value of masked point to nan

    # Create figure
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    # m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    # m.scatter(xpr, ypr, color='black')
    cax = plt.pcolor(xpr, ypr, var_r, cmap='YlOrBr', edgecolors='none')
    # cax = plt.pcolor(xpt, ypt, var_r, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var_r), vmax=np.nanmax(var_r))
    plt.clim(0, 22)

    # masked_array = np.ma.array(temp_5, mask=np.isnan(temp_5)) #mask the nan values
    color_map = plt.cm.get_cmap()
    color_map.set_bad('w')  # set the nan values to white on the plot

    cbar = fig.colorbar(cax, shrink=0.7, extend='both')  # set scale bar
    cbar.set_label('Temperature [°C]', size=14)  # scale label
    parallels = np.arange(bot_lat - 1, top_lat + 1,
                          3.)  # parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  # draw parallel lat lines
    meridians = np.arange(-140, -120.0, 5.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, True, True])
    # labels = [left,right,top,bottom]
    # title_name = "Climatology_T_" + season + "_" + depth + 'm'
    # plt.title(title_name, size=15, y=1.08)

    # plt.show()
    png_name = os.path.join(file_path, output_folder + '/T_' + season + '_reg_' + depth + 'm.png')
    fig.savefig(png_name, dpi=400)
    # plt.savefig(png_name, dpi=400)

    # save the lat, lon and var on regular grid
    data_dict_new = dict()
    data_dict_new['x_lon_r'] = x_lon_r - 360
    data_dict_new['y_lat_r'] = y_lat_r
    data_dict_new[var_name] = var_r
    return data_dict_new


def triangle_to_regular_v2(data_dict, tri_dir, output_dir, var_name, depth,
                           season, left_lon, right_lon, bot_lat, top_lat,
                           var_units, var_cmap):
    tri_filename = os.path.join(tri_dir, 'nep35_reord.tri')
    tri_ds = np.genfromtxt(
        tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1

    # build regular grid mesh and interpolate value on to the regular mesh
    # xi = np.linspace(221, 239, 5400)  # ~ 333m ~ 0.003 degree
    # print((239 - 221) / 5400)
    # yi = np.linspace(46, 55, 2700)  # ~ 333m ~ 0.003 degree
    xi = np.linspace(200, 245, 13500)  # ~ 333m ~ 0.003 degree
    yi = np.linspace(30, 60, 9000)  # ~ 333m ~ 0.003 degree
    x_lon_r, y_lat_r = np.meshgrid(xi, yi)  # create regular grid

    # create basemap
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    xpr, ypr = m(x_lon_r, y_lat_r)  # convert lat/lon to x/y map projection coordinates in meters using basemap

    # get triangular mesh information
    x_lon = data_dict['x_lon']
    y_lat = data_dict['y_lat']
    xpt, ypt = m(x_lon, y_lat)  # convert lat/lon to x/y map projection coordinates in meters
    tri_pt = mtri.Triangulation(xpt, ypt, tri_ds)
    # trifinder = tri_pt.get_trifinder()
    # trifinder= mtri.Triangulation.get_trifinder(tri_pt), return the default of this triangulation

    # var_name = 'grid_depth_' + depth + 'm'
    # var = np.array(data_dict[var_name])

    var = data_dict["var_data"]

    # interpolate from triangular to regular mesh
    interp_lin = mtri.LinearTriInterpolator(
        tri_pt, var, trifinder=None)  # conduct interpolation on lcc projection, not on lat/long
    var_r = interp_lin(xpr, ypr)
    var_r[var_r.mask] = np.nan  # set the value of masked point to nan

    # # Create figure MEMORY ERROR
    # fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    # m.drawcoastlines(linewidth=0.2)
    # # m.drawmapboundary(fill_color='white')
    # m.fillcontinents(color='0.8')
    # # m.scatter(xpr, ypr, color='black')
    # cax = plt.pcolor(xpr, ypr, var_r, cmap=var_cmap, edgecolors='none', shading='auto')
    # # cax = plt.pcolor(xpt, ypt, var_r, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var_r), vmax=np.nanmax(var_r))
    # # Set colour limits of the image
    # # plt.clim(0, 22)
    #
    # # masked_array = np.ma.array(temp_5, mask=np.isnan(temp_5)) #mask the nan values
    # color_map = plt.cm.get_cmap().copy()
    # color_map.set_bad('w')  # set the nan values to white on the plot
    #
    # cbar = fig.colorbar(cax, shrink=0.7, extend='both')  # set scale bar
    # cbar.set_label('{} [{}]'.format(var_name, var_units), size=14)  # scale label
    # parallels = np.arange(bot_lat - 1, top_lat + 1,
    #                       3.)  # parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
    # m.drawparallels(parallels, labels=[True, False, False, False])  # draw parallel lat lines
    # # meridians = np.arange(-140, -120.0, 5.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    # meridians = np.arange(left_lon, -100.0, 15.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    # m.drawmeridians(meridians, labels=[False, False, True, True])
    #
    # # labels = [left,right,top,bottom]
    # # title_name = "Climatology_T_" + season + "_" + depth + 'm'
    # # plt.title(title_name, size=15, y=1.08)
    #
    # # plt.show()
    # png_name = os.path.join(output_dir + '{}_{}m_{}_TG_reg.png'.format(var_name, depth, season))
    # fig.savefig(png_name, dpi=400)
    # # plt.savefig(png_name, dpi=400)
    # plt.close(fig)

    # save the lat, lon and var on regular grid
    data_dict_new = dict()
    data_dict_new['x_lon_r'] = x_lon_r - 360
    data_dict_new['y_lat_r'] = y_lat_r
    data_dict_new[var_name] = var_r

    return data_dict_new

# -----------------------------------------------------------------------------------------------------------------------
# Write interpolated data into geoTiff file

# latlon = '+proj=longlat +datum=WGS84'
# proj_basemap = pyproj.Proj(m.proj4string) # find out the basemap projection


def convert_to_tif(data_dict, file_path, output_folder, season, depth):
    var_name = 'grid_depth_' + depth + 'm'
    x_lon_r = data_dict['x_lon_r']
    y_lat_r = data_dict['y_lat_r']
    var_r = data_dict[var_name]
    res = (x_lon_r[0][-1] - x_lon_r[0][0]) / 5400
    transform = Affine.translation(
        x_lon_r[0][0] - res / 2, y_lat_r[0][0] - res / 2) * Affine.scale(res, res)

    tif_name = os.path.join(
        file_path, output_folder + '/T_' + season + '_' + depth + 'm.tif')

    raster_output = rasterio.open(
        tif_name,
        'w',
        driver='GTiff',
        height=var_r.shape[0],
        width=var_r.shape[1],
        count=1,
        dtype=var_r.dtype,
        crs='+proj=longlat +datum=WGS84',
        # crs='epsg:4269', #epsg:4326, 4269, 3005 crs='+proj=latlong', #epsg:4326, Proj4js.defs["EPSG:4326"] = "+proj=longlat +ellps=GRS80 +datum=NAD83 +no_defs"
        # crs = '+proj=aea +lat_1=50 +lat_2=58.5 +lat_0=45 +lon_0=-126 +x_0=1000000 +y_0=0 +datum=NAD83 +units=m +no_defs',
        transform=transform,
        nodata=0
    )

    raster_output.write(var_r.data, 1)
    raster_output.close()
    return


def convert_to_tif_v2(data_dict, clim_data_file, output_dir, var_name, depth, season):
    """
    Convert txt to tif
    :param data_dict: dictionary
    :param clim_data_file:
    :param output_dir:
    :param var_name:
    :param depth:
    :param season:
    :return:
    """

    # # Read in triangle data
    # tri_ds = np.genfromtxt(
    #     tri_fullpath, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1
    #
    # tri = mtri.Triangulation(grid_ds['lon'], grid_ds['lat'],
    #                          tri_ds)  # attributes: .mask, .triangles, .edges, .neighbors

    # Open the climatology data file (Oxygen, Temperature or Salinity)
    clim_df = pd.read_csv(clim_data_file, sep="\t")
    var_r = np.array(clim_df["SL_value_30yr_avg @ Season={}.00".format(szn_str2int(season))])

    x_lon_r = data_dict['x_lon_r']
    y_lat_r = data_dict['y_lat_r']

    # res = (x_lon_r[0][-1] - x_lon_r[0][0]) / 5400
    res = (x_lon_r[0][-1] - x_lon_r[0][0]) / 13500
    # Affine.translation(xoff, yoff)
    # Affine.scale(scaling) -- Create a scaling transform from a scalar or vector
    # transform = Affine.translation(
    #     x_lon_r[0][0] - res / 2, y_lat_r[0][0] - res / 2) * Affine.scale(res, res)
    transform = Affine.translation(
        x_lon_r[0] - res / 2, y_lat_r[0] - res / 2) * Affine.scale(res, res)

    tif_name = os.path.join(output_dir + "{}_{}m_{}_mean_est.tif".format(
        var_name, depth, season))

    raster_output = rasterio.open(
        tif_name,
        'w',
        driver='GTiff',
        height=var_r.shape[0],
        width=var_r.shape[1],
        count=1,
        dtype=var_r.dtype,
        crs='+proj=longlat +datum=WGS84',
        # crs='epsg:4269', #epsg:4326, 4269, 3005 crs='+proj=latlong', #epsg:4326, Proj4js.defs["EPSG:4326"] = "+proj=longlat +ellps=GRS80 +datum=NAD83 +no_defs"
        # crs = '+proj=aea +lat_1=50 +lat_2=58.5 +lat_0=45 +lon_0=-126 +x_0=1000000 +y_0=0 +datum=NAD83 +units=m +no_defs',
        transform=transform,
        nodata=0
    )

    # raster_output.write(var_r.data, 1)
    raster_output.write(var_r, 1)
    raster_output.close()
    return tif_name


# ------------------------- fit into EEZ polygon shapefile------------------------

def EEZ_clip(file_path, output_folder, season, depth):
    tif_name = os.path.join(
        file_path, output_folder + '/T_' + season + '_' + depth + 'm.tif')
    tif_name_mask = os.path.join(
        file_path, output_folder + '/T_' + season + '_' + depth + 'm_BCEEZ.tif')
    with fiona.open("/home/guanl/Desktop/MSP/Shapefiles/BC_EEZ/BC_EEZ/bc_eez.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(tif_name) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(tif_name_mask, "w", **out_meta) as dest:
        dest.write(out_image)


EEZ_clip(file_path='/home/guanl/Desktop/MSP/Climatology', output_folder='T_sum',
         season='sum', depth='0')


# --------------------------------------------------------------------------------
# ----------------------------Set parameters--------------------------------------

# Specify index
variable_name = "Oxy"
variable_units = r"$\mu$" + "mol/kg"
var_colourmap = "Blues"
season_abbrev = "JFM"  # ["JFM", "AMJ", "JAS", "OND"] 'spr'
season_abbrevs = ["JFM", "AMJ", "JAS", "OND"]
standard_depth = '0'
# output_folder = 'T_spr'
# output_folder = "{}m".format(standard_depth)

# -----------------------------set file paths-------------------------------------------

# file_path = '/home/guanl/Desktop/MSP/Climatology'
trigrid_path = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\MForeman\\"
clim_data_path = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\" \
                 "ODV_outputs\\{}m\\".format(standard_depth)
# output_path = '/home/guanl/Desktop/MSP/Climatology/'
output_path = clim_data_path
grd_file = os.path.join(trigrid_path, 'nep35_reord_latlon_wgeo.ngh')
tri_file = os.path.join(trigrid_path, 'nep35_reord.tri')
# tem_file = os.path.join(file_path, 'nep35_tem_' + season + '_extrap2.dat')
# tem_reformat = os.path.join(file_path, 'nep35_tem_' + season + '_extrap2_reformat')

# ------------------------------run functions-------------------------------------------

fname = clim_data_path + "Oxy_{}m_{}_TG_mean_est.txt".format(
    standard_depth, season_abbrev)
dat_dict = read_climatologies_v2(trigrid_path, fname, season_abbrev)

left_lon, right_lon, bot_lat, top_lat = [-160, -102, 25, 62]

dat_dict_new = triangle_to_regular_v2(
    dat_dict, trigrid_path, clim_data_path, "Oxy", standard_depth, season_abbrev,
    left_lon=-160, right_lon=-102, bot_lat=25, top_lat=62, var_units=variable_units,
    var_cmap=var_colourmap)

for sa in season_abbrevs[:]:
    print(sa)
    clim_data_filename = os.path.join(clim_data_path + "{}_{}m_{}_TG_mean_est.txt".format(
        variable_name, standard_depth, sa))

    # # Plot the climatology for the selected variable
    # plot_clim_triangle_v2(
    #     trigrid_path, clim_data_filename, left_lon=-160, right_lon=-102, bot_lat=25,
    #     top_lat=62, output_dir=output_path, var_name=variable_name,
    #     var_units=variable_units, var_cmap=var_colourmap, season=sa,
    #     depth=standard_depth)

    # Must convert triangle to regular before converting to tif
    clim_data_r = triangle_to_regular_v2()

    # Convert txt file to tif file
    convert_to_tif_v2(trigrid_path, clim_data_filename, output_path,
                      var_name=variable_name, depth=standard_depth, season=sa)

# ---------------------------------------------------------------------------
# ------------------------See what this .npy file is like--------------------
npy_filename = "C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\" \
               "nep35_tem_spr_extrap2_reformat.npy"
npy_data = np.load(npy_filename)
print(npy_data.shape)
# Use np.save("filename", data_array) to save data to a .npy file
# ----------------------------------------------------------------------------

# Lu's code...

# reformat the climatology data to array
array = reformat_array(trigrid_path, season_abbrev)

# Read and plot climatology on triangle grid
clim_data = read_climatologies(file_path='/home/guanl/Desktop/MSP/Climatology',
                               output_folder=output_path, season=season_abbrev)

# plot climatology on triangle grid
plot_clim_triangle(clim_data, file_path='/home/guanl/Desktop/MSP/Climatology',
                   left_lon=-160, right_lon=-102, bot_lat=25, top_lat=62,
                   output_folder=output_path, season=season_abbrev,
                   depth=standard_depth)

# Convert and plot climatology on regular grid
left_lon, right_lon, bot_lat, top_lat = [-140, -120, 45, 56]

clim_data_r = triangle_to_regular(
    clim_data, file_path='/home/guanl/Desktop/MSP/Climatology', left_lon=-140,
    right_lon=-120, bot_lat=45, top_lat=56, output_folder='T_spr',
    season=season_abbrev, depth=standard_depth)

# Convert to raster layer and save in .tif format
convert_to_tif(clim_data_r, file_path='/home/guanl/Desktop/MSP/Climatology',
               output_folder='T_spr', season=season_abbrev, depth=standard_depth)

# use EEZ polygon to clip on GeoTiff
EEZ_clip(file_path='/home/guanl/Desktop/MSP/Climatology', output_folder='T_spr',
         season=season_abbrev, depth=standard_depth)

# ------------------------------- producing plots together --------------------------

for i in range(0, 10, 5):
# for i in range(10, 210, 10):
# for i in range(220, 420, 20):
# for i in range(500, 1100, 100):
# for i in range(1200, 2600, 200):
# for i in range(3000, 4000, 500):
    clim_data_r = triangle_to_regular(
        clim_data, file_path='/home/guanl/Desktop/MSP/Climatology', left_lon=-140,
        right_lon=-120, bot_lat=45, top_lat=56, output_folder='T_win', season='win',
        depth=str(i))
    convert_to_tif(clim_data_r, file_path='/home/guanl/Desktop/MSP/Climatology',
                   output_folder='T_win', season='win', depth=str(i))
    EEZ_clip(file_path='/home/guanl/Desktop/MSP/Climatology', output_folder='T_win',
             season='win', depth=str(i))

# ---------------------produce template files for Oxygen-----------------------------
output_folder = 'T_win'
season = 'win'

grid_filename = os.path.join(trigrid_path, 'nep35_reord_latlon_wgeo.ngh')
tri_filename = os.path.join(trigrid_path, 'nep35_reord.tri')

data = np.genfromtxt(grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                     names=['node', 'lon', 'lat', 'type', 'depth',
                            's1', 's2', 's3', 's4', 's5', 's6'],
                     delimiter="", skip_header=3)

tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1  # python starts from 0
array_filename = os.path.join(trigrid_path, output_folder + '/nep35_tem_' + season + '_extrap2_reformat.npy')

array = np.load(array_filename)
array_t = array[1:]
array_t = np.transpose(array_t)
grid_depth = abs(array[0])
array_t = np.vstack((array_t, data['depth']))

for i in range(0, 51, 1):
    array_t[i] = np.where(array_t[52] < grid_depth[i], np.nan, array_t[i])

array_t_na = array_t[0:52]
array_t_na = np.transpose(array_t_na)

t = np.zeros(shape=(97959, 2))
t[:, 0] = data['lat']  # write latitude first column
t[:, 1] = data['lon']  # write longitude first column
t_output = np.hstack((t, array_t_na))
pd.DataFrame(t_output).to_csv("output_t_win_na.csv")

# ----------------  prepare data for defined region-----------------------------------------------------

data = pd.read_csv('/home/guanl/Desktop/MSP/Climatology/Climatology_output_for_Nick/Climatology_S_win.csv')

data_1 = data.loc[(data['Lon'] > 231.80) & (data['Lon'] < 233.16) & (data['Lat'] > 50.55) & (data['Lat'] < 51.3)]

# define the left line:
lon_min_1 = 231.8
lon_max_1 = 232.2
lat_min_1 = 50.8
lat_max_1 = 51.3

# equation of the line using points A and B -> y=m*x + z
m_1 = (lat_max_1 - lat_min_1) / (lon_max_1 - lon_min_1)
z_1 = lat_max_1 - m_1 * lon_max_1

# select only points below the line
data_2 = data_1[data_1['Lon'] * m_1 + z_1 > data_1['Lat']]

# define the right line:
lon_min_2 = 232.95
lon_max_2 = 233.16
lat_min_2 = 50.55
lat_max_2 = 50.9

# equation of the line using points A and B -> y=m*x + z
m_2 = (lat_max_2 - lat_min_2) / (lon_max_2 - lon_min_2)
z_2 = lat_max_2 - m_2 * lon_max_2

# select only points below the line
data_3 = data_2[data_2['Lon'] * m_2 + z_2 < data_2['Lat']]

left_lon, right_lon, bot_lat, top_lat = [-126, -129, 50, 52]

m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',  # width=40000, height=40000, #lambert conformal project
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

x_lon = data_3['Lon']
y_lat = data_3['Lat']
x_lon = x_lon.to_numpy()
y_lat = y_lat.to_numpy()
xpt, ypt = m(x_lon, y_lat)  # convert lat/lon to x/y map projection coordinates in meters
fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
m.drawcoastlines(linewidth=0.2)
m.scatter(xpt, ypt, marker='D', color='m', s=4)

pd.DataFrame(data_3).to_csv("/home/guanl/Desktop/MSP/Climatology/Climatology_output_for_Nick/Di/Climatology_S_win.csv")
