"""Sept 8, 2021
Functions to facilitate working with climatology files
"""

import pandas as pd
from xarray import open_dataset
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.basemap import Basemap
import os


def concat_vvd_files(flist, outdir, dfname):
    # Concatenate value vs depth csv files and retain unique profile numbers for each profile
    df_all = pd.DataFrame()

    profile_counter = 0
    for f in flist:
        df_add = pd.read_csv(f)
        if df_all.empty:
            df_add.loc[:, 'Profile_number'] += profile_counter
        else:
            df_add.loc[:, 'Profile_number'] += profile_counter + 1

        df_all = pd.concat([df_all, df_add])
        df_all.reset_index(drop=True, inplace=True)
        profile_counter += df_all.loc[len(df_all) - 1, 'Profile_number']  # index up to len - 1

    # all_name = 'ALL_Oxy_1991_2020_value_vs_depth_nan_rm.csv'
    all_name = outdir + dfname

    df_all.to_csv(all_name, index=False)

    return all_name


def date_string_to_datetime(df):
    # Add a column containing the date string in pandas datetime format
    # Return the updated dataframe
    # df MUST CONTAIN COLUMN TITLED "Date_string"
    df.insert(len(df.columns), 'Time_pd',
              pd.to_datetime(df.Date_string, format='%Y%m%d%H%M%S'))

    return df


def open_by_source(full_path):
    # Open data file based on which data centre it came from
    # IOS and NODC files are netCDF
    # MEDS files are csv
    if full_path.endswith('.nc'):
        data = open_dataset(full_path)
    elif full_path.endswith('.csv'):
        data = pd.read_csv(full_path)
    else:
        print("Warning: data of incorrect format; must be .csv or .nc")
        return None
    return data


def vvd_apply_value_flag(df, flag_name):
    # Apply flag and generate new copy of df
    # Flag=0 means data passed the check so want to keep those data

    df = df.loc[df[flag_name] == 0]

    df_return = df.drop(columns=flag_name)

    return df_return


def get_standard_levels(fpath_sl):
    # Return array of standard levels from the standard levels text file
    # fpath_sl is the full file path of the standard levels txt file
    # which contains 102 standard levels between 0m depth and 5500m depth

    # Initialize list with each element being a row in file_sl
    sl_list = []
    with open(fpath_sl, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            sl_list += row

    # Remove empty elements: '' and ' '
    # Gotta love list comprehension
    sl_list_v2 = [int(x.strip(' ')) for x in sl_list if x not in ['', ' ']]

    # Convert list to array
    sl_arr = np.array(sl_list_v2)
    return sl_arr


def deg2km(dlat):
    # From DIVAnd.jl
    # Convert decimal degrees distance into km distance
    # Mean radius (http://en.wikipedia.org/wiki/Earth_radius) in km
    R = 6371.009

    return dlat * (2 * np.pi * R) / 360


def plot_divand_analysis(output_dir, lon2d, lat2d, var_field, var_cmap, var_name, var_units,
                         lon_obs, lat_obs, depth, yr, szn, corlen_method):
    """
    Plot the output field from DIVAnd interpolation
    :param output_dir: output directory path
    :param lon2d: 2d mesh grid of longitude values with shape (m, n)
    :param lat2d: 2d mesh grid of latitude values with shape (m, n)
    :param var_field: 2d field of interpolated observations with shape (m, n)
    :param var_cmap: colormap for the plot; use "Blues" for Oxygen and salinity
    :param var_name: "Oxy", "Temp" or "Sal"
    :param var_units: string specifing the variable units
    :param lon_obs: 1d array of longitude of standard level observations
    :param lat_obs: 1d array of latitude of standard level observations
    :param depth: depth of var_field; int type
    :param yr: year; int type
    :param szn: season, one of "JFM", "AMJ", "JAS", or "OND"
    :param corlen_method: method used to estimate the correlation length;
    "fithorzlen" or "GCV" (generalized cross-validation)
    :return:
    """
    plt.pcolormesh(lon2d, lat2d, var_field, shading='auto', cmap=var_cmap)  # , vmin=150, vmax=400)
    plt.colorbar(label='{} [{}]'.format(var_name, var_units))  # ticks=np.arange(150, 400 + 1, 50)

    # Scatter plot the observation points
    plt.scatter(lon_obs, lat_obs, c='k', s=0.1)
    plt.title('{} {}m {} {}: lenxy from {}'.format(var_name, depth, yr, szn, corlen_method))

    # Set limits
    plt.xlim((-160., -102.))
    plt.ylim((25., 62.))

    plt_filename = os.path.join(output_dir + "{}_{}m_{}_{}_analysis2d_gebco.png".format(
        var_name, depth, yr, szn))
    plt.savefig(plt_filename, dpi=400)

    plt.close()

    return plt_filename


def szn_str2int(szn_string):
    szns = np.array(["JFM", "AMJ", "JAS", "OND"])
    if szn_string in szns:
        return np.where(szns == szn_string)[0][0] + 1
    else:
        print("Warning: Season {} not in {}".format(szn_string, szns))
        return None


def plot_linterp_tg_data(tg_var_file, tri_dir, output_dir, var_name, var_units, var_cmap,
                         depth, yr, szn, avged=False):
    var_ds = open_dataset(tg_var_file)

    if avged:
        var_ds_data = var_ds.SL_value_30yr_avg.data
    else:
        var_ds_data = var_ds.SL_value.data

    tri_fullpath = os.path.join(tri_dir, 'nep35_reord.tri')

    # Read in triangle data .tri
    tri_ds = np.genfromtxt(
        tri_fullpath, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1

    tri = mtri.Triangulation(var_ds.longitude.data, var_ds.latitude.data,
                             tri_ds)  # attributes: .mask, .triangles, .edges, .neighbors

    triangles = tri.triangles

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

    # convert lat/lon to x/y map projection coordinates in meters
    xpt, ypt = m(var_ds.longitude.data, var_ds.latitude.data)

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')

    cax = plt.tripcolor(xpt, ypt, triangles, var_ds_data, cmap=var_cmap,
                        edgecolors='none', vmin=200, vmax=300)
    # vmin=np.nanmin(var_data), vmax=np.nanmax(var_data))

    # set the nan to white on the map
    color_map = plt.cm.get_cmap().copy()
    color_map.set_bad('w')

    # Set more plot parameters
    cbar = fig.colorbar(cax, shrink=0.7)  # set scale bar
    cbar.set_label('{} [{}]'.format(var_name, var_units), size=14)  # scale label
    # labels = [left,right,top,bottom]
    parallels = np.arange(bot_lat, top_lat,
                          4.)  # parallels = np.arange(48., 54, 0.2); parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  # draw parallel lat lines
    meridians = np.arange(left_lon, -100.0, 15.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, True, True])

    # Use y parameter to move title up to avoid overlap with axis ticks
    if avged:
        plt.title("{} {}m 1991-2020 {} TG".format(var_name, depth, szn), y=1.08)
    else:
        plt.title("{} {}m {} {} TG".format(var_name, depth, yr, szn), y=1.08)

    png_filename = os.path.join(output_dir + '{}_{}m_{}_{}_tg.png'.format(
        var_name, depth, yr, szn))
    fig.savefig(png_filename, dpi=400)
    plt.close(fig)

    return png_filename
