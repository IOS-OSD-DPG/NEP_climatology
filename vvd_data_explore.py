"""Sept. 9, 2021
Plots of spatial and temporal distribution of value vs depth data

First goal is for using on cleaned oxygen data
"""

import glob
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from clim_helpers import date_string_to_datetime


def subset_latlon(filepath, months):
    # Read in data
    df = pd.read_csv(filepath)
    
    # Add column to df for datetime format time
    df = date_string_to_datetime(df=df)
    
    # Subset df
    subsetter_szn = np.where((df.Time_pd.dt.month >= months[0]) &
                             (df.Time_pd.dt.month <= months[-1]))[0]
    
    prof_start_ind = np.unique(df.Profile_number, return_index=True)[1]
    
    lon_subset = np.array(df.loc[np.intersect1d(subsetter_szn, prof_start_ind),
                                 'Longitude'])
    
    lat_subset = np.array(df.loc[np.intersect1d(subsetter_szn, prof_start_ind),
                                 'Latitude'])
    
    return lon_subset, lat_subset


def subset_max_depth(filepath, months):
    # Get maximum depth of each profile
    
    df = pd.read_csv(filepath)

    # Add column to df for datetime format time
    df = date_string_to_datetime(df=df)

    # Subset df
    subsetter_szn = np.where((df.Time_pd.dt.month >= months[0]) &
                             (df.Time_pd.dt.month <= months[-1]))[0]

    prof_start_ind = np.unique(df.Profile_number, return_index=True)[1]
    
    prof_end_ind = np.concatenate([prof_start_ind[1:] - 1, np.array([len(df) - 1])])

    # Need to account for any upcasts (is mostly downcasts)
    downcast_where = np.where(
        np.array(df.loc[prof_start_ind, 'Depth_m']) <
        np.array(df.loc[prof_end_ind, 'Depth_m']))[0]

    upcast_where = np.where(
        np.array(df.loc[prof_start_ind, 'Depth_m']) >
        np.array(df.loc[prof_end_ind, 'Depth_m']))[0]
    
    subsetter_depth = np.union1d(prof_end_ind[downcast_where],
                                 prof_start_ind[upcast_where])

    max_depths = np.array(df.loc[np.intersect1d(
        subsetter_depth, subsetter_szn), 'Depth_m'])
    
    time_subset = np.array(df.loc[np.intersect1d(
        subsetter_depth, subsetter_szn), 'Time_pd'])
    
    return max_depths, time_subset


def vvd_prof_map(dflist, output_folder, var, qced=False, verbose=False):
    # Check how many files in dflist
    if len(dflist) > 1:
        multifile = True
    else:
        multifile = False
        
    if verbose:
        print('multifile', multifile)

    # Initialize figure
    # fig = plt.figure(num=None, figsize=(8, 8), dpi=100)
    fig = plt.figure(figsize=(7.2, 5.4))

    szn_abbrevs = ['JFM', 'AMJ', 'JAS', 'OND']

    # Iterate through the seasons
    for j in range(4):
        # Add subplot per season
        ax = fig.add_subplot(2, 2, j + 1)
        
        months = np.arange(3 * j + 1, 3 * j + 4)
        
        # Subset lat and lon by season
        lon_subset, lat_subset = subset_latlon(dflist[0], months=months)

        left_lon = -162.
        bot_lat = 22.
        right_lon = -100.
        top_lat = 62.

        # Set up Lambert conformal map
        m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                    urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
                    resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                    lon_0=0.5 * (left_lon + right_lon))

        m.drawcoastlines(linewidth=0.2)
        m.drawmapboundary(fill_color='white')
        m.fillcontinents(color='0.8')

        # Plot the locations of the samples
        x, y = m(lon_subset, lat_subset)
        # Plot on the subplot ax
        m.scatter(x, y, marker='o', color='r', s=0.5)
        
        if multifile:
            for i in range(1, len(dflist)):
                lon_subset, lat_subset = subset_latlon(dflist[i], months)
                # Plot the locations of the samples
                x, y = m(lon_subset, lat_subset)
                # Plot on the subplot ax
                m.scatter(x, y, marker='o', color='r', s=0.5)

        # Make subplot titles
        ax.set_title(szn_abbrevs[j])
                
    # Set figure title
    if qced:
        plt.suptitle('{} 1991-2020 qced observed level data'.format(var))
    else:
        plt.suptitle('{} 1991-2020 observed level data'.format(var))

    # Export plot
    if qced:
        note = '_qced'
    else:
        note = ''
    png_name = output_folder + '{}_1991_2020_spatial_dist{}.png'.format(var, note)
    plt.savefig(png_name, dpi=400)
    plt.close(fig)
    
    return png_name


def count_prof_per_yr(filepath, months):
    df = pd.read_csv(filepath)

    prof_start_ind = np.unique(df.Profile_number, return_index=True)[1]

    # Add datetime column: Time_pd
    df = date_string_to_datetime(df)

    years = np.arange(1991, 2021, 1)

    nyr = len(years)

    counts = np.zeros(nyr, dtype='int32')

    for j in range(nyr):
        subsetter = np.where((df.Time_pd.dt.year == years[j]) &
                             (df.Time_pd.dt.month >= months[0]) &
                             (df.Time_pd.dt.month <= months[-1]))
        counts[j] = len(np.intersect1d(prof_start_ind, subsetter))

    return years, counts


def vvd_temporal_dist(dflist, output_folder, var, qced=False, verbose=False):
    # Check how many files in dflist
    if len(dflist) > 1:
        multifile = True
    else:
        multifile = False
    
    if verbose:
        print('multifile', multifile)

    # Initialize figure
    fig = plt.figure(figsize=(7.2, 5.4))

    szn_abbrevs = ['JFM', 'AMJ', 'JAS', 'OND']

    # Iterate through the seasons
    for j in range(4):
        # Add subplot per season
        ax = fig.add_subplot(2, 2, j + 1)
    
        months = np.arange(3 * j + 1, 3 * j + 4)

        years, counts = count_prof_per_yr(dflist[0], months)

        ax.scatter(years, counts, s=0.8)

        if multifile:
            for i in range(1, len(dflist)):
                years, counts = count_prof_per_yr(dflist[i], months)

                ax.scatter(years, counts, s=0.8)

        ax.set_title(szn_abbrevs[j])

    # Adjust hspace
    fig.subplots_adjust(hspace=0.3)

    if qced:
        plt.suptitle('{} 1991-2020 profile counts qced'.format(var))
        note = '_qced'
    else:
        plt.suptitle('{} 1991-2020 profile counts'.format(var))
        note = ''

    png_name = output_folder + '{}_1991_2020_prof_counts{}.png'.format(var, note)

    plt.savefig(png_name, dpi=400)

    plt.close(fig)

    return png_name


def vvd_max_depth_vs_time(dflist, output_folder, var, qced=False, verbose=False):
    # Check how many files in dflist
    if len(dflist) > 1:
        multifile = True
    else:
        multifile = False
    
    if verbose:
        print('multifile', multifile)

    # Initialize figure
    fig = plt.figure(figsize=(7.2, 5.4))

    szn_abbrevs = ['JFM', 'AMJ', 'JAS', 'OND']

    # Iterate through the seasons
    for j in range(4):
        # Add subplot per season
        ax = fig.add_subplot(2, 2, j + 1)
    
        months = np.arange(3 * j + 1, 3 * j + 4)

        depth_subset, time_subset = subset_max_depth(dflist[0], months)

        ax.scatter(time_subset, depth_subset, s=0.3)

        if multifile:
            for i in range(1, len(dflist)):
                depth_subset, time_subset = subset_max_depth(dflist[i], months)

                ax.scatter(time_subset, depth_subset, s=0.3)

        # Invert y axis
        ax.set_ylim(ax.get_ylim()[::-1])

        # ax.set_xticks(pd.to_datetime(
        #     pd.Series(['1991', '1995', '2000', '2005',
        #                '2010', '2015', '2020']), format='%Y'))

        # Label each subplot
        ax.set_title(szn_abbrevs[j])

    # Adjust hspace
    fig.subplots_adjust(hspace=0.3)

    if qced:
        plt.suptitle('{} 1991-2020 profile maximum depths qced'.format(var))
        note = '_qced'
    else:
        plt.suptitle('{} 1991-2020 profile maximum depths'.format(var))
        note = ''

    png_name = output_folder + '{}_1991_2020_max_dep{}.png'.format(var, note)

    plt.savefig(png_name, dpi=400)

    plt.close(fig)
      
    return png_name


indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
        'value_vs_depth\\8_gradient_check\\latlon_check\\'
infiles = glob.glob(indir + '*Oxy*done.csv')

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
         'data_explore\\oxygen\\qced\\'

vvd_prof_map(infiles, outdir, 'Oxy', qced=True, verbose=True)

vvd_temporal_dist(infiles, outdir, 'Oxy', qced=True, verbose=True)

vvd_max_depth_vs_time(infiles, outdir, 'Oxy', qced=True, verbose=True)
