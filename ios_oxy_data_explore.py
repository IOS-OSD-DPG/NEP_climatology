# Python script for plotting spatial and temporal distributions of oxygen data
# Want to evaluate whether the data are suitable to use for climatology

# Explore titration (bottle) data only first
# Choose bottle first because WOA18 only used titration oxygen data
# All IOS oxygen sensor data is calibrated against Winkler titration
# oxygen data, so do PCTD data next

import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import trange
from pandas import to_datetime, DataFrame
from mpl_toolkits.basemap import Basemap
from itertools import chain
# import cartopy.crs as ccrs ???


def ios_map_dist(profile_data, time_data, lat_data, lon_data, output_folder,
                 instrument, left_lon, bot_lat, right_lon, top_lat, szn,
                 var='oxygen', multifile=False, verbose=False):
    # Plot spatial distribution of data on a map using the Basemap package
    # See if some geographic regions are underrepresented
    # Use the Basemap package for creating maps
    
    # ncdata: netCDF file data that was read in with xarray.open_dataset
    # instrument: 'BOT' for bottle, 'PCTD' for profiling CTD, 'MCTD' for
    #             moored CTD
    # left_lon, bot_lat, right_lon, top_lat: corner coordinates for the
    #                                        Basemap map
    # szn: 'Winter', 'Spring', 'Summer', 'Fall', or 'All'
    # multifile: Parameter to prevent closing figure prematurely. Default False
    # verbose: Prints out messages during code running if True. Default False

    # Get list of unique profiles
    _, unique_indices = np.unique(profile_data, return_index=True)

    # Assign months to plot
    if szn == 'Winter':
        months = np.arange(1, 4)
    elif szn == 'Spring':
        months = np.arange(4, 7)
    elif szn == 'Summer':
        months = np.arange(7, 10)
    elif szn == 'Fall':
        months = np.arange(10, 13)
    elif szn == 'All':
        months = np.arange(1, 13)
    else:
        print('Invalid value for szn:', szn)

    # Convert to pandas datetime object for easier indexing by year, month
    time_subset = to_datetime(time_data[unique_indices])
    
    if verbose:
        print('Unique profile times extracted')
    
    # Extract the data from the selected season
    subsetter = np.where((time_subset.month >= months[0]) & (time_subset.month <= months[-1]))
    lat_subset = lat_data[subsetter]
    lon_subset = lon_data[subsetter]
    
    if verbose:
        print('Lat and lon subsetted by input season')

    # Set up Lambert conformal map
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))

    # Use NASA's "Blue Marble" image
    # m.bluemarble()

    # Plot lat and lon data as red markers

    # Initialize figure
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    # ax = plt.subplot(1, 1, 1) # need in order to add points to plot iteratively?
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    
    # Plot the locations of the samples
    x, y = m(lon_subset, lat_subset)
    # Plot on the subplot ax
    m.scatter(x, y, marker='o', color='r', s=0.5)

    if verbose:
        print('Map and data plotted')

    plt.title('IOS {} {} 1991-2020 {}'.format(instrument, var, szn))

    png_name = output_folder + 'IOS_{}_{}_{}_spatial_dist.png'.format(instrument, var, szn)
    
    # Close the figure if done plotting all datasets
    if not multifile:
        if verbose:
            print('Closing map figure')
        plt.savefig(png_name, dpi=400)
        plt.close(fig)

    return png_name


def ios_cmap_dist(profile_data, time_data, lat_data, lon_data, depth_data,
                  output_folder, instrument, left_lon, bot_lat, right_lon,
                  top_lat, szn, var='oxygen', multifile=False, verbose=False):
    # DID NOT USE (points too small to discern colour. It was a fair idea though..
    # Plot spatial distribution of data on a map using the Basemap package
    # Use colour map where colour of points depends on maximum depth of profiles
    # See if some geographic regions are underrepresented
    # Use the Basemap package for creating maps
    
    # ncdata: netCDF file data that was read in with xarray.open_dataset
    # instrument: 'BOT' for bottle, 'PCTD' for profiling CTD, 'MCTD' for
    #             moored CTD
    # left_lon, bot_lat, right_lon, top_lat: corner coordinates for the
    #                                        Basemap map
    # szn: 'Winter', 'Spring', 'Summer', 'Fall', or 'All'
    # multifile: Parameter to prevent closing figure prematurely. Default False
    # verbose: Prints out messages during code running if True. Default False
    
    # Get list of unique profiles
    _, unique_indices = np.unique(profile_data, return_index=True)
    
    # Assign months to plot
    if szn == 'Winter':
        months = np.arange(1, 4)
    elif szn == 'Spring':
        months = np.arange(4, 7)
    elif szn == 'Summer':
        months = np.arange(7, 10)
    elif szn == 'Fall':
        months = np.arange(10, 13)
    elif szn == 'All':
        months = np.arange(1, 13)
    else:
        print('Invalid value for szn:', szn)
    
    # Convert to pandas datetime object for easier indexing by year, month
    time_subset = to_datetime(time_data[unique_indices])
    
    if verbose:
        print('Unique profile times extracted')
    
    # Extract the data from the selected season
    subsetter = np.where((time_subset.month >= months[0]) & (time_subset.month <= months[-1]))[0]
    print(len(subsetter))
    lat_subset = lat_data[subsetter]
    lon_subset = lon_data[subsetter]
    # Index the maximum depth of each profile
    # Include the very last depth entry: index == -1
    max_dep_subsetter = np.concatenate((subsetter[1:] - 1, np.array([-1])), axis=None)
    depth_subset = depth_data[max_dep_subsetter]
    
    if verbose:
        print('Lat and lon subsetted by input season')
    
    # Set up Lambert conformal map
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))
    
    # Use NASA's "Blue Marble" image
    # m.bluemarble()
    
    # Plot lat and lon data as red markers
    
    # Initialize figure
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    # ax = plt.subplot(1, 1, 1) # need in order to add points to plot iteratively?
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    
    # Plot the locations of the samples
    x, y = m(lon_subset, lat_subset)
    # Plot on the subplot ax; use white-to-red colormap
    scatter = m.scatter(x, y, c=depth_subset, marker='o', cmap='OrRd', s=1, edgecolor='black')
    
    if verbose:
        print('Map and data plotted')
    
    handles, labels = scatter.legend_elements(prop='colors')
    plt.legend(handles, labels, loc='lower left', title='Profile max depth')
    
    if verbose:
        print('Legend completed')
    
    plt.title('IOS {} {} 1991-2020 {}'.format(instrument, var, szn))
    
    png_name = output_folder + 'IOS_{}_{}_{}_spatial_dist_cmap.png'.format(instrument, var, szn)
    plt.savefig(png_name, dpi=400)
    
    # Close the figure if done plotting all datasets
    if not multifile:
        if verbose:
            print('Closing map figure')
        plt.close(fig)
    
    return png_name


def ios_get_sll(ncfile, months, verbose=False):
    # Get subsetted latitude and longitude from an IOS netCDF file
    # Adjust the code for one file first, then incorporate multifile option
    # Read in netCDF data
    ncdata = xr.open_dataset(ncfile)

    # Get list of unique profiles
    _, unique_indices = np.unique(ncdata.profile.data, return_index=True)

    # Convert to pandas datetime object for easier indexing by year, month
    time_subset = to_datetime(ncdata.time.data[unique_indices])

    if verbose:
        print('Unique profile times extracted')

    # Extract the data from the selected season
    subsetter = np.where((time_subset.month >= months[0]) & (time_subset.month <= months[-1]))
    lat_subset = ncdata.latitude.data[subsetter]
    lon_subset = ncdata.longitude.data[subsetter]

    if verbose:
        print('Lat and lon subsetted by input season')
        
    # Close the netCDF file for memory reasons
    ncdata.close()
    
    return lat_subset, lon_subset


def ios_map_dist_v2(nclist, output_folder, instrument, left_lon, bot_lat,
                    right_lon, top_lat, szn, var='oxygen', verbose=False):
    # TAKES DATA FILE PATH AS INPUT instead of arrays of data
    # Plot spatial distribution of data on a map using the Basemap package
    # See if some geographic regions are underrepresented
    # Use the Basemap package for creating maps
    
    # nclist: list of netCDF file paths to be read in with xarray.open_dataset
    # output_folder: full path of output folder for files generated by this function
    # instrument: 'BOT' for bottle, 'PCTD' for profiling CTD, 'MCTD' for
    #             moored CTD
    # left_lon, bot_lat, right_lon, top_lat: corner coordinates for the
    #                                        Basemap map
    # szn: 'Winter', 'Spring', 'Summer', 'Fall', or 'All'
    # verbose: Prints out messages during code running if True. Default False
    
    # Check the multifile option
    if len(nclist) > 1:
        multifile = True
    else:
        multifile = False
        
    if verbose:
        print('multifile', multifile)

    # Assign months to plot
    if szn == 'Winter':
        months = np.arange(1, 4)
    elif szn == 'Spring':
        months = np.arange(4, 7)
    elif szn == 'Summer':
        months = np.arange(7, 10)
    elif szn == 'Fall':
        months = np.arange(10, 13)
    elif szn == 'All':
        months = np.arange(1, 13)
    else:
        print('Invalid value for szn:', szn)
        
    # Get subsetted latitude and longitude
    # Subsetting extracts a single time for each unique profile
    lat_subset, lon_subset = ios_get_sll(nclist[0], months, verbose)
    
    # Print accounting statistics:
    print('Printing counting statistics:')
    print('Min/max latitude bounds:', min(lat_subset), max(lat_subset))
    print('Min/max longitude bounds:', min(lon_subset), max(lon_subset))
    
    # Set up Lambert conformal map
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))
    
    # Use NASA's "Blue Marble" image
    # m.bluemarble()
    
    # Plot lat and lon data as red markers
    
    # Initialize figure
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    ax = plt.subplot(1, 1, 1)  # need in order to add points to plot iteratively?
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    
    # Plot the locations of the samples
    # REPEATED IF MULTIFILE
    x, y = m(lon_subset, lat_subset)
    # Plot on the subplot ax
    m.scatter(x, y, marker='o', color='r', s=0.5)
    
    # Add scatter points to plot from other files
    if multifile:
        for i in range(1, len(nclist)):
            lat_subset, lon_subset = ios_get_sll(nclist[i], months, verbose)
            # Plot the locations of the samples
            x, y = m(lon_subset, lat_subset)
            # Plot on the subplot ax
            m.scatter(x, y, marker='o', color='r', s=0.5)
            
    if verbose:
        print('Map and data plotted')
    
    plt.title('IOS {} {} 1991-2020 {}'.format(instrument, var, szn))
    
    png_name = output_folder + 'IOS_{}_{}_spatial_dist_{}.png'.format(instrument, var, szn)
    plt.savefig(png_name, dpi=400)
    
    # Close the figure so it can't be written to any more
    plt.close(fig)
    
    return png_name


def ios_profile_time_dist(ncdata, instrument, var='oxygen', multihist=False):
    # Barplot (?) of data over time
    # See if some seasons and/or years are underrepresented
    # Plot all unique profiles
    # ncdata: netCDF data object
    # instrument: 'BOT', or 'PCTD', or 'MCTD'
    # prof_indices: 1D numpy array whose elements are indices of the first
    #               occurrence of each unique profile in ncdata

    # Get list of unique profiles
    _, unique_indices = np.unique(ncdata.profile.data, return_index=True)
    # print(prof_indices[:10])

    time_subset = ncdata.time.data[unique_indices]

    # Now plot the unique profiles over time
    nbin30 = 30*4 #30 years times 4 seasons
    plt.hist(time_subset, bins=nbin30)
    plt.ylabel('Profile count')
    plt.title('IOS {} {} temporal distribution'.format(instrument, var))

    # Get start and end years represented in ncdata
    startyr = to_datetime(time_subset[0]).year
    endyr = to_datetime(time_subset[-1]).year

    plt.savefig('/home/hourstonh/Documents/climatology/data_explore/{}/IOS/'
                'IOS_{}_{}_time_dist_{}_{}.png'.format(var, instrument, var, startyr, endyr))
    plt.close()

    # Plot in 5-year increments to better determine seasonal variations
    # ncdata.profile.data are data of string type, so elements are of form "1991-002-0001"

    # Plot histograms for smaller subsets of years if desired
    if multihist:
        nbin5 = 5 * 4 * 2  # 5 years times 4 seasons times 2 for higher resolution

        for yr in range(1991, 2020, 5):
            subsetter = np.where((time_subset >= np.datetime64(str(yr))) & (time_subset < np.datetime64(str(yr + 5))))
            plt.hist(time_subset[subsetter], bins=nbin5)
            plt.ylabel('Profile count')
            plt.title('IOS {} {} temporal distribution'.format(instrument, var))
            plt.savefig('/home/hourstonh/Documents/climatology/data_explore/{}/'
                        'IOS_{}_{}_time_dist_{}_{}.png'.format(var, instrument, var, str(yr), str(yr+4)))
            plt.close()

    return


def stack_time(time_data):
    # time_data: 1D array of time data in pandas datetime format
    # Must build new array from scratch due to mutable operations TypeError
    # Initialize array
    stacked_time = np.empty(len(time_data), dtype='datetime64[ns]')

    # Populate the stacked time array
    for i in trange(len(time_data)):
        stacked_time[i] = time_data[i].replace(year=1996)  # 1996 is an arbitrary leap year

    # Convert to pandas datetime format
    stacked_time = to_datetime(stacked_time)
    print(stacked_time[:20])

    return stacked_time


def ios_prof_szn_dist(ncdata, instrument, var='oxygen'):
    # Plot temporal distribution by season (including all 30 years)

    # Get list of unique profiles
    _, unique_indices = np.unique(ncdata.profile.data, return_index=True)

    # Convert to pandas datetime object
    time_subset = to_datetime(ncdata.time.data[unique_indices])

    # Plot seasons:
    # Winter: Jan, Feb, Mar
    # Spring: Apr, May, Jun
    # Summer: Jul, Aug, Sep
    # Autumn: Oct, Nov, Dec

    # Bottle data is all in one file for the 30 years
    # Create a new array of time data without the year component
    # Or with the same year component, e.g. 2099-2100 ?
    # And only plot the month and day in that case

    # New year data with year = 2099 for all measurements
    # Convert to pandas datetime object first
    # Is there a faster way of doing this?

    month_day = stack_time(time_subset)

    # Now plot the data with new year

    nbin = 12*4 #12 seasons times 4 for higher resolution

    # Change x-axis tick labels so that they don't show the year
    # Just show month-day
    szns = ['Winter', 'Spring', 'Summer', 'Fall']

    for i in range(len(szns)):
        fig, ax = plt.subplots()

        # Calculate the month numbers for each specific season
        szn_months = 3 * i + np.array([1, 2, 3], dtype='int')

        subsetter = np.where((month_day.month >= szn_months[0]) & (month_day.month <= szn_months[2]))
        ax.hist(month_day[subsetter], bins=nbin)
        ax.set_ylabel('Profile count')
        ax.set_title('IOS {} {} {} temporal distribution 1991-2020'.format(instrument, var, szns[i]))

        # Fix the x-axis tick labeling
        month_day_fmt = mdates.DateFormatter('%b %d')  # "Locale's abbreviated month name. + day of the month"
        ax.xaxis.set_major_formatter(month_day_fmt)

        fig.savefig('/home/hourstonh/Documents/climatology/data_explore/{}/IOS/'
                    'IOS_{}_{}_time_dist_stacked_{}.png'.format(var, instrument, var, szns[i]))

        plt.close(fig)

    return


def ios_prof_scatter(time_data, out_dir, instrument, prof_indices, var='oxygen'):
    # Most recent temporal plotting function
    # time_data: array of timestamp data in numpy.datetime64 format

    # Scatter plot for all seasons first

    # Convert to pandas datetime object
    time_subset = to_datetime(time_data[prof_indices])

    # Count the number of profiles per each year
    years = np.arange(1991, 2021, 1)
    prof_year_counts = np.zeros(len(years), dtype='int64')

    # Populate the prof_year_counts array
    for i in range(len(years)):
        prof_year_counts[i] = len(time_subset[years[i] == time_subset.year])

    # Make a scatter plot covering all 30 years
    fig, ax = plt.subplots()
    ax.scatter(years, prof_year_counts)

    if instrument == 'BOT':
        ylab = 'bottles'
    elif instrument == 'PCTD':
        ylab = 'CTD profiles'
    else:
        ylab = 'profiles'

    ax.set_ylabel('Number of {}'.format(ylab))

    ax.set_title('IOS {} {} temporal distribution 1991-2020'.format(instrument, var))

    png_name = out_dir + 'IOS_{}_{}_time_dist_scatter.png'.format(var, instrument, instrument, var)
    fig.savefig(png_name)
    plt.close(fig)

    # Scatter plot by season (4 total)

    # Count the number of profiles for each season of each year
    szn_names = ['Winter', 'Spring', 'Fall', 'Winter']
    nszn = len(szn_names) # number of seasons
    nyr = 30 # number of years covered

    # Initialize 2D array of number of profiles per each season per each year
    prof_szn_counts = np.empty((nyr, nszn), dtype='int64')

    for y in range(len(years)):
        for s in range(nszn):
            szn_start = 3 * s + 1 # get number of start month s (Jan=1, Feb=2, ...)
            szn_end = 3 * s + 3 # get number of end month of season
            subsetter = np.where(
                (time_subset.year == years[y]) & (time_subset.month >= szn_start) & (time_subset.month <= szn_end))
            prof_szn_counts[y, s] = len(time_subset[subsetter])

    # Create a figure with 4 subplots
    fig = plt.figure(figsize=(7.2, 5.4))

    for s, n in zip(range(nszn), szn_names):
        ax = fig.add_subplot(2, 2, s+1)
        ax.scatter(years, prof_szn_counts[:, s])
        # Only put y-axis label on the LHS plots
        if s % 2 == 0:
            ax.set_ylabel('Number of {}'.format(ylab))
        ax.set_title(n)

    # Add space for subplot titles
    fig.subplots_adjust(hspace=0.3)

    # Set main figure title above all the subplots
    fig.suptitle('IOS {} {} temporal distribution'.format(instrument, var))

    png_name_szn = out_dir + 'IOS_{}_{}_time_scatter_byszn.png'.format(instrument, var)

    fig.savefig(png_name_szn)

    plt.close(fig)

    return [png_name, png_name_szn]


def ios_moor_time_dist(ncdata, instrument, prof_indices=None, var='oxygen', multihist=False):
    # Must be treated differently than the profile data

    # Plot seasons:
    # Winter: Jan, Feb, Mar
    # Spring: Apr, May, Jun
    # Summer: Jul, Aug, Sep
    # Autumn: Oct, Nov, Dec

    return


def ios_depth_scatter(profile_data, time_data, depth_data,
                      output_folder, instrument,
                      szn, var='oxygen', verbose=False):
    # USED THIS FUNCTION
    # Scatter plot of maximum profile depth vs time

    # Get list of unique profiles
    _, unique_indices = np.unique(profile_data, return_index=True)

    # Assign months to plot
    if szn == 'Winter':
        months = np.arange(1, 4)
    elif szn == 'Spring':
        months = np.arange(4, 7)
    elif szn == 'Summer':
        months = np.arange(7, 10)
    elif szn == 'Fall':
        months = np.arange(10, 13)
    elif szn == 'All':
        months = np.arange(1, 13)
    else:
        print('Invalid value for szn:', szn)

    # Convert to pandas datetime object for easier indexing by year, month
    time_subset = to_datetime(time_data[unique_indices])

    if verbose:
        print('Unique profile times extracted')

    # Extract the data from the selected season
    subsetter1 = np.where((time_subset.month >= months[0]) & (time_subset.month <= months[-1]))[0]
    time_subset_szn = time_subset[subsetter1]
    
    # Index the maximum depth of each profile
    # Include the very last depth entry: index == -1
    subsetter2 = np.concatenate((subsetter1[1:] - 1, np.array([-1])), axis=None)
    depth_subset_szn = depth_data[subsetter2]

    if verbose:
        print('Depth subsetted by input season')
    
    # Remove really high values of magnitude 1e36 from depth
    subsetter3 = np.where((depth_subset_szn != max(depth_subset_szn)))[0]
    time_subset_szn = time_subset_szn[subsetter3]
    depth_subset_szn = depth_subset_szn[subsetter3]
    
    # Make scatter plot
    plt.scatter(time_subset_szn, depth_subset_szn, s=0.5)
    plt.ylabel('Depth (m)')
    plt.title('IOS {} {} maximum profile depth: {}'.format(instrument, var, szn))

    # Invert the y axis (depth)
    plt.gca().invert_yaxis()

    png_name = output_folder + 'IOS_{}_{}_max_depths_{}.png'.format(instrument, var, szn)
    plt.savefig(png_name, dpi=400)
    plt.close()
    
    return png_name


def ios_runBOT():
    BOTfile = '/home/hourstonh/Documents/climatology/data/oxy_clim/IOS_CIOOS/' \
              'IOS_BOT_Profiles_Oxy_19910101_20200101.nc'

    # Output directory
    dest_dir = '/home/hourstonh/Documents/climatology/data_explore/oxygen/IOS/BOT/'

    data = xr.open_dataset(BOTfile)
    
    # ios_profile_time_dist(data, prof_indices=indices)

    # ios_prof_szn_dist(data, 'BOT', indices)

    # ios_prof_scatter(data.time.data, dest_dir, 'BOT', indices)

    szns = ['Winter', 'Spring', 'Summer', 'Fall']
    
    # for s in szns:
    #     ios_map_dist(data.profile.data, data.time.data, data.latitude.data, data.longitude.data,
    #                  dest_dir, 'BOT', left_lon=-160, right_lon=-102, bot_lat=25,
    #                  top_lat=62, szn=s, var='oxygen')

    fnames = []

    # for s in szns:
    #     png_name = ios_cmap_dist(data.profile.data, data.time.data, data.latitude.data,
    #                              data.longitude.data, data.depth.data,
    #                              dest_dir, 'BOT', left_lon=-160, right_lon=-102, bot_lat=25,
    #                              top_lat=62, szn=s, var='oxygen')
    #     fnames.append(png_name)
    
    for s in szns:
        png_name = ios_depth_scatter(data.profile.data, data.time.data, data.depth.data,
                                     dest_dir, 'BOT', s, 'oxygen', verbose=True)
        
        fnames.append(png_name)

    return fnames


def ios_runPCTD():
    CTDfiles = glob.glob('/home/hourstonh/Documents/climatology/data/oxy_clim/IOS_CIOOS/IOS_CTD_Profiles*',
                         recursive=False)
    CTDfiles.sort()

    # Output directory
    dest_dir = '/home/hourstonh/Documents/climatology/data_explore/oxygen/IOS/PCTD/'

    # Season names
    szns = ['Winter', 'Spring', 'Summer', 'Fall']

    # Initialize list of file names of output pngs
    png_names = []

    # Iterate through the seasons
    for s in szns:
        outname = ios_map_dist_v2(CTDfiles, dest_dir, 'PCTD', left_lon=-160,
                                  right_lon=-102, bot_lat=25, top_lat=62, szn=s,
                                  var='oxygen', verbose=True)
        png_names.append(outname)

    return png_names


def ios_extractPCTD():
    # DID NOT USE
    CTDfiles = glob.glob('/home/hourstonh/Documents/climatology/data/oxy_clim/IOS_CIOOS/IOS_CTD_Profiles*',
                         recursive=False)
    
    # Output directory
    dest_dir = '/home/hourstonh/Documents/climatology/data_explore/oxygen/IOS/PCTD/'
    
    # Make scatter plots of time vs number of profiles
    # Need to pass ALL the CTD files to the plotting function
    # Append all time and all profile data to lists
    # Profile data are of the form ['1991-001-0005', '1991-001-0006', ...]
    profile_list = []
    time_list = []
    lat_list = []
    lon_list = []
    
    for f in CTDfiles:
        data = xr.open_dataset(f)
        profile_list.append(list(data.profile.data))
        time_list.append(list(data.time.data))
        lat_list.append(list(data.latitude.data))
        lon_list.append(list(data.longitude.data))
        # Close dataset
        data.close()
    
    # Flatten the lists to 1D
    profile_array = np.array(list(chain(*profile_list)))
    time_array = np.array(list(chain(*time_list)))
    lat_array = np.array(list(chain(*lat_list)))
    lon_array = np.array(list(chain(*lon_list)))
    
    # Write the data to a csv file for convenience
    # colnames = ['Profile', 'Time', 'Lat', 'Lon']
    colnames = ['Profile', 'Time']
    
    ios_pctd_extracts = DataFrame(np.array([
        profile_array, time_array, lat_array, lon_array]).transpose(), columns=colnames)
    
    # Export to csv file
    extracts_name = dest_dir + 'IOS_PCTD_extracts_ProfTime.csv'
    ios_pctd_extracts.to_csv(extracts_name, index=False)
    return


def ios_runMCTD():
    CTDfiles = glob.glob('/home/hourstonh/Documents/climatology/data/oxy_clim/IOS_CIOOS/IOS_CTD_Moorings*',
                         recursive=False)

    for f in CTDfiles:
        data = xr.open_dataset(f)
        ios_moor_time_dist(data)

    data.close()

    return


# Plot the BOT files
ios_runBOT()

# Plot the CTD files
ios_runPCTD()


BOTfile = '/home/hourstonh/Documents/climatology/data/oxy_clim/IOS_CIOOS/' \
          'IOS_BOT_Profiles_Oxy_19910101_20200101.nc'


# Read in the netCDF file
data = xr.open_dataset(BOTfile)
# data = xr.open_dataset(PCTDfile)

# IOS: profile is profile data

# Histogram
# indices = get_unique_profiles(data)
# ios_time_dist(data, 'PCTD', indices)


# Close the netCDF object
# data.close()

