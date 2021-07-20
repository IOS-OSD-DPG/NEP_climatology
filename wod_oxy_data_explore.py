# Data exploration for the WOD extracts
# Start with oxygen bottle data: OSD type data

from xarray import open_dataset
import numpy as np
import matplotlib.pyplot as plt
from pandas import to_datetime
from mpl_toolkits.basemap import Basemap
from tqdm import trange
import glob


# OSD variables of interest: use odat.data_vars to print out all variable names
# lat, lon, time (in numpy.datetime64), Oxygen (in umol/kg), Oxygen_row_size_data

# Oxygen is a flattened (ragged?) array
# Oxygen_obs counts the number of observations (starts at 0)
# sum(Oxygen_row_size.data) == len(Oxygen)
# Number of unique profiles == len(Oxygen_row_size.data)


def wod_map_dist(ncdata, output_folder, instrument, left_lon, bot_lat, right_lon,
                 top_lat, szn, var='oxygen'):
    # Plot spatial distribution of data on a map using the Basemap package
    # See if some geographic regions are underrepresented
    # Use the Basemap package for creating maps
    
    # ncdata: netCDF file data that was read in with xarray.open_dataset
    # instrument: 'OSD' for bottle, others to be added later...
    # left_lon, bot_lat, right_lon, top_lat: corner coordinates for the
    #                                        Basemap map
    # szn: 'Winter', 'Spring', 'Summer', 'Fall', or 'All'

    lat_subset = ncdata.lat.data
    lon_subset = ncdata.lon.data
    
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
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')

    # Plot the locations of the samples
    x, y = m(lon_subset, lat_subset)
    m.scatter(x, y, marker='o', color='r', s=0.5)

    plt.title('WOD {} {} 1991-2020 {}'.format(instrument, var, szn))

    png_name = output_folder + 'WOD_{}_{}_spatial_dist_{}.png'.format(instrument, var, szn)
    plt.savefig(png_name, dpi=400)

    plt.close(fig)
    
    return png_name


def wod_time_scatter(nclist, instrument, var, output_folder):
    # nclist: list of netcdf file paths
    # Make scatter plot of time stamps of all unique profiles

    szn_names = ['Winter', 'Spring', 'Summer', 'Fall']
    nszn = len(szn_names)  # number of seasons
    years = np.arange(1991, 2021, 1)
    nyr = len(years)  # number of years covered
    
    # Combine all time data from all netCDF data that was read in
    alltime = []
    for i in range(len(nclist)):
        alltime.append(nclist[i].time.data)
    alltime = np.array(alltime)
    
    # Convert time.data from numpy.datetime64 to pandas datetime
    # Iterate through the 4 arrays in alltime
    for i in range(len(alltime)):
        alltime[i] = to_datetime(alltime[i])

    # Count number of profiles per each season per each year
    # Initialize array to hold these counts
    prof_szn_counts = np.zeros((nyr, nszn), dtype='int64')
    
    # Figure this out for one file first before tackling all 4
    for arr in range(len(alltime)):
        for y in range(nyr):
            for s in range(nszn):
                # Include data from all input netCDF files
                # time.data is times for all unique profiles
                szn_start = 3 * s + 1  # get number of start month s (Jan=1, Feb=2, ...)
                szn_end = 3 * s + 3  # get number of end month of season
                
                # Subset the time data and count the number of time stamps
                subsetter = np.where(
                    (alltime[arr].year == years[y]) & (
                            alltime[arr].month >= szn_start) & (
                            alltime[arr].month <= szn_end))
                prof_szn_counts[y, s] += len(alltime[arr][subsetter])

    # Create a figure with 4 subplots, one subplot for each season
    fig = plt.figure()

    for s in range(nszn):
        ax = fig.add_subplot(2, 2, s + 1)
        ax.scatter(years, prof_szn_counts[:, s])
        # Only put y-axis label on the LHS plots
        if s % 2 == 0:
            ax.set_ylabel('Number of profiles')
        ax.set_title(szn_names[s])

    # Add space for subplot titles
    fig.subplots_adjust(hspace=0.3)

    # Set main figure title above all the subplots
    fig.suptitle('WOD {} {} temporal distribution'.format(instrument, var))

    png_name = output_folder + 'WOD_{}_{}_time_scatter_byszn.png'.format(instrument, var)
    fig.savefig(png_name)

    plt.close(fig)

    return png_name


def wod_depth_scatter(row_size_data, time_data, depth_data, output_folder, instrument,
                      szn, var='oxygen', verbose=False):
    # USE THIS FUNCTION
    # Scatter plot of maximum profile depth vs time
    
    # Index the maximum depth of each profile
    # Use Oxygen_row_size to count number of measurements in each profile
    # And index the last (deepest) measurement in each profile
    # Get the indices of the deepest measurement from each profile
    # -1 accounts for Python starting indexing at 0, while len() method starts at 1
    max_depth_indices = np.cumsum(row_size_data, dtype='int') - 1
    depth_subset = depth_data[max_depth_indices]
    
    if verbose:
        print('Maximum depth per profile extracted')
    
    # Make scatter plot
    plt.scatter(time_data, depth_subset, s=0.5)
    plt.ylabel('Depth (m)')
    plt.title('WOD {} {} maximum profile depth: {}'.format(instrument, var, szn))
    if instrument == 'PFL':
        xmin = to_datetime('2004-01-01')
        xmax = to_datetime('2020-12-31')
        plt.xlim(xmin, xmax)
    elif instrument == 'CTD':
        xmin = to_datetime('1991-01-01')
        xmax = to_datetime('2020-12-31')
        plt.xlim(xmin, xmax)
    
    # Invert the y axis (depth)
    plt.gca().invert_yaxis()
    
    png_name = output_folder + 'WOD_{}_{}_max_depths_{}.png'.format(instrument, var, szn)
    plt.savefig(png_name, dpi=400)
    plt.close()
    
    return png_name


def wod_runOSD():
    indir = '/home/hourstonh/Documents/climatology/data/oxy_clim/WOD_extracts/Oxy_WOD_May2021_extracts/'
    outdir = '/home/hourstonh/Documents/climatology/data_explore/WOD/'
    
    osd = ['Oxy_1991_2020_JFM_OSD.nc', 'Oxy_1991_2020_AMJ_OSD.nc',
           'Oxy_1991_2020_JAS_OSD.nc', 'Oxy_1991_2020_OND_OSD.nc']

    szns = ['Winter', 'Spring', 'Summer', 'Fall']

    # nclist = []
    # for f in osd:
    #     nclist.append(xr.open_dataset(indir + f))

    # Make scatter plots of number of bottles vs time
    # wod_time_scatter(nclist, 'OSD', 'oxygen')

    # Comment the for loop out if not doing depth scatter
    for i in trange(len(osd)):
        data = open_dataset(indir + osd[i])
    
        # Make Basemap maps showing spatial distribution per season
        # wod_map_dist(data, outdir, 'OSD', left_lon=-160, right_lon=-102, bot_lat=25,
        #              top_lat=62, szn=szns[i], var='oxygen')
        
        # Scatter depth over time per season
        wod_depth_scatter(data.Oxygen_row_size.data, data.time.data, data.z.data,
                          outdir, 'OSD', szns[i], var='oxygen', verbose=True)
    
    return


wod_runOSD()


# Run NODC WOD PFL data
indir = '/home/hourstonh/Documents/climatology/data/oxy_clim/WOD_extracts/Oxy_WOD_May2021_extracts/'
outdir = '/home/hourstonh/Documents/climatology/data_explore/WOD/'
infiles = [indir + 'Oxy_1991_2020_JFM_PFL.nc',
           indir + 'Oxy_1991_2021_AMJ_PFL.nc',
           indir + 'Oxy_1991_2020_JAS_PFL.nc',
           indir + 'Oxy_1991_2020_OND_PFL.nc']

szns = ['Winter', 'Spring', 'Summer', 'Fall']

# Maps
for f, s in zip(infiles, szns):
    nc = open_dataset(f)
    wod_map_dist(nc, outdir, 'PFL', left_lon=-160, right_lon=-102, bot_lat=25,
                 top_lat=62, szn=s)

# Maximum depths
for f, s in zip(infiles, szns):
    nc = open_dataset(f)
    wod_depth_scatter(nc.Oxygen_row_size.data, nc.time.data, nc.z.data,
                      outdir, 'PFL', s, var='oxygen', verbose=True)

# Testing
# indir = '/home/hourstonh/Documents/climatology/oxy_clim/WOD_extracts/Oxy_WOD_May2021_extracts/'

# osd = [indir+'Oxy_1991_2020_JFM_OSD.nc', indir+'Oxy_1991_2020_AMJ_OSD.nc',
#        indir+'Oxy_1991_2020_JAS_OSD.nc', indir+'Oxy_1991_2020_OND_OSD.nc']

#osddat = [xr.open_dataset(osd[0]), xr.open_dataset(osd[1]), xr.open_dataset(osd[2]), xr.open_dataset(osd[3])]

# Array of arrays; can't be flattened
#alltime = np.array([osddat[0].time.data, osddat[1].time.data, osddat[2].time.data, osddat[3].time.data])

f = '/home/hourstonh/Documents/climatology/data/oxy_clim/WOD_extracts/Oxy_WOD_May2021_extracts/Oxy_1991_2020_JFM_OSD.nc'

dat = open_dataset(f)


### Explore Canadian non-IOS data from NODC ###
dest_dir = '/home/hourstonh/Documents/climatology/data_explore/WOD/CDN_nonIOS/'
in_dir = '/home/hourstonh/Documents/climatology/data/WOD_extracts/WOD_July_CDN_nonIOS_extracts/'

szns = ['Winter', 'Spring', 'Summer', 'Fall']

oxy_files = glob.glob(in_dir + 'Oxy*.nc', recursive=False)
temp_files = glob.glob(in_dir + 'Temp*GLD.nc', recursive=False)
sal_files = glob.glob(in_dir + 'Sal*GLD.nc', recursive=False)

# Sort the list by season
oxy_files = [oxy_files[3], oxy_files[1], oxy_files[0], oxy_files[2]]
temp_files_CTD = [temp_files[2], temp_files[3], temp_files[0], temp_files[1]]
sal_files_CTD = [sal_files[1], sal_files[0], sal_files[3], sal_files[2]]
temp_files_GLD = [None, temp_files[1], temp_files[2], temp_files[0]]
sal_files_GLD = [sal_files[2], sal_files[1], sal_files[0]]
# nc = open_dataset(oxy_files[0])

# Maps
for f, s in zip(sal_files_GLD, szns[1:]):
    nc = open_dataset(f)
    wod_map_dist(nc, dest_dir, 'GLD', left_lon=-162, right_lon=-115, bot_lat=25,
                 top_lat=62, szn=s, var='salinity')

# Maximum depths
# for f, s in zip(temp_files, szns):
#     nc = open_dataset(f)
#     wod_depth_scatter(nc.Oxygen_row_size.data, nc.time.data, nc.z.data,
#                       dest_dir, 'CTD', s, var='oxygen', verbose=True)

# for f, s in zip(temp_files, szns):
#     nc = open_dataset(f)
#     wod_depth_scatter(nc.Temperature_row_size.data, nc.time.data, nc.z.data,
#                       dest_dir, 'CTD', s, var='Sal', verbose=True)

for f, s in zip(sal_files_GLD, szns[1:]):
    nc = open_dataset(f)
    wod_depth_scatter(nc.Salinity_row_size.data, nc.time.data, nc.z.data,
                      dest_dir, 'GLD', s, var='salinity', verbose=True)

# Make list of netCDF objects
# temp_data = [open_dataset(temp_files[0]), open_dataset(temp_files[1]),
#              open_dataset(temp_files[2]), open_dataset(temp_files[3])]

sal_data = [open_dataset(sal_files[0]), open_dataset(sal_files[1]),
            open_dataset(sal_files[2]), open_dataset(sal_files[3])]
wod_time_scatter(sal_data, 'CTD', 'salinity', dest_dir)

sal_data_GLD = [open_dataset(sal_files_GLD[0]),
                open_dataset(sal_files_GLD[1]),
                open_dataset(sal_files_GLD[2])]
wod_time_scatter(sal_data_GLD, 'GLD', 'salinity', dest_dir)

