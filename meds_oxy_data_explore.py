"""
Data exploration for the MEDS extracts
Start with oxygen bottle data if available

Need to convert MEDS extracts to netCDF?

Variables of interest:
- MEDS_Sta (MEDS station?)
- Latitude, Longitude
- Cruise_ID_r (check for duplicates with IOS or WOD data)
- Obs_Year_r, Obs_Month_r, Obs_Day_r, Obs_Time_r
- Profile_Type(_r) (e.g., TEMP, PSAL, DOXY, NTRZ, SLCA, ...)
- No_Depths (Number of depths?)
- Level
- D/Press (Depth/Pressure)
- ProfParm (Profile parameter?)

How to find oxygen data by Profile_Type_r?
- Profile_Type_r in column L (==12)

How to find instrument type?
- Data_Type is instrument type or type of IGOSS radio message
    - BA: Bathythermograph
    - BO: Bottle
    - CD: CTD downcast
    - CU: CTD upcast
    - DT: Digital expendable bathythermograph
    - TE: Uncalibrated, low-res real time transmission CTD data
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pandas import to_datetime, read_csv, Series
import glob
from itertools import chain
import gsw

infile = '/home/hourstonh/Documents/climatology/data/MEDS_TSO/MEDS_ASCII_1991_2000.csv'

# Read in the CSV data as string type into a pandas dataframe
# prefix='C' specifies the prefix to each column name; names are 'C0', 'C1', ...
# Could use nrows= argument to specify the number of lines to read
# File is large: shape is (1048576, 28)
# skiprows=835506 because DOXY data doesn't start until there
mdat = read_csv(infile, header=None, prefix='C', dtype='str', skiprows=835506)

# Extract lat/lon data from mdat
bo_doxy_rows_ll = mdat.loc[:, (mdat['C11'] == 'Latitude')]

# Extract time data from mdat
bo_doxy_rows_t = mdat.loc[(mdat['C11'] == 'DOXY') & (mdat['C9'] == 'BO')]

bo_doxy_t = bo_doxy_rows_t.iloc[:, 0:15]

bo_doxy_t.head()

# MKey-general information:
# C3: Obs_Year // C4: Obs_Month // C5: Obs_Day // C6: Obs_Time //
# C12: Latitude // C13: Longitude

# Prof_Rec-specific information within MKeys:
# C5: Obs_Year_r // C6: Obs_Month_r // C7: Obs_Day_r // C8: Obs_Time_r (in HHMM?)

# C9: Data_Type_r (BO) // C11: Profile_Type_r (DOXY) // C13: No_Depths (# depth levels)

# Level, D/Press, ProfParm 2 rows below Prof_Rec row


def meds_map_dist(csvlist, output_folder, instrument, var, szn,
                  left_lon, bot_lat, right_lon, top_lat):
    # Create maps of the spatial distribution of the MEDS data
    # Plot seasons:
    # Winter: Jan, Feb, Mar
    # Spring: Apr, May, Jun
    # Summer: Jul, Aug, Sep
    # Autumn: Oct, Nov, Dec

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
        
    # Initialize lists for latitude and longitude data
    lat_list = []
    lon_list = []
    mth_list = []
    
    # Iterate through the csv files in the csvlist
    for csvfile in csvlist:
        # Extract the time data
        data = read_csv(csvfile, header=0)
        # Iterate through the whole dataset
        lat_list.append(list(data['Lat'].values))
        lon_list.append(list(data['Lon'].values))
        mth_list.append(list(data['Month'].values))

    # Flatten the lists
    lat_list = list(chain(*lat_list))
    lon_list = list(chain(*lon_list))
    mth_list = list(chain(*mth_list))
    
    # Subset the lat and lon data based on Month
    mth_array = np.array(mth_list, dtype='int')
    # Retain the first element in the np.where() tuple, which is the indexer
    subsetter = np.where((mth_array >= months[0]) & (mth_array <= months[-1]))[0]
    lat_subset = np.array(lat_list)[subsetter]
    lon_subset = np.array(lon_list)[subsetter] * (-1) # Convert direction of longitude increase ?????
    
    print(len(lat_subset))
    
    # Set up Lambert conformal map
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))

    # Plot lat and lon data as red markers

    # Initialize figure
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')

    # Plot the locations of the samples using small red circles
    x, y = m(lon_subset, lat_subset)
    m.scatter(x, y, marker='o', color='r', s=0.5)

    plt.title('MEDS {} {} 1991-2020 {}'.format(instrument, var, szn))

    png_name = output_folder + 'MEDS_{}_{}_spatial_dist_{}.png'.format(instrument, var, szn)
    plt.savefig(png_name, dpi=400)

    plt.close(fig)
    
    return png_name


def meds_time_scatter(csvlist, output_folder, instrument, var):
    # Create a scatter plot of number of profiles vs year for each season
    # Plot seasons:
    # Winter: Jan, Feb, Mar
    # Spring: Apr, May, Jun
    # Summer: Jul, Aug, Sep
    # Autumn: Oct, Nov, Dec
    
    # MEDS data are in subsets of decades (1991-2000, 2001-2010, 2011-2020)
    
    # Use extracted time and lat/lon csv data
    
    # Combine the time data into one pandas dataframe
    # Convert the time data with pandas to_datetime
    time_pd = []
    for csvfile in csvlist:
        # Extract the time data
        data = read_csv(csvfile, header=0)
        # Iterate through the whole dataset
        for i in range(len(data)):
            # YYYY MM DD HHMM
            timestamp = str(data.iloc[i, 0]) + '0{}'.format(data.iloc[i, 1])[-2:] + \
                        '0{}'.format(data.iloc[i, 2])[-2:] + '0{}'.format(data.iloc[i, 3])[-4:]
            time_pd.append(to_datetime(timestamp, format='%Y%m%d%H%M'))
          
    # Convert list to array
    time_pd = np.array(time_pd, dtype='datetime64[ns]')
    time_pd = to_datetime(time_pd)

    # Initialize seasons and years
    szn_names = ['Winter', 'Spring', 'Summer', 'Fall']
    nszn = len(szn_names)  # number of seasons
    years = np.arange(1991, 2021, 1)
    nyr = len(years)  # number of years covered

    # Count number of profiles per each season per each year
    # Initialize array to hold these counts
    prof_szn_counts = np.zeros((nyr, nszn), dtype='int64')
    
    # Populate the array of profile counts
    for y in range(nyr):
        for s in range(nszn):
            szn_start = 3 * s + 1  # get number of start month s (Jan=1, Feb=2, ...)
            szn_end = 3 * s + 3  # get number of end month of season
            subsetter = np.where(
                (time_pd.year == years[y]) & (time_pd.month >= szn_start) & (time_pd.month <= szn_end))
            prof_szn_counts[y, s] = len(time_pd[subsetter])
    
    # Begin plotting
    
    # Create figure
    fig = plt.figure()
    
    # Iterate through the seasons
    # Make one subplot for each season
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
    fig.suptitle('MEDS {} {} temporal distribution'.format(instrument, var))

    fname = output_folder + 'MEDS_{}_{}_time_byszn.png'.format(instrument, var)
    fig.savefig(fname)
    plt.close(fig)
    
    return fname


def meds_get_time_depth(df, szn):
    # Get correctly subsetted and formatted time and depth data
    
    # Calculate depth variable
    Depth_array = df['Depth/Press'].to_numpy(dtype='float')
    P_subsetter = df['D_P_code'].to_numpy(dtype='str') == 'P'
    Depth_array[P_subsetter] = gsw.z_from_p(Depth_array[P_subsetter],
                                            df['Lat'].to_numpy(dtype='float')[P_subsetter])
    df['Depth'] = Series(Depth_array)
    
    # Create pandas datetime data from time data
    # Ignore hours-minutes for now
    df['time_pandas'] = to_datetime(df[['Year', 'Month', 'Day']])
    
    # Get the first index of each unique profile
    _, unique_indices = np.unique(df['Num'], return_index=True)
    
    # Subset the time and depth data
    # Extract the deepest (last) measurement depth from each unique profile
    time_pd_subset = df['time_pandas'].iloc[unique_indices]
    depth_subset = df['Depth'].iloc[unique_indices]

    # Extract the desired season
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

    szn_subsetter = np.where(
        (time_pd_subset.dt.month >= months[0]) & (time_pd_subset.dt.month <= months[-1]))
    time_pd_subset = time_pd_subset.iloc[szn_subsetter]
    depth_subset = depth_subset.iloc[szn_subsetter]
    
    return time_pd_subset, depth_subset


def meds_depth_scatter(flist, output_folder, instrument,
                       szn, var='oxygen', multifile=False, verbose=False):
    
    # Read in the first file in flist
    # Read in as pandas dataframe
    df = read_csv(flist[0])

    # Get time in pandas datetime format
    # Get depth data from pressure with gsw
    # Get maximum profile depth vs time
    time_pd_subset, depth_subset = meds_get_time_depth(df, szn)
    
    if verbose:
        print('Time and depth subsetted by season and max depth')
    
    # Make scatter plot
    # Depth data is positive up, so take negative of it
    plt.scatter(time_pd_subset, -depth_subset, s=0.5, c='blue')
    plt.ylabel('Depth (m)')
    plt.title('MEDS {} {} maximum profile depth: {}'.format(instrument, var, szn))
    
    if multifile:
        for i in range(1, len(flist)):
            df = read_csv(flist[i])
            time_pd_subset, depth_subset = meds_get_time_depth(df, szn)
            plt.scatter(time_pd_subset, -depth_subset, s=0.5, c='blue')

    # Invert the y axis (depth)
    plt.gca().invert_yaxis()

    # Save the figure and close it
    pngname = output_folder + 'meds_{}_{}_max_depths_{}.png'.format(instrument, var, szn)
    plt.savefig(pngname, dpi=400)
    plt.close()
    
    return pngname


def meds_run_bo():
    fdir = '/home/hourstonh/Documents/climatology/data_explore/oxygen/MEDS/bo_profile_extracts/'
    flist = glob.glob(fdir + '*edr.csv')

    dest_dir = '/home/hourstonh/Documents/climatology/data_explore/MEDS/plots/'

    szns = ['Winter', 'Spring', 'Summer', 'Fall']
    
    for i in range(len(szns)):
        meds_depth_scatter(flist, dest_dir, 'BO', szns[i], 'oxygen', multifile=True,
                           verbose=True)
    
    return


meds_run_bo()


# Make season plots for time and spatial distribution
meds_fdir = '/home/hourstonh/Documents/climatology/data_explore/MEDS/'
csvfiles = glob.glob(meds_fdir + '*extracts.csv')
dest_dir = '/home/hourstonh/Documents/climatology/data_explore/MEDS/plots/'
timename = meds_time_scatter(csvfiles, dest_dir, 'BO', 'DOXY')

for elem in ['Winter', 'Spring', 'Summer', 'Fall']:
    mapname = meds_map_dist(csvfiles, dest_dir, 'BO', 'DOXY', elem,
                            left_lon=-160, right_lon=-102, bot_lat=25, top_lat=62)
    print(mapname)

# left_lon=-160, right_lon=-102
# left_lon=112, right_lon=160

# Import MEDS csv files
datadir = '/home/hourstonh/Documents/climatology/data/MEDS_TSO/'
datlist = ['MEDS_ASCII_1991_2000.csv', 'MEDS_ASCII_2001_2010.csv',
           'MEDS_ASCII_2011_2020.csv']

# Read the CSV files into pandas dataframes
# dflist = [read_csv(datlist[0], header=None, prefix='C', dtype='str', skiprows=835506)]


fdir = '/home/hourstonh/Documents/climatology/data_explore/oxygen/MEDS/bo_profile_extracts/'
flist = glob.glob(fdir + '*edr.csv')

df = read_csv(flist[0])
        

