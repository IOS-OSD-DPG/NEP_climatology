"""Post-vertical interpolation checks

* Stats checks
    - Separate 5-standard-deviation checks for each season
    - Separate 5-standard-deviation checks for each standard level?
    - Use 5-degree squares
"""

import pandas as pd
import numpy as np
import glob
from os.path import basename
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import csv


# GRID SNEAK PEAK


def map_5deg_grid():
    # gradient check done file
    vvd_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
               'value_vs_depth\\10_replicate_check\\' \
               'Oxy_1991_2020_value_vs_depth_rr_rep_val_check.csv'

    vvd_df = pd.read_csv(vvd_file)

    # Create 5-degree square grid
    # xi = np.linspace(200, 245, 5)
    xi = np.arange(-160, -114, 5)
    yi = np.arange(30, 61, 5)
    x_lon_r, y_lat_r = np.meshgrid(xi, yi)

    left_lon = -162.
    bot_lat = 22.
    right_lon = -100.
    top_lat = 62.

    # Set up Lambert conformal map
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat, projection='lcc',
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))

    # Initialize figure
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    # ax = plt.subplot(1, 1, 1) # need in order to add points to plot iteratively?
    m.drawcoastlines(linewidth=0.2)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')

    # Plot the locations of the profiles
    profile_subsetter = np.unique(vvd_df.Profile_number, return_index=True)[1]

    lon_subset = np.array(vvd_df.loc[profile_subsetter, 'Longitude'])
    lat_subset = np.array(vvd_df.loc[profile_subsetter, 'Latitude'])

    xobs, yobs = m(lon_subset, lat_subset)
    m.scatter(xobs, yobs, marker='o', color='r', s=0.5)

    # Plot the locations of the grid nodes
    x, y = m(x_lon_r, y_lat_r)
    # Plot on the subplot ax
    m.plot(x, y, marker='*', color='b')

    plt.title('10_Oxy 5-degree grid')

    dest_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
               'value_vs_depth\\10_replicate_check\\'
    png_name = 'Oxy_1991_2020_5deg_grid_60.png'

    plt.savefig(dest_dir + png_name, dpi=400)

    plt.close(fig)

    return dest_dir + png_name


# STANDARD LEVEL STANDARD DEVIATION CHECK


def sl_std_check(df_in):
    # Initialize column for standard deviation flags
    df_in['STD_flag'] = np.zeros(len(df_in), dtype='int32')

    # Calculate the standard deviation of all values above and including 50m
    subsetter_le_50m = np.where(df_in.SL_depth_m <= 50)[0]

    mean_le_50m = np.mean(df_in.loc[subsetter_le_50m, 'SL_value'])
    std_le_50m = np.std(df_in.loc[subsetter_le_50m, 'SL_value'])

    # Calcuate the standard deviation of all values below 50m
    subsetter_gt_50m = np.where(df_in.SL_depth_m > 50)[0]

    mean_gt_50m = np.mean(df_in.loc[subsetter_gt_50m, 'SL_value'])
    std_gt_50m = np.std(df_in.loc[subsetter_gt_50m, 'SL_value'])

    std_flag_loc = np.where(
        (df_in.loc[subsetter_le_50m, 'SL_value'] > mean_le_50m + 5 * std_le_50m) |
        (df_in.loc[subsetter_gt_50m, 'SL_value'] > mean_gt_50m + 5 * std_gt_50m))[0]

    df_in.loc[std_flag_loc, 'STD_flag'] = 1

    # Repeat std check once on the data that haven't already been flagged

    # Calculate the standard deviation of all values above and including 50m
    subsetter_le_50m_2 = np.where((df_in.STD_check == 0) &
                                  (df_in.SL_depth_m <= 50))[0]

    mean_le_50m_2 = np.mean(df_in.loc[subsetter_le_50m_2, 'SL_value'])
    std_le_50m_2 = np.std(df_in.loc[subsetter_le_50m_2, 'SL_value'])

    # Calcuate the standard deviation of all values below 50m
    subsetter_gt_50m_2 = np.where((df_in.STD_check == 0) &
                                  (df_in.SL_depth_m > 50))[0]

    mean_gt_50m_2 = np.mean(df_in.loc[subsetter_gt_50m_2, 'SL_value'])
    std_gt_50m_2 = np.std(df_in.loc[subsetter_gt_50m_2, 'SL_value'])

    std_flag_loc_2 = np.where(
        (df_in.loc[subsetter_le_50m_2, 'SL_value'] > mean_le_50m_2 + 5 * std_le_50m_2) |
        (df_in.loc[subsetter_gt_50m_2, 'SL_value'] > mean_gt_50m_2 + 5 * std_gt_50m_2))[0]

    df_in.loc[std_flag_loc_2, 'STD_flag'] = 2

    # Export the updated df
    outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\9_vertical_interp\\by_season\\10_vertical_interp_check\\'

    # Shorten file name...
    outname = basename(infiles[0]).replace('_value_vs_depth', '')
    outname = outname.replace('.csv', 'std_check.csv')

    df_in.to_csv(outdir + outname, index=False)

    # Export the updated df with the flagged values removed

    # Remove flagged data
    df_out = df_in.loc[df_in['STD_flag'] == 0]

    # Drop the column containing the flags
    df_out.drop(columns='STD_flag', inplace=True)

    outname2 = outname.replace('.csv', '_done.csv')

    df_out.to_csv(outdir + outname2, index=False)

    return outdir + outname, outdir + outname2


def get_standard_levels():
    # Import standard levels file
    file_sl = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\lu_docs\\' \
              'WOA_Standard_Depths.txt'

    # Initialize list with each element being a row in file_sl
    sl_list = []
    with open(file_sl, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            sl_list += row

    # Remove empty elements: '' and ' '
    # Gotta love list comprehension
    sl_list_v2 = [int(x.strip(' ')) for x in sl_list if x not in ['', ' ']]

    # Convert list to array
    sl_arr = np.array(sl_list_v2)
    return sl_arr


def sl_std_5deg_check(vvd):
    # WOA standard deviation multiplier file
    sd_multiplier_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                         'lu_docs\\five_degrees_sd_multiplier_sept1.txt'

    sd_multiplier_df = pd.read_csv(sd_multiplier_file, skiprows=1)

    # Standard level file -- returns an array of the standard levels
    sl_arr = get_standard_levels()

    # # Index vvd by unique profile number
    # prof_start_ind = np.unique(vvd.Profile_number, return_index=True)[1]

    # Initialize column to hold stdev flag in vvd
    vvd['SD_flag'] = np.zeros(len(vvd))

    # Initialize dataframes to hold square statistics for each point
    square_nprof_df = sd_multiplier_df.copy(deep=True)
    square_nprof_df.iloc[:, 3:] = 0

    square_mean1_df = square_nprof_df.copy(deep=True)

    square_sd1_df = square_nprof_df.copy(deep=True)

    square_mean2_df = square_nprof_df.copy(deep=True)

    square_sd2_df = square_nprof_df.copy(deep=True)

    # Create 5-degree square grid
    # xi = np.linspace(200, 245, 5)
    xi = np.arange(-160, -114, 5)
    yi = np.arange(30, 61, 5)
    x_lon_r, y_lat_r = np.meshgrid(xi, yi)

    # Iterate through each cell in the 2D grid for each standard level
    # For each cell, calculate the number of profiles, the mean and stdev
    for k in range(len(sl_arr)):
        for i in range(len(x_lon_r) - 1):
            for j in range(len(x_lon_r[i]) - 1):
                # # Get coordinates of square points
                # topleft = (x_lon_r[i], y_lat_r[j])
                # topright = (x_lon_r[i], y_lat_r[j + 1])
                # botright = (x_lon_r[i + 1], y_lat_r[j + 1])
                # botleft = (x_lon_r[i + 1], y_lat_r[j])

                # Subset the vvd dataframe by level and lat/lon space
                subsetter = np.where((vvd.SL_depth_m == sl_arr[k]) &
                                     (vvd.Longitude >= x_lon_r[i]) &
                                     (vvd.Longitude <= x_lon_r[i + 1]) &
                                     (vvd.Latitude >= y_lat_r[j]) &
                                     (vvd.Latitude <= y_lat_r[j + 1]))[0]

                # Subset the vvd by profile start indices and by subsetter
                vvd_subset = vvd.loc[subsetter]

                # Calculate the number of profiles in the selected 5-degree square
                # Output to a file at the end
                sq_num_of_profiles = len(vvd_subset)

                # Calculate the mean and stdev of the values in the square
                sq_mean1 = np.mean(vvd_subset.SL_value)
                sq_sd1 = np.std(vvd_subset.SL_value)

                # Apply the sd multiplier (points are at the center of each square)
                sd_multiplier_where = np.where(
                    (sd_multiplier_df.Longitude >= x_lon_r[i]) &
                    (sd_multiplier_df.Longitude <= x_lon_r[i + 1]) &
                    (sd_multiplier_df.Latitude >= y_lat_r[j]) &
                    (sd_multiplier_df.Latitude <= y_lat_r[j + 1]))[0]

                # Extract multiplier for the specific square and standard level
                sd_multiplier = sd_multiplier_df.iloc[sd_multiplier_where, k]

                std_flag1_raise = np.where(
                    (vvd_subset.SL_value > sq_mean1 + sd_multiplier * sq_sd1) |
                    (vvd_subset.SL_value < sq_mean1 - sd_multiplier * sq_sd1))[0]

                # Add flag to vvd
                vvd.loc[std_flag1_raise, 'SD_flag'] = 1

                # Repeat flagging without first flagged values
                flag_subsetter = ''
                sq_mean2 = np.mean(vvd_subset.SL_value)
                sq_sd2 = np.std(vvd_subset.SL_value)

    return vvd


in_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\10_replicate_check\\by_season\\'

infiles = glob.glob(in_dir + '*.csv')

# Need the seasons in order
infiles.sort()

# Do separate calculations for each season
df = pd.read_csv(infiles[0])

# STANDARD DEVIATION CHECKS
path_list = sl_std_check(df)


