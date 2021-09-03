"""Post-vertical interpolation checks

* Stats checks
    - Separate 5-standard-deviation checks for each season
    - Separate 5-standard-deviation checks for each standard level?
    - Use 5-degree squares
"""

import pandas as pd
import numpy as np
import glob
from os.path import basename, exists
from os import mkdir
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import csv
from tqdm import trange
from copy import deepcopy


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


def sl_std_check_basic(df_in):
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
    # Return array of standard levels from the standard levels text file

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


def sl_std_5deg_check(vvd_path, out_dir, szn):
    # Compute 5-degree square statistics in the Northeast Pacific Ocean

    # vvd_path = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
    #            'value_vs_depth\\10_replicate_check\\by_season\\' \
    #            'Oxy_1991_2020_value_vs_depth_rr_1_3.csv'

    vvd = pd.read_csv(vvd_path)

    # WOA standard deviation multiplier file
    sd_multiplier_file = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                         'lu_docs\\five_degrees_sd_multiplier_sept1.txt'

    sd_multiplier_df = pd.read_csv(sd_multiplier_file, skiprows=1)

    # Standard level file -- returns an array of the standard levels
    sl_arr = get_standard_levels()

    # # Index vvd by unique profile number
    # prof_start_ind = np.unique(vvd.Profile_number, return_index=True)[1]

    # Initialize column to hold stdev flag in vvd
    vvd['SD_flag'] = np.zeros(len(vvd), dtype='int32')

    # Initialize dataframes to hold square statistics for each square cell

    # Square number of values df
    sq_nval_df = sd_multiplier_df.copy(deep=True)

    # Replace multiplier values with zeros
    sq_nval_df.iloc[:, 2:] = 0

    # Replace the column names, eg. 'mult_0m' --> '0m'
    for colname in sq_nval_df.columns:
        sq_nval_df.rename(columns={colname: colname.replace('mult_', '')},
                          inplace=True)

    sq_mean1_df = sq_nval_df.copy(deep=True)

    sq_sd1_df = sq_nval_df.copy(deep=True)

    # Repeat for second run of stdev check
    sq_mean2_df = sq_nval_df.copy(deep=True)

    sq_sd2_df = sq_nval_df.copy(deep=True)

    # Create 5-degree square grid
    # xi = np.linspace(200, 245, 5)
    xi = np.arange(-160, -114, 5)  # lon
    yi = np.arange(30, 61, 5)  # lat
    # x_lon_r, y_lat_r = np.meshgrid(xi, yi)

    # Iterate through each cell in the 2D grid for each standard level
    # For each cell, calculate the number of profiles, the mean and stdev
    for i in trange(len(xi) - 1):
        for j in range(len(yi) - 1):
            # Find the correct sd multiplier in the dataframe
            # Start by finding the correct lat/lon row index
            # (lat/lon points are at the center of each square cell)
            subsetter_sd_multiplier = np.where(
                (sd_multiplier_df.Longitude >= xi[i]) &
                (sd_multiplier_df.Longitude <= xi[i + 1]) &
                (sd_multiplier_df.Latitude >= yi[j]) &
                (sd_multiplier_df.Latitude <= yi[j + 1]))[0]

            for k in range(len(sl_arr)):
                # # Get coordinates of square points
                # topleft = (x_lon_r[i], y_lat_r[j])
                # topright = (x_lon_r[i], y_lat_r[j + 1])
                # botright = (x_lon_r[i + 1], y_lat_r[j + 1])
                # botleft = (x_lon_r[i + 1], y_lat_r[j])

                # Index the level column name (e.g., 'mult_0m', 'mult_5m', ...)
                sd_multiplier_lvl = sd_multiplier_df.columns[k + 2]
                # Index the level column name for stats dfs (e.g., '0m', '5m', ...)
                stats_lvl = sq_nval_df.columns[k + 2]

                # Extract multiplier for the specific square and standard level
                # k=standard level index; +2 to skip first two cols: lat & lon
                # Convert from pd.Series int to scalar int
                sd_multiplier = int(sd_multiplier_df.loc[subsetter_sd_multiplier,
                                                         sd_multiplier_lvl])

                # Subset the vvd dataframe by level and lat/lon space
                # Use inclusive inequalities because double-counting
                # observations located on square borders is ok
                subsetter_vvd = np.where((vvd.SL_depth_m == sl_arr[k]) &
                                         (vvd.Longitude >= xi[i]) &
                                         (vvd.Longitude <= xi[i + 1]) &
                                         (vvd.Latitude >= yi[j]) &
                                         (vvd.Latitude <= yi[j + 1]))[0]

                # Subset the vvd by profile start indices and by subsetter
                vvd_subset = vvd.loc[subsetter_vvd]

                # Calculate the number of profiles in the selected 5-degree square
                # Output to a file at the end
                sq_num_of_profiles = len(vvd_subset)

                # Write number of profiles to df
                sq_nval_df.loc[subsetter_sd_multiplier, k + 2] = sq_num_of_profiles

                # Check if the square cell is void of observations
                if len(vvd_subset) == 0:
                    # print('Warning: Zero measurements in cell with',
                    #       float(sd_multiplier_df.loc[
                    #                 subsetter_sd_multiplier, 'Latitude']), 'latitude,',
                    #       float(sd_multiplier_df.loc[
                    #                 subsetter_sd_multiplier, 'Longitude']),
                    #       'longitude, and', sl_arr[k], 'm depth')

                    # Write -99s or NaNs to stats dataframes
                    sq_mean1_df.loc[subsetter_sd_multiplier, stats_lvl] = -99
                    sq_sd1_df.loc[subsetter_sd_multiplier, stats_lvl] = -99
                    sq_mean2_df.loc[subsetter_sd_multiplier, stats_lvl] = -99
                    sq_sd2_df.loc[subsetter_sd_multiplier, stats_lvl] = -99

                    # Skip to next iteration
                    continue

                # The rest assumes there is at least 1 standard level observation
                # in the 5-degree square

                # Calculate the mean and stdev of the values in the square
                sq_mean1 = np.mean(vvd_subset.SL_value)
                sq_sd1 = np.std(vvd_subset.SL_value)

                # Write these to the appropriate dfs
                sq_mean1_df.loc[subsetter_sd_multiplier, stats_lvl] = sq_mean1
                sq_sd1_df.loc[subsetter_sd_multiplier, stats_lvl] = sq_sd1

                # Find where standard level values are outside the range
                # (mean - multiplier * stdev, mean + multiplier * stdev)
                std_flag1_raise_where = np.where(
                    (vvd_subset.SL_value > sq_mean1 + sd_multiplier * sq_sd1) |
                    (vvd_subset.SL_value < sq_mean1 - sd_multiplier * sq_sd1))[0]

                # Add flag to vvd
                vvd.loc[std_flag1_raise_where, 'SD_flag'] = 1

                # Repeat flagging without first flagged values
                sq_mean2 = np.mean(vvd_subset.SL_value[vvd_subset.SD_flag == 0])
                sq_sd2 = np.std(vvd_subset.SL_value[vvd_subset.SD_flag == 0])

                # Write these to the appropriate dfs
                sq_mean2_df.loc[subsetter_sd_multiplier, stats_lvl] = sq_mean2
                sq_sd2_df.loc[subsetter_sd_multiplier, stats_lvl] = sq_sd2

                # Find where standard level values are outside sd range
                std_flag2_raise_where = np.where(
                    (vvd_subset.SL_value > sq_mean2 + sd_multiplier * sq_sd2) |
                    (vvd_subset.SL_value < sq_mean2 - sd_multiplier * sq_sd2))[0]

                # Flag as 1 or 2?
                vvd.loc[std_flag2_raise_where, 'SD_flag'] = 2

    out_subdir = out_dir + 'nprof_mean_sd\\'
    if not exists(out_subdir):
        mkdir(out_subdir)

    # Export square statistics dataframes to csv files
    sq_nval_df.to_csv(out_subdir + 'sl_num_val_{}.csv'.format(szn), index=False)
    sq_mean1_df.to_csv(out_subdir + 'sl_mean1_{}.csv'.format(szn), index=False)
    sq_sd1_df.to_csv(out_subdir + 'sl_sd1_{}.csv'.format(szn), index=False)
    sq_mean1_df.to_csv(out_subdir + 'sl_mean2_{}.csv'.format(szn), index=False)
    sq_sd1_df.to_csv(out_subdir + 'sl_sd2_{}.csv'.format(szn), index=False)

    vvd.to_csv(out_dir + basename(vvd_path).replace('.', '_sd.'), index=False)

    return vvd


in_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'value_vs_depth\\10_replicate_check\\by_season\\'

infiles = glob.glob(in_dir + '*.csv')

# Need the seasons in order
infiles.sort()

# Do separate calculations for each season
df = pd.read_csv(infiles[0])

# STANDARD DEVIATION CHECKS
path_list = sl_std_check_basic(df)

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
             'value_vs_depth\\11_stats_check\\'

for f in infiles:
    print(basename(f))

    # Get season names
    szn_abbrev = f[-7:-4]
    dfout = sl_std_5deg_check(f, output_dir, szn_abbrev)

    # See printout below -- I am concerned
    print(len(dfout))
    print(len(dfout.loc[dfout.SD_flag == 0]))
    print(len(dfout.loc[dfout.SD_flag == 1]))
    print(len(dfout.loc[dfout.SD_flag == 2]))
    print()

    # Remove values that failed the sd check
    dfout_drop = deepcopy(dfout.loc[dfout.SD_flag == 0])

    dfout_drop_name = basename(f).replace('.', '_sd_done.')

    dfout_drop.to_csv(output_dir + dfout_drop_name, index=False)

    # continue

"""Output:
Oxy_1991_2020_value_vs_depth_rr_AMJ.csv
100%|██████████| 9/9 [00:46<00:00,  5.17s/it]
270167
269529
1
637

Oxy_1991_2020_value_vs_depth_rr_JAS.csv
100%|██████████| 9/9 [00:54<00:00,  6.07s/it]
296690
296215
0 WHYYY
475

Oxy_1991_2020_value_vs_depth_rr_JFM.csv
100%|██████████| 9/9 [00:42<00:00,  4.69s/it]
153229
152784
1
444

Oxy_1991_2020_value_vs_depth_rr_OND.csv
100%|██████████| 9/9 [00:34<00:00,  3.86s/it]
141805
141407
0 WHYYY
398
"""