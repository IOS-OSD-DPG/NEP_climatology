""" Sept 7, 2021
Generate summary tables for NEP climatology
Organize by instrument, season/year

Do lat/lon check earlier??
"""

import pandas as pd
import numpy as np
import glob
from vvd_check_latlon import vvd_subset_latlon
from clim_helpers import date_string_to_datetime
from os.path import basename


def count_prof_by_szn(infiles):
    # Now compute summary statistics for original files
    # Initialize arrays to hold counts for each season
    bot_count = np.zeros(4, dtype='int32')
    ctd_count = np.zeros(4, dtype='int32')
    # total_count1 = bot_count1 + ctd_count1

    for f in infiles:
        print(basename(f))
        df = pd.read_csv(f)
        # Get the indices of the start of each profile
        prof_start_ind = np.unique(df.Profile_number, return_index=True)[1]

        # Convert Date_string to pandas datetime??
        # Create a new column for Date_string in pandas datetime format
        df = date_string_to_datetime(df)

        # Count the bottle and ctd data per season
        for i in range(4):
            szn_start = 3 * i + 1
            szn_end = 3 * i + 3
            bot_where1 = np.where((df.Instrument_type == 'BOT') &
                                  (df.Time_pd.dt.month >= szn_start) &
                                  (df.Time_pd.dt.month <= szn_end))[0]
            bot_count[i] += len(np.intersect1d(prof_start_ind, bot_where1))

            ctd_where1 = np.where((df.Instrument_type == 'CTD') &
                                  (df.Time_pd.dt.month >= szn_start) &
                                  (df.Time_pd.dt.month <= szn_end))[0]
            ctd_count[i] += len(np.intersect1d(prof_start_ind, ctd_where1))

    print(bot_count)
    print(ctd_count)
    
    return bot_count, ctd_count


def make_summary_table(file_list1, file_list2):
    # Make summary table of vvd table profiles
    
    bot_count1, ctd_count1 = count_prof_by_szn(file_list1)
    
    bot_count2, ctd_count2 = count_prof_by_szn(file_list2)
    
    bot_in_out = np.zeros(len(bot_count1) * 2 + 2)
    ctd_in_out = np.zeros(len(ctd_count1) * 2 + 2)
    
    for i in range(len(bot_count1)):
        bot_in_out[2 * i] = bot_count1[i]
    
    for i in range(len(bot_count2)):
        bot_in_out[2 * i + 1] = bot_count2[i]
    
    for i in range(len(ctd_count1)):
        ctd_in_out[2 * i] = ctd_count1[i]
    
    for i in range(len(bot_count2)):
        ctd_in_out[2 * i + 1] = ctd_count2[i]
    
    bot_in_out[-2] = sum(bot_count1)
    bot_in_out[-1] = sum(bot_count2)
    
    ctd_in_out[-2] = sum(ctd_count1)
    ctd_in_out[-1] = sum(ctd_count2)
    
    total_in_out = bot_in_out + ctd_in_out
    
    colnames = ['JFM_in', 'JFM_out', 'AMJ_in', 'AMJ_out', 'JAS_in', 'JAS_out',
                'OND_in', 'OND_out', 'Total_in', 'Total_out']
    
    df_sum = pd.DataFrame(data=np.array([bot_in_out, ctd_in_out, total_in_out]),
                          columns=colnames, dtype='int32')
    
    # rename rows in df
    df_sum.rename(index={df_sum.index[0]: 'num_BOT'}, inplace=True)
    df_sum.rename(index={df_sum.index[1]: 'num_CTD'}, inplace=True)
    df_sum.rename(index={df_sum.index[2]: 'num_total'}, inplace=True)
    
    # Add columns for MTH_%
    df_sum.insert(loc=2, column='JFM_%', value=[
        df_sum.loc['num_BOT', 'JFM_out']/df_sum.loc['num_BOT', 'JFM_in'],
        df_sum.loc['num_CTD', 'JFM_out']/df_sum.loc['num_CTD', 'JFM_in'],
        df_sum.loc['num_total', 'JFM_out']/df_sum.loc['num_total', 'JFM_in']])
    df_sum.insert(loc=5, column='AMJ_%', value=[
        df_sum.loc['num_BOT', 'AMJ_out']/df_sum.loc['num_BOT', 'AMJ_in'],
        df_sum.loc['num_CTD', 'AMJ_out']/df_sum.loc['num_CTD', 'AMJ_in'],
        df_sum.loc['num_total', 'AMJ_out']/df_sum.loc['num_total', 'AMJ_in']])
    df_sum.insert(loc=8, column='JAS_%', value=[
        df_sum.loc['num_BOT', 'JAS_out']/df_sum.loc['num_BOT', 'JAS_in'],
        df_sum.loc['num_CTD', 'JAS_out']/df_sum.loc['num_CTD', 'JAS_in'],
        df_sum.loc['num_total', 'JAS_out']/df_sum.loc['num_total', 'JAS_in']])
    df_sum.insert(loc=11, column='OND_%', value=[
        df_sum.loc['num_BOT', 'OND_out']/df_sum.loc['num_BOT', 'OND_in'],
        df_sum.loc['num_CTD', 'OND_out']/df_sum.loc['num_CTD', 'OND_in'],
        df_sum.loc['num_total', 'OND_out']/df_sum.loc['num_total', 'OND_in']])
    df_sum.insert(loc=14, column='Total_%', value=[
        df_sum.loc['num_BOT', 'Total_out'] / df_sum.loc['num_BOT', 'Total_in'],
        df_sum.loc['num_CTD', 'Total_out'] / df_sum.loc['num_CTD', 'Total_in'],
        df_sum.loc['num_total', 'Total_out'] / df_sum.loc['num_total', 'Total_in']])
    
    return df_sum


# Do for first and last
dir1 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
       'value_vs_depth\\1_original\\'

files1 = glob.glob(dir1 + '*Oxy*.csv')

dir8 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
       'value_vs_depth\\9_gradient_check\\'

file8 = dir8 + 'Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

# # Need to apply lat/lon check to file8 !!!!!!
# file8_ll = vvd_subset_latlon(file8, dir8 + 'latlon_check\\')

df_out = make_summary_table(files1, [file8])

# Export
df_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_explore\\' \
          'oxygen\\Oxy_summary_prof_count_table.csv'

# Want to keep index
df_out.to_csv(df_name, index=True)


# Do at each processing step
indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\'

subdirs = ['1_original', '3_filtered_for_duplicates', '4_latlon_check',
           '5_filtered_for_quality_flag', '6_filtered_for_nans', '7_depth_check',
           '8_range_check', '9_gradient_check']

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_explore\\' \
         'oxygen\\profile_counts\\'

for i in range(len(subdirs) - 1):
    # Check for files with "done" in the file name
    infiles1 = glob.glob(indir + subdirs[i] + '\\*Oxy*done.csv')
    infiles2 = glob.glob(indir + subdirs[i + 1] + '\\*Oxy*done.csv')
    
    if len(infiles1) == 0:
        infiles1 = glob.glob(indir + subdirs[i] + '\\*Oxy*.csv')
        
    if len(infiles2) == 0:
        infiles2 = glob.glob(indir + subdirs[i + 1] + '\\*Oxy*.csv')
    
    df_out = make_summary_table(infiles1, infiles2)
    
    step1 = subdirs[i][0]
    step2 = subdirs[i + 1][0]
    outname = outdir + 'Oxy_summary_prof_count_table{}{}.csv'.format(step1, step2)
    
    df_out.to_csv(outname, index=True)


