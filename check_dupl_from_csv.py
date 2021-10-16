# USED
# Check for duplicates in data for climatology

import pandas as pd
import numpy as np
from tqdm import trange
from copy import deepcopy


# # Now with the one big csv file
# infile = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
#          'ALL_Profiles_Oxy_1991_2020.csv'

infile = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'profile_data_tables\\Argo\\NODC_noCAD_PFL_Profiles_Oxy_1991_2020.csv'

df_all = pd.read_csv(infile)

df_all.drop(columns=['Unnamed: 0'], inplace=True)

###############################################
# Check for NaN values
nan_ind = np.where(pd.isna(df_all.Date_string))
np.where(pd.isna(df_all))
# These are equal, so only these rows contain NaNs

print(df_all.loc[nan_ind, ('Latitude', 'Longitude')])

# Check MEDS file for NaN times
fmeds = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\meds_data_extracts\\' \
        'bo_extracts\\MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'
dmeds = pd.read_csv(fmeds)

dmeds['Hour'] = dmeds.Time.astype(str).apply(lambda x: ('000' + x)[-4:][:-2])
dmeds['Minute'] = dmeds.Time.astype(str).apply(lambda x: ('000' + x)[-4:][-2:])

dmeds['Timestring'] = pd.to_datetime(
    dmeds[['Year', 'Month', 'Day', 'Hour', 'Minute']]).dt.strftime(
    '%Y%m%d%H%M%S')

############################################
# Find duplicates

# Exact duplicates
df_all['Exact_duplicate_row'] = df_all.duplicated(
    subset=['Instrument_type', 'Date_string', 'Latitude', 'Longitude'])

# Accounting statistics
print(len(df_all['Exact_duplicate_row'].iloc[(df_all['Exact_duplicate_row'] == True).values]))
print(len(df_all['Exact_duplicate_row'].iloc[(df_all['Exact_duplicate_row'] == False).values]))

# edf: Exact duplicates flagged
# edf_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
#            'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_edf.csv'

edf_name = infile.replace('.csv', '_edf.csv')

df_all.to_csv(edf_name)

# How to specify to keep CTD data over BOT data regardless of order in the df?
# Sort df by instrument type before running duplicated(); sort reverse order to
# have CTD before BOT
# Check for exact duplicates between CTD and bottle data
df_all['CTD_BOT_duplicate_row'] = df_all.sort_values(
    by=['Instrument_type'], ascending=False).duplicated(
    subset=['Date_string', 'Latitude', 'Longitude'])

# Exclude exact duplicates from previous step by subsetting by the negation of
# Exact_duplicate_row
# [mask, column_name]
df_all.loc[df_all['Exact_duplicate_row'].values, 'CTD_BOT_duplicate_row'] = False

# Accounting statistics
print(len(df_all))
print(len(df_all.iloc[df_all['Exact_duplicate_row'].values]))
print(len(df_all.iloc[df_all['CTD_BOT_duplicate_row'].values]))

# CTD-BOT exact duplicates flagged
cb_edf_name = edf_name.replace('edf', 'cb_edf')
df_all.to_csv(cb_edf_name, index=False)

############################################
# For speeding up inexact duplicate checking


def pdt_inexact_dupl(df):
    # "|" represents the inclusive "or"
    subsetter = (
        ~(df['Exact_duplicate_row'] | df['CTD_BOT_duplicate_row'])).values

    # How many rows subsetter removes
    print(len(subsetter[subsetter]), len(subsetter[~subsetter]))

    # deepcopy to help avoid memory problems
    df_copy = deepcopy(df.loc[subsetter, :])

    # Convert time to pandas datetime format
    df_copy['Time_pd'] = pd.to_datetime(df_copy.loc[:, 'Date_string'],
                                        format='%Y%m%d%H%M%S')

    # Initialize column for inexact duplicates
    df_copy['Inexact_duplicate_row'] = np.repeat(False, len(df_copy))

    # Get array of indices of dataframe, since they are not linearly spaced
    # e.g., 1,2,3,5,6,7,11,...
    ind = df_copy.index.values

    # Set ranges (limits) for inexact duplicate checking
    latlon_lim = 0.01  # decimal degrees
    t_lim = pd.Timedelta(1, unit='h')  # hours

    # Iterate through dataframe to check validity of inexact duplicate checking
    # Partner index is the index of the row that other row(s) are flagged as a
    # duplicate of
    df_copy['Partner_index'] = np.repeat(-1, len(df_copy))
    for i in trange(len(df_copy)):
        # Create masks to check for values in between selected ranges
        mask_lat = df_copy.loc[:, 'Latitude'].between(
            df_copy.loc[ind[i], 'Latitude'] - latlon_lim,
            df_copy.loc[ind[i], 'Latitude'] + latlon_lim,
            inclusive=True)
        mask_lon = df_copy.loc[:, 'Longitude'].between(
            df_copy.loc[ind[i], 'Longitude'] - latlon_lim,
            df_copy.loc[ind[i], 'Longitude'] + latlon_lim,
            inclusive=True)
        mask_time = df_copy.loc[:, 'Time_pd'].between(
            df_copy.loc[ind[i], 'Time_pd'] - t_lim,
            df_copy.loc[ind[i], 'Time_pd'] + t_lim,
            inclusive=True)

        # Mask to check for same instrument
        instrument_type = df_copy.loc[ind[i], 'Instrument_type']
        mask_inst = df_copy.loc[:, 'Instrument_type'] == instrument_type

        # Perform intersection of masks (set 'and')
        mask_llt = mask_lat & mask_lon & mask_time & mask_inst

        # Exclude the first True occurrence and flag its inexact duplicates
        # Make sure that the change "sticks"
        # Find index of first occurrence of "True"
        # IndexError: index 0 is out of bounds for axis 0 with size 0 for row 50165
        # Need to search for rows that have Time_pd == NaT (b/c of Date_string == NaN)

        # print(ind[i], len(mask_llt.loc[(mask_llt == True).values]))

        if len(mask_llt.loc[(mask_llt == True).values]) > 1:
            # Note down partner index?
            first_true_ind = mask_llt.loc[mask_llt == True].index[0]
            remainder_true_ind = mask_llt.loc[mask_llt == True].index[1:]
            df_copy.loc[remainder_true_ind, 'Partner_index'] = first_true_ind
            # Union intersect (set inclusive "or") with the "all" mask
            df_copy.Inexact_duplicate_row = df_copy.Inexact_duplicate_row | mask_llt
        else:
            # Set non-duplicate row flag to False
            # Don't need to intersect with the Inexact_duplicate_row column
            # because there's nothing new to add
            first_true_ind = mask_llt.loc[mask_llt == True].index[0]
            mask_llt.loc[first_true_ind] = False

    # Accounting statistics
    print(len(df_copy.Inexact_duplicate_row))
    print(len(df_copy.Inexact_duplicate_row.iloc[df_copy.Inexact_duplicate_row.values]))
    # Print the number of non-first occurrences of nonexact duplicates
    print(len(df_copy.iloc[(df_copy.Partner_index != -1).values]))

    # Remove column
    df_copy = df_copy.drop(columns='Time_pd')

    # Intersect the ie_subset dataframe with the rows that weren't part of the subset
    # Get inverse of subsetter
    df_copy_inv = df[~subsetter]

    # Add columns to df_copy_inv that are in df_copy
    df_copy_inv.insert(len(df_copy_inv.columns), 'Inexact_duplicate_row',
                       np.repeat(False, len(df_copy_inv)))
    df_copy_inv.insert(len(df_copy_inv.columns), 'Partner_index',
                       np.repeat(-1, len(df_copy_inv)))

    df_all_out = pd.concat([df_copy, df_copy_inv])

    return df_all_out


# # ie_subs stands for inexact subset
# # pi stands for PARTNER INDEX (all duplicate rows are flagged including first occurrence
# # Will need to intersect these rows and the rows of the *cb_edr.csv dataframe
# df_copy_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_extracts\\' \
#                'duplicates_flagged\\ALL_Profiles_Oxy_1991_2020_ie_subs_001ll_pi.csv'
#
# df_out = pdt_inexact_dupl(df_all)
#
# df_out.to_csv(df_copy_name)

# cb_edf_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
#               'profile_data_tables\\duplicates_flagged\\' \
#               'ALL_Profiles_Oxy_1991_2020_cb_edf.csv'

df_all = pd.read_csv(cb_edf_name)

df_out = pdt_inexact_dupl(df_all)

which = 'PFL'  # ALL

df_all_out_name = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                  'duplicates_flagged\\' \
                  '{}_Profiles_Oxy_1991_2020_ie_001ll_pi.csv'.format(which)

df_out.to_csv(df_all_out_name)

##################################
