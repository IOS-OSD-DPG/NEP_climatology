"""
Notes:
    https://stackoverflow.com/questions/57000903/what-is-the-fastest-and-most-efficient-way-to-append-rows-to-a-dataframe
    
    Script to retrieve Year, Month, Day, Time, Latitude, and Longitude data from MEDS csv files
"""
from os.path import basename
import numpy as np
from pandas import read_csv, DataFrame
from tqdm import trange
import glob


def meds_extract(df, outdir, instrument, var):
    # df: pandas dataframe
    # outdir: output directory
    # instrument: instrument code, e.g., "BO" for bottle
    # var: variable name, e.g., "DOXY" for dissolved oxygen
    # Extract the time, lat, lon, depth, and parameter value data to a new csv file?
    # Do depth later when I can think better
    
    # Check if desired variable is in the dataframe
    if instrument not in df.values:
        print('Instrument', instrument, 'not in dataframe. Quitting routine')
        return None
    elif var not in df.values:
        print('Variable', var, 'not in dataframe. Quitting routine')
        return None
    else:
        print('Instrument', instrument, 'and variable', var, 'are in dataframe. Continuing')
    
    # Initialize list of dictionaries to make into new dataframe
    dictionary_list = []
    
    # List the column indices of the names of important attributes
    # The value of these attributes is in the row below the name (ind + 1)
    ind_dtp = 7 #Data_Type column index
    ind_npr = 24 #No_Prof column index; number of variables profiled
    
    # Row indices: number of rows below the "MKey" row section header
    ind_ptp = 3 #Prof_Type; its values start three rows below the "MKey" row in the third column
    
    # Flag to indicate when to get start time for file name
    startflag = 0
    
    # Iterate through the rows in the dataframe
    for r in trange(len(df)):
        # Check for cruise header that begins each section of the MEDS csv file
        # And check that r is not the index of the last row in the dataframe
        if df.iloc[r, 0] == 'MKey' and r < len(df) - 1:
            # Get time
            y, m, d, t = [df.iloc[r + 1, 3], df.iloc[r + 1, 4], df.iloc[r + 1, 5], df.iloc[r + 1, 6]]
            # Get lat/lon
            lat, lon = [df.iloc[r + 1, 12], df.iloc[r + 1, 13]]
            
            # Get start date (YYYYMMDD) from df
            if startflag == 0:
                # Format month and day to be two characters long
                # (i.e., add leading zero if needed)
                startdate = str(y) + '0{}'.format(m)[-2:] + '0{}'.format(d)[-2:]
                startflag += 1
        
            # Get the number of variables profiled
            # Such variables include DOXY and TEMP; only want DOXY at this point
            No_Prof = int(df.iloc[r + 1, ind_npr])
            
            # Create boolean conditions to check for BO DOXY data
            instr_DType = df.iloc[r+1, ind_dtp] == instrument
            var_PType = var in df.iloc[r+ind_ptp:r+ind_ptp+No_Prof, 2].values
            
            # Check for BO data type and DOXY profile type
            if instr_DType and var_PType:
                # Add depth later; use No_Depths to index the data over depth
                dictionary_list.append({'Year':y, 'Month':m, 'Day':d, 'Time':t, 'Lat':lat, 'Lon':lon})
                
            # Get the last date (YYYYMMDD) in the file; it'll be replaced each iteration
            enddate = str(y) + '0{}'.format(m)[-2:] + '0{}'.format(d)[-2:]
                
    # Write list of dictionaries to a pandas dataframe
    tll_df = DataFrame.from_dict(dictionary_list)
    
    # Write the populated dataframe to a new csv file
    csvname = 'MEDS_{}_{}_{}_{}_extracts.csv'.format(startdate, enddate, instrument, var)
    outfname = outdir + csvname
    
    print(outfname)
    
    # Use index=False argument to remove the first column containing row indices
    tll_df.to_csv(outfname, index=False)
    
    return outfname


def meds_extract_v2(df, outdir, instrument, var):
    # Extract profile data and depths (and cruise ID?)
    # df: pandas dataframe
    # outdir: output directory
    # instrument: instrument code, e.g., "BO" for bottle
    # var: variable name, e.g., "DOXY" for dissolved oxygen
    # Extract the time, lat, lon, depth, and parameter value data to a new csv file?

    # Check if desired variable is in the dataframe
    # Doesn't guarantee if the variable measured by the specific instrument
    # is in the dataframe
    if instrument not in df.values:
        print('Instrument', instrument, 'not in dataframe. Quitting routine')
        return None
    elif var not in df.values:
        print('Variable', var, 'not in dataframe. Quitting routine')
        return None
    else:
        print('Instrument', instrument, 'and variable', var, 'are in dataframe. Continuing')

    # Initialize list of dictionaries to make into new dataframe
    dictionary_list = []

    # List the column indices of the names of important attributes
    # The value of these attributes is in the row below the name (ind + 1)
    ind_dtp = 7  # Data_Type column index
    ind_npr = 24  # No_Prof column index; number of variables profiled
    ind_ptr = 11 # Profile_Type_r; value is name of variable profiled (e.g., DOXY)

    # Row indices: number of rows below the "MKey" row section header
    ind_ptp = 3  # Prof_Type; its values start three rows below the "MKey" row in the third column

    # Flag to indicate when to get start time for file name
    startflag = 0
    startdate = 'YYYYMMDD'
    enddate = 'YYYYMMDD'
    
    # Iterate through the rows in the dataframe
    # Select data using the indices defined above
    for r in trange(len(df) - 1):
        # Check for cruise header that begins each section of the MEDS csv file
        # And check that the data are not IOS or NODC duplicates
        # And check that r is not the index of the last row in the dataframe
        duplicate_source = df.iloc[r + 1, 20] in ['IOS', 'NODC', 'WOD'] # boolean value
        
        if df.iloc[r, 0] == 'MKey' and not duplicate_source:
            # Get time
            y, m, d, t = [df.iloc[r + 1, 3], df.iloc[r + 1, 4], df.iloc[r + 1, 5], df.iloc[r + 1, 6]]
            # Get lat/lon
            lat, lon = [df.iloc[r + 1, 12], df.iloc[r + 1, 13]]
            # Get source_ID
            Source_ID = df.iloc[r + 1, 20]
        
            # Get start date (YYYYMMDD) from df
            if startflag == 0:
                # Format month and day to be two characters long
                # (i.e., add leading zero if needed)
                startdate = str(y) + '0{}'.format(m)[-2:] + '0{}'.format(d)[-2:]
                print(startdate)
                startflag += 1
        
            # Get the number of variables profiled (variables e.g., DOXY, TEMP, PSAL)
            No_Prof = int(df.iloc[r + 1, ind_npr])
        
            # Create boolean conditions to check for correct data type and profile type
            instr_DType = df.iloc[r + 1, ind_dtp] == instrument
            var_PType = var in df.iloc[r + ind_ptp:r + ind_ptp + No_Prof, 2].values
        
            # Check for correct data type and correct profile type
            if instr_DType and var_PType:
                # Just want to find the section containing the variable data over depth
                # Break for loop when this data is found
                for j in range(r, len(df)):
                    # Check that the dataframe element value is the variable requested by the user
                    # As the value for "Profile_Type_r"
                    if df.iloc[j, ind_ptr] == var:
                        Cruise_ID_r = df.iloc[j, ind_ptr - 7]
                        No_Depths = int(df.iloc[j, ind_ptr + 2])
                        D_P_code = df.iloc[j, ind_ptr + 3]
                        # Iterate through the number of depths
                        # ProfParm starts 2 rows below the variable header row
                        # Take min of number of rows since the list of depths
                        #  in the 1991-2000 file is cut off
                        for k in range(2, min((No_Depths + 2, len(df) - j))):
                            # Get the variable data
                            D_or_P = df.iloc[j + k, 1] # Depth or pressure, depending on code
                            DPq = df.iloc[j + k, 2] # Quality flag for depth or pressure
                            ProfParm = df.iloc[j + k, 3] # Profile Parameter: value of var at depth
                            PPq = df.iloc[j + k, 4] # Quality flag for depth or pressure
                            # Add to list of dictionaries
                            # RowNum is the row number in the original csv data file
                            dictionary_list.append({'RowNum': r, 'SourceID': Source_ID, 'CruiseID': Cruise_ID_r,
                                                    'Year': y, 'Month': m, 'Day': d, 'Time': t, 'Lat': lat,
                                                    'Lon': lon, 'Depth/Press': D_or_P, 'D_P_code': D_P_code,
                                                    'D_P_flag': DPq, 'ProfParm': ProfParm, 'PP_flag': PPq})
                        
                        # End the loop for this cruise and proceed to next cruise
                        break
        
                # Get the last date (YYYYMMDD) in the dataframe; it'll be replaced each iteration
                enddate = str(y) + '0{}'.format(m)[-2:] + '0{}'.format(d)[-2:]

    # Write list of dictionaries to a pandas dataframe
    tll_df = DataFrame.from_dict(dictionary_list)

    # Check if the dataframe is empty; if not export as csv file
    if len(tll_df.index) == 0:
        print('No matches found for {} and {}. Returning None'.format(instrument, var))
        return None
    else:
        # Write the populated dataframe to a new csv file
        csvname = 'MEDS_{}_{}_{}_{}_profiles_source.csv'.format(startdate, enddate, instrument, var)
        outfname = outdir + csvname
    
        print(outfname)

        # Use index=False argument to remove the first column containing row indices
        tll_df.to_csv(outfname, index=False)
    
        return outfname


# NEEDS FIXING
def meds_extract_TSO(df, outdir, instrument):
    # Extract profile data and depths (and cruise ID?)
    # df: pandas dataframe
    # outdir: output directory
    # instrument: instrument code, e.g., "BO" for bottle
    # var: variable name, e.g., "DOXY" for dissolved oxygen
    # Extract the time, lat, lon, depth, and parameter value data to a new csv file?

    # Check if desired variable is in the dataframe
    # Doesn't guarantee if the variable measured by the specific instrument
    # is in the dataframe
    if instrument not in df.values:
        print('Instrument', instrument, 'not in dataframe. Quitting routine')
        return None
    elif 'TEMP' in df.values and 'PSAL' in df.values and 'DOXY' in df.values:
        print('Instrument', instrument, 'and variables are in dataframe. Continuing')
    else:
        print('One or more variables not in dataframe. Quitting routine')
        return None

    # Initialize list of dictionaries to make into new dataframe
    dictionary_list = []

    # List the column indices of the names of important attributes
    # The value of these attributes is in the row below the name (ind + 1)
    ind_dtp = 7  # Data_Type column index
    ind_npr = 24  # No_Prof column index; number of variables profiled
    ind_ptr = 11  # Profile_Type_r; value is name of variable profiled (e.g., DOXY)

    # Row indices: number of rows below the "MKey" row section header
    ind_ptp = 3  # Prof_Type; its values start three rows below the "MKey" row in the third column

    # Flag to indicate when to get start time for file name
    startflag = 0
    startdate = 'YYYYMMDD'
    enddate = 'YYYYMMDD'

    # Iterate through the rows in the dataframe
    # Select data using the indices defined above
    for r in trange(len(df) - 1):
        # Check for cruise header that begins each section of the MEDS csv file
        # And check that the data are not IOS or NODC duplicates
        # And check that r is not the index of the last row in the dataframe
        duplicate_source = df.iloc[r + 1, 20] in ['IOS', 'NODC', 'WOD']  # boolean value

        if df.iloc[r, 0] == 'MKey' and not duplicate_source:
            # Get time
            y, m, d, t = [df.iloc[r + 1, 3], df.iloc[r + 1, 4], df.iloc[r + 1, 5], df.iloc[r + 1, 6]]
            # Get lat/lon
            lat, lon = [df.iloc[r + 1, 12], df.iloc[r + 1, 13]]
            # Get source_ID
            Source_ID = df.iloc[r + 1, 20]

            # Get start date (YYYYMMDD) from df
            if startflag == 0:
                # Format month and day to be two characters long
                # (i.e., add leading zero if needed)
                startdate = str(y) + '0{}'.format(m)[-2:] + '0{}'.format(d)[-2:]
                print(startdate)
                startflag += 1

            # Get the number of variables profiled (variables e.g., DOXY, TEMP, PSAL)
            No_Prof = int(df.iloc[r + 1, ind_npr])

            # Create boolean conditions to check for correct data type and profile type
            instr_DType = df.iloc[r + 1, ind_dtp] == instrument
            var_PType = var in df.iloc[r + ind_ptp:r + ind_ptp + No_Prof, 2].values

            # Check for correct data type and correct profile type
            if instr_DType and var_PType:
                # Just want to find the section containing the variable data over depth
                # Break for loop when this data is found
                for j in range(r, len(df)):
                    # Check that the dataframe element value is the variable requested by the user
                    # As the value for "Profile_Type_r"
                    if df.iloc[j, ind_ptr] == var:
                        Cruise_ID_r = df.iloc[j, ind_ptr - 7]
                        No_Depths = int(df.iloc[j, ind_ptr + 2])
                        D_P_code = df.iloc[j, ind_ptr + 3]
                        # Iterate through the number of depths
                        # ProfParm starts 2 rows below the variable header row
                        # Take min of number of rows since the list of depths
                        #  in the 1991-2000 file is cut off
                        for k in range(2, min((No_Depths + 2, len(df) - j))):
                            # Get the variable data
                            D_or_P = df.iloc[j + k, 1]  # Depth or pressure, depending on code
                            DPq = df.iloc[j + k, 2]  # Quality flag for depth or pressure
                            ProfParm = df.iloc[j + k, 3]  # Profile Parameter: value of var at depth
                            PPq = df.iloc[j + k, 4]  # Quality flag for depth or pressure
                            # Add to list of dictionaries
                            # RowNum is the row number in the original csv data file
                            dictionary_list.append({'RowNum': r, 'SourceID': Source_ID, 'CruiseID': Cruise_ID_r,
                                                    'Year': y, 'Month': m, 'Day': d, 'Time': t, 'Lat': lat,
                                                    'Lon': lon, 'Depth/Press': D_or_P, 'D_P_code': D_P_code,
                                                    'D_P_flag': DPq, 'ProfParm': ProfParm, 'PP_flag': PPq})

                        # End the loop for this cruise and proceed to next cruise
                        break

                # Get the last date (YYYYMMDD) in the dataframe; it'll be replaced each iteration
                enddate = str(y) + '0{}'.format(m)[-2:] + '0{}'.format(d)[-2:]

    # Write list of dictionaries to a pandas dataframe
    tll_df = DataFrame.from_dict(dictionary_list)

    # Check if the dataframe is empty; if not export as csv file
    if len(tll_df.index) == 0:
        print('No matches found for {} and {}. Returning None'.format(instrument, var))
        return None
    else:
        # Write the populated dataframe to a new csv file
        csvname = 'MEDS_{}_{}_{}_{}_profiles_source.csv'.format(startdate, enddate, instrument, var)
        outfname = outdir + csvname

        print(outfname)

        # Use index=False argument to remove the first column containing row indices
        tll_df.to_csv(outfname, index=False)

        return outfname


def meds_read_csv(fname, skiprows=None, chunksize=None, nrows=None):
    # If file is large (>1 GB), use chunksize for lazy reading
    # If using chunksize, this function returns an iterable object
    # instead of a pandas dataframe

    # Create a list of names for the 28 columns
    namelist = np.repeat('CXX', 28)
    for i in range(len(namelist)):
        namelist[i] = 'C' + str(i)
    
    if chunksize is None:
        df = read_csv(fname, header=None, dtype='str', skiprows=skiprows,
                      names=namelist, nrows=nrows)
        return df
    else:
        # Create iterable object of type TextFileReader
        iterob = read_csv(fname, header=None, dtype='str', skiprows=skiprows,
                          names=namelist, engine='python', chunksize=chunksize)
        return iterob


def meds_flag_check(csvname, outdir):
    # Remove data based on the qc flag it has
    # 1=good, 3=suspicious, 4=bad
    # Only keep flag=1=good
    # Export the good data in a new csv file
    
    # Read the csv file into a pandas dataframe
    df = read_csv(csvname)
    
    # Get the name of the input csv file
    csv_basename = basename(csvname)
    
    # Name the output csv file based on the input csv file
    csv_outname = outdir + csv_basename.replace('.', '_qc1.')
    
    # Clean the dataframe
    df_new = df.loc[df['PP_flag'] == 1]
    
    # Export the cleaned dataframe as a new csv file
    df_new.to_csv(csv_outname, index=False)
    
    return csv_outname


##### first extracts #####

# Define MEDS csv file name to use
# infile = '/home/hourstonh/Documents/climatology/data/MEDS_TSO/MEDS_ASCII_1991_2000.csv'
infile = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
         'meds\\MEDS_ASCII_1991_2000.csv'

# dest_dir = '/home/hourstonh/Documents/climatology/data_explore/MEDS/'
dest_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
           'meds_data_extracts\\bo_extracts\\'

# Instrument types to extract
itypes = ['BO', 'CD', 'XB']
# Data types to extract
dtypes = ['TEMP', 'PSAL']
# Output subfolders
out_subfolders = ['bo_extracts/', 'cd_extracts/', 'xb_extracts/']

# For 1991-2000 file
mdat = meds_read_csv(infile)

# Extract desired data type
# for i in range(1, len(dtypes)):
    # meds_extract_v2(mdat, dest_dir, 'TO', dtypes[i])
    
meds_extract_v2(mdat, dest_dir, 'BO', 'TEMP')
meds_extract_v2(mdat, dest_dir, 'BO', 'PSAL')

# For 2001-2010 file and 2011-2020 file (large)
infiles = ['/home/hourstonh/Documents/climatology/data/MEDS_TSO/MEDS_ASCII_2001_2010.csv',
           '/home/hourstonh/Documents/climatology/data/MEDS_TSO/MEDS_ASCII_2011_2020.csv']

extract_chunks = []

# Single variable single instrument
for f in infiles:
    print(basename(f))
    mdat = meds_read_csv(f, skiprows=None, chunksize=2000000)
    
    # Iterate through chunks; append each output csv name to list
    for chunk in mdat:
        extract_chunks.append(meds_extract_v2(chunk, dest_dir, 'BO', 'TEMP'))


# Multi-instrument, multi-variable
for f in infiles:
    print(basename(f))
    # Only BO, CD and XB available for oxygen
    for inst, subfolder in zip(itypes, out_subfolders):
        print(inst, subfolder)
        mdat = meds_read_csv(f, skiprows=None, chunksize=2000000)
    
        # Iterate through chunks; append each output csv name to list
        for chunk in mdat:
            extract_chunks.append(meds_extract_v2(chunk, dest_dir+subfolder, inst, 'TEMP'))
            extract_chunks.append(meds_extract_v2(chunk, dest_dir+subfolder, inst, 'PSAL'))
    

##### CHECK MEDS DATA SOURCEID'S #####

# extracts_path = '/home/hourstonh/Documents/climatology/data_explore/MEDS/bo_extracts/'
# flist = glob.glob(extracts_path + '*.csv')
#
# colnames = 'RowNum SourceID CruiseID Year Month Day Time Lat Lon Depth/Press D_P_code D_P_flag ProfParm PP_flag'.split()
# df_new = DataFrame(columns=colnames)
#
# # Initialize list of unique data sources that aren't IOS or NODC
# unique_sources = []
#
# for f in flist:
#     print(basename(f))
#     df = read_csv(f)
#
#     unique_sources += list(set(df['SourceID']))
#
#     # Write non-duplicates to a new csv file
#     for r in range(len(df)):
#         if df.iloc[r, 1] not in ['IOS', 'NODC']:
#             df_new = df_new.append(Series(df.iloc[r]))
#
# print(unique_sources)
#
# # Export to new csv file
# newname = extracts_path + 'MEDS_1991_2020_BO_DOXY_profiles_source_duprmv.csv'
# df_new.to_csv(newname, index=False)


##### check qc flags #####
indir = '/home/hourstonh/Documents/climatology/data_explore/MEDS/xb_extracts/'
infile = indir + 'MEDS_20020527_20060913_XB_TEMP_profiles_source.csv'

dat = read_csv(infile)

dat['PP_flag'] == 1

dat.loc[dat['PP_flag'] == 1]

indir = '/home/hourstonh/Documents/climatology/data_explore/MEDS/'
xb_files = glob.glob(indir + 'xb_extracts/*.csv')
cd_files = glob.glob(indir + 'cd_extracts/*.csv')
# bo_files = glob.glob(indir + 'bo_extracts/*.csv', recursive=False)

instrument_folders = ['xb_extracts/', 'cd_extracts/']
meds_files = [xb_files, cd_files]
# meds_files = [bo_files]
out_files = []

for i in range(len(meds_files)):
    for f in meds_files[i]:
        out_files.append(
            meds_flag_check(f, outdir=indir+instrument_folders[i]))

print(out_files)

bo_file = indir + 'bo_extracts/MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'
meds_flag_check(bo_file, outdir=indir+'bo_extracts/')
