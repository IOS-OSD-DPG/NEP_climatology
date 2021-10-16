# Export all oxygen climatology data to common csv format: Profile data table
# Sources: IOS, NODC, MEDS

from xarray import open_dataset
import pandas as pd
import numpy as np
import glob
from os.path import basename


def ios_to_pdt(nclist, var):
    if var == 'Oxy':
        varcode = 'DOXMZZ01'
    elif var == 'Temp':
        varcode = 'TEMPS901'
    elif var == 'Sal':
        varcode = 'PSALST01'

    df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
               "Instrument_type", "Date_string", "Latitude",
               "Longitude", "Quality_control_flag"]

    ios_df = pd.DataFrame(columns=df_cols)

    # Open IOS files
    for i, f in enumerate(nclist):
        print(i, f)
        ios_data = open_dataset(f)

        # Check if var is available in each file
        flag = 0
        try:
            var_data = ios_data[varcode].data
        except KeyError:
            print(basename(f))
            print('Warning: requested variable', varcode, 'not available in dataset')
            flag += 1

        # If variable not present, skip to next iteration
        if flag == 1:
            continue

        # Get unique profile indices to allow filtering through "row" dimension
        indices = np.unique(ios_data.profile.data, return_index=True)[1]

        ios_fname_array = np.repeat(basename(f), len(indices))
        ios_institute_array = np.repeat(ios_data.institution, len(indices))

        if 'CTD' in f:
            inst = 'CTD'
        elif 'BOT' in f:
            inst = 'BOT'

        print(inst)
        ios_instrument_type_array = np.repeat(inst, len(indices))

        # Time strings: yyyymmddhhmmsszzz; slow to run
        ios_time_strings = pd.to_datetime(
            ios_data.time.data[indices]).strftime('%Y%m%d%H%M%S')

        # QC flags: good data by default, according to Germaine
        ios_flags = np.ones(len(indices))

        # Take transpose of arrays?
        ios_df_add = pd.DataFrame(
            data=np.array([ios_fname_array,
                           ios_institute_array,
                           ios_data.mission_id.data[indices],
                           ios_instrument_type_array,
                           ios_time_strings,
                           ios_data.latitude.data[indices],
                           ios_data.longitude.data[indices],
                           ios_flags]).transpose(), columns=df_cols)

        ios_df = pd.concat([ios_df, ios_df_add])

        # Close dataset
        ios_data.close()

    return ios_df


def ios_wp_to_pdt(nclist, var):
    if var == 'Oxy':
        varcode = 'DOXMZZ01'
    elif var == 'Temp':
        varcode = 'TEMPS901'
    elif var == 'Sal':
        varcode = 'PSALST01'

    # df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
    #            "Instrument_type", "Date_string", "Latitude",
    #            "Longitude", "Quality_control_flag"]

    # Initialize dataframe for IOS data
    # ios_wp_df = pd.DataFrame(columns=df_cols)

    dict_list = []

    for i, f in enumerate(nclist):
        print(i, f)

        # Open file
        ncdata = open_dataset(f)

        # Check if var is available in each file
        flag = 0
        try:
            var_data = ncdata[varcode].data
        except KeyError:
            print(basename(f))
            print('Warning: requested variable', varcode, 'not available in dataset')
            flag += 1

        # If variable not present, skip to next iteration
        if flag == 1:
            continue

        if 'bot' in f:
            instrument_type = 'BOT'
        elif 'ctd' in f:
            instrument_type = 'CTD'

        # Initialize dataframe
        # df_add = pd.DataFrame(
        #     data=np.array([]).transpose(),
        #     columns=df_cols)

        # ios_wp_df = pd.concat([ios_wp_df, df_add])

        dict_list.append({'Source_data_file_name': basename(f),
                          'Institute': ncdata.institution,
                          'Cruise_number': ncdata.mission_id.data,
                          'Instrument_type': instrument_type,
                          'Date_string': pd.to_datetime(ncdata.time.data).strftime(
                              '%Y%m%d%H%M%S'),
                          'Latitude': ncdata.latitude.data,
                          'Longitude': ncdata.longitude.data,
                          'Quality_control_flag': 1})

    df_out = pd.DataFrame.from_dict(dict_list)

    return df_out


def nodc_to_pdt(nodc_files, sourcetype, var, output_folder):
    df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
               "Instrument_type", "Date_string", "Latitude",
               "Longitude", "Quality_control_flag"]

    nodc_df = pd.DataFrame(columns=df_cols)

    for f in nodc_files:
        # Read in netCDF file
        nodc_nocad_data = open_dataset(f)

        # Casts is the dim counting the number of profiles
        nodc_nocad_fname_array = np.repeat(
            basename(f), len(nodc_nocad_data.casts.data))

        # Make array of institute name
        nodc_nocad_institute_array = np.repeat(
            nodc_nocad_data.institution, len(nodc_nocad_data.casts.data))

        # Get instrument type from file name
        if 'CTD' in f:
            inst = 'CTD'
        elif 'OSD' in f:
            inst = 'BOT'
        elif 'PFL' in f:  # Profiling float (Argo) -- only temp (and sal?) data
            inst = 'PFL'
        elif 'DRB' in f:
            inst = 'DRB'  # Drifting buoy
        elif inst == 'GLD':
            inst = 'GLD'  # Glider

        nodc_nocad_instrument_array = np.repeat(inst,
                                                len(nodc_nocad_data.casts.data))

        # Convert time data to time string type
        nodc_nocad_timestring = pd.to_datetime(
            nodc_nocad_data.time.data).strftime('%Y%m%d%H%M%S%z')

        if var == 'Oxy':
            var_flag = 'Oxygen_WODprofileflag'
        elif var == 'Temp':
            var_flag = 'Temperature_WODprofileflag'
        elif var == 'Sal':
            var_flag = 'Salinity_WODprofileflag'

        nodc_df_add = pd.DataFrame(
            data=np.array([nodc_nocad_fname_array,
                           nodc_nocad_institute_array,
                           nodc_nocad_data.WOD_cruise_identifier.data.astype(str),
                           nodc_nocad_instrument_array,
                           nodc_nocad_timestring,
                           nodc_nocad_data.lat.data,
                           nodc_nocad_data.lon.data,
                           nodc_nocad_data[var_flag].data]).transpose(),
            columns=df_cols)

        # Append the new dataframe to the existing dataframe
        nodc_df = pd.concat([nodc_df, nodc_df_add],
                            ignore_index=True)

    print(nodc_df.columns)
    print(nodc_df)

    print(min(nodc_df['Date_string']), max(nodc_df['Date_string']))

    # Export to csv file
    # output_folder = '/home/hourstonh/Documents/climatology/data_extracts/'
    nodc_name = 'NODC_{}_Profiles_{}_1991_2020.csv'.format(sourcetype, var)
    nodc_df.to_csv(output_folder + nodc_name)

    return


def meds_to_pdt(csvfiles, var):
    # MEDS is the only one that cals Salinity PSAL, so need to update
    if var == 'Sal':
        var = 'PSAL'

    df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
               "Instrument_type", "Date_string", "Latitude",
               "Longitude", "Quality_control_flag"]

    # MEDS data: initialize empty dataframe
    meds_df = pd.DataFrame(columns=df_cols)

    # Iterate through csv files
    for i in range(len(csvfiles)):
        # Skip if var not in file
        if var.upper() not in basename(csvfiles[i]):
            continue

        meds_data = pd.read_csv(csvfiles[i])

        # print(meds_data.head())

        # Get number of unique profiles
        unique = np.unique(meds_data.loc[:, 'RowNum'], return_index=True)[1]

        # Oxy data spans 1991-01-22 05:13:00 to 1995-03-09 23:35:00
        meds_fname_array = np.repeat(basename(csvfiles[i]), len(unique))

        # Get instrument from file name
        if 'CD' in basename(csvfiles[i]):
            inst = 'CTD'
        elif 'BO' in basename(csvfiles[i]):
            inst = 'BOT'
        elif 'XB' in basename(csvfiles[i]):
            inst = 'XBT'

        meds_instrument_array = np.repeat(inst, len(unique))

        # Time string data
        meds_data['Hour'] = meds_data.Time.astype(str).apply(lambda x: ('000' + x)[-4:][:-2])
        meds_data['Minute'] = meds_data.Time.astype(str).apply(lambda x: ('000' + x)[-4:][-2:])

        # meds_data['Hour'] = meds_data.Time.astype(str).apply(lambda x: ('000' + x)[:-2])
        # meds_data['Minute'] = meds_data.Time.astype(str).apply(lambda x: ('000' + x)[-2:])

        print(np.where(pd.isnull(meds_data.Hour)))
        print(np.where(pd.isnull(meds_data.Minute)))

        meds_data['Timestring'] = pd.to_datetime(
            meds_data[['Year', 'Month', 'Day', 'Hour', 'Minute']]).dt.strftime(
            '%Y%m%d%H%M%S')

        print(np.where(pd.isnull(meds_data.Timestring)))

        # meds_data['Time_pd'] = pd.to_datetime(
        #     meds_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        #
        # print(min(meds_data['Time_pd']), max(meds_data['Time_pd']))

        # # DataFrame columns
        # df_cols = ["Source_data_file_name", "Institute", "Cruise_number",
        #            "Instrument_type", "Date_string", "Latitude",
        #            "Longitude", "Quality_control_flag"]

        # Need to convert MEDS longitude from positive towards West to positive
        # towards East
        meds_df_add = pd.DataFrame(
            data=np.array([meds_fname_array,
                           meds_data.loc[unique, 'SourceID'],
                           meds_data.loc[unique, 'CruiseID'],
                           meds_instrument_array,
                           meds_data.loc[unique, 'Timestring'],
                           meds_data.loc[unique, 'Lat'],
                           -meds_data.loc[unique, 'Lon'],
                           meds_data.loc[unique, 'PP_flag']]).transpose(),
            columns=df_cols
        )

        meds_df = pd.concat([meds_df, meds_df_add])

    # print(np.where(pd.isna(meds_df)))

    return meds_df


def gather_raw_data(var, output_folder):
    # Find all oxygen data
    # var = 'Oxy', 'Temp', or 'Sal'

    # IOS CIOOS Pacific files
    # ios_path = '/home/hourstonh/Documents/climatology/data/IOS_CIOOS/'
    ios_path = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
               'source_format\\IOS_CIOOS\\'
    ios_files = glob.glob(ios_path + '*{}*.nc'.format(var), recursive=False)
    ios_files.sort()
    print('Number of IOS files', len(ios_files))

    # IOS Water Properties files
    ios_wp_path = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
                  'source_format\\SHuntington\\'
    # Get bot files
    ios_wp_files = glob.glob(ios_wp_path + '*.bot.nc', recursive=False)
    # Get ctd files
    ios_wp_files += glob.glob(ios_wp_path + 'WP_unique_CTD_forHana\\*.ctd.nc',
                              recursive=False)
    print('Number of IOS WP files', len(ios_wp_files))

    # NODC WODSelect files, non-Canadian
    if var == 'Oxy':
        nodc_nocad_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\' \
                          'data\\source_format\\WOD_extracts\\' \
                          'Oxy_WOD_May2021_extracts\\'
    else:
        nodc_nocad_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\data\\' \
                          'source_format\\WOD_extracts\\WOD_July_extracts\\'
    # nodc_nocad_path = '/home/hourstonh/Documents/climatology/data/WOD_extracts/' \
    #                   'Oxy_WOD_May2021_extracts/'
    nodc_nocad_files = glob.glob(nodc_nocad_path + '{}*.nc'.format(var),
                                 recursive=False)
    nodc_nocad_files.sort()
    print('Number of NODC nocad files', len(nodc_nocad_files))

    # NODC WODSelect files, Canadian non-IOS
    nodc_cad_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\data\\' \
                    'source_format\\WOD_extracts\\WOD_July_CDN_nonIOS_extracts\\'
    # nodc_cad_path = '/home/hourstonh/Documents/climatology/data/WOD_extracts/' \
    #                 'WOD_July_CDN_nonIOS_extracts/'
    nodc_cad_files = glob.glob(nodc_cad_path + '{}*.nc'.format(var),
                               recursive=False)
    nodc_cad_files.sort()
    print('Number of NODC cad files', len(nodc_cad_files))

    # MEDS files (Canadian waters)
    meds_path = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\data\\' \
                'source_format\\meds_data_extracts\\'
    meds_files = glob.glob(meds_path + '*\\*{}*source.csv'.format(var.upper()),
                           recursive=False)
    meds_files.sort()
    print('number of meds files', len(meds_files))

    # Create PDT files
    # Start with IOS CIOOS files
    ios_out_df = ios_to_pdt(ios_files, var=var)
    ios_df_name = output_folder + 'IOS_Profiles_{}_1991_2020_pdt.csv'.format(var)
    ios_out_df.to_csv(ios_df_name, index=False)

    ios_wp_out_df = ios_wp_to_pdt(ios_wp_files, var=var)
    ios_wp_name = output_folder + 'IOS_WP_Profiles_{}_1991_2020_pdt.csv'.format(var)
    ios_wp_out_df.to_csv(ios_wp_name, index=False)

    nodc_to_pdt(nodc_nocad_files, sourcetype='noCAD', var=var)
    nodc_to_pdt(nodc_cad_files, sourcetype='CAD', var=var)

    meds_out_df = meds_to_pdt(meds_files, var=var)
    meds_csv_name = output_folder + 'MEDS_Profiles_{}_1991_1995_pdt.csv'.format(var)
    meds_out_df.to_csv(meds_csv_name)

    return


outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
         'profile_data_tables\\'
gather_raw_data('Temp', outdir)
gather_raw_data('Sal', outdir)

# Second version of oxygen data that includes Argo oxygen sensor data
argo_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
           'profile_data_tables\\Argo\\'

indir = 'C:\\Users\HourstonH\\Documents\\NEP_climatology\\' \
        'data\\source_format\\WOD_extracts\\' \
        'Oxy_WOD_May2021_extracts\\'

argo_files = glob.glob(indir + 'Oxy*PFL.nc')

nodc_to_pdt(argo_files, sourcetype="noCAD_PFL", var='Oxy',
            output_folder=argo_dir)

###################################
# COMBINE ALL PROFILE DATA TABLES

def combine_all_pdt():
    # extract_folder = '/home/hourstonh/Documents/climatology/data_extracts/'
    extract_folder = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\' \
                     'data_extracts\\'
    extracts = glob.glob(extract_folder + '*.csv', recursive=False)
    extracts.sort()

    colnames = ["Source_data_file_name", "Institute", "Cruise_number",
                "Instrument_type", "Date_string", "Latitude",
                "Longitude", "Quality_control_flag"]

    df_all = pd.DataFrame(columns=colnames)

    for fi in extracts:
        df_add = pd.read_csv(fi)
        df_all = pd.concat([df_all, df_add], ignore_index=True)

    # Remove unwanted column
    df_all = df_all.drop(columns=['Unnamed: 0'])

    df_all['Quality_control_flag'] = df_all['Quality_control_flag'].astype(int)

    # Write to new csv file for ease
    df_all_name = 'ALL_Profiles_Oxy_1991_2020.csv'
    df_all.to_csv(extract_folder + df_all_name)

    return

