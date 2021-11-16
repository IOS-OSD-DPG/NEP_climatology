from clim_helpers import date_string_to_datetime
import os
import numpy as np
import pandas as pd

# Separate cleaned observation level data by year and season for using in
# DIVAnd fithorzlen() to determine ranges for correlation length

var_name = 'Oxy'
years = np.arange(1991, 2021)
szns = ['JFM', 'AMJ', 'JAS', 'OND']

indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
        '9_gradient_check\\'
infile_name = os.path.join(indir + 'Oxy_1991_2020_value_vs_depth_grad_check_done.csv')

outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
         '15_sep_by_year_szn_obslevel\\'

df_in = pd.read_csv(infile_name)
# Add time column to dataframe
df_in_t = date_string_to_datetime(df_in)
# Add year column to dataframe
df_in_t['Year'] = df_in_t.Time_pd.dt.year
# Add month column to dataframe
df_in_t['Month'] = df_in_t.Time_pd.dt.month
# Iterate through the years
for y in years:
    print(y)
    for i in range(len(szns)):
        months = (3 * i + 1, 3 * i + 2, 3 * i + 3)
        # Subset dataframe by year and month
        df_out = df_in_t[
            (df_in_t.Year == y) &
            ((df_in_t.Month == months[0]) | (df_in_t.Month == months[1]) |
             (df_in_t.Month == months[2]))]
        # Remove unnecessary columns
        df_out.drop(columns=['Cruise_number', 'Instrument_type', 'Time_pd', 'Year', 'Month'],
                    inplace=True)
        # Export the dataframe
        df_out_name = os.path.join(outdir + '{}_{}_{}_obslevel.csv'.format(var_name, y, szns[i]))
        df_out.to_csv(df_out_name, index=False)
