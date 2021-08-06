# Add matching temperature and salinity data to oxygen data in one table

import pandas as pd
from copy import deepcopy
from tqdm import trange
import numpy as np

in_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\source_format\\' \
         'meds_data_extracts\\bo_extracts\\'

o_file = in_dir + 'MEDS_19940804_19930816_BO_DOXY_profiles_source.csv'
t_file = in_dir + 'MEDS_19940804_19930816_BO_TEMP_profiles_source.csv'
s_file = in_dir + 'MEDS_19940804_19930816_BO_PSAL_profiles_source.csv'

df_o = pd.read_csv(o_file)
df_t = pd.read_csv(t_file)
df_s = pd.read_csv(s_file)

df_out = deepcopy(df_o)

# Rename oxygen column from ProfParm and oxygen flag
df_out = df_out.rename(columns={'ProfParm': 'DOXY', 'PP_flag': 'DOXY_flag'})


# Initialize new columns in df_out
df_out['TEMP'] = np.repeat(-9., len(df_out))
df_out['TEMP_flag'] = np.zeros(len(df_out), dtype=int)
df_out['PSAL'] = np.repeat(-9., len(df_out))
df_out['PSAL_flag'] = np.zeros(len(df_out), dtype=int)


for i in trange(len(df_out)):
    # Make masks (True or False for each row)
    # 12th column is D_P_flag
    mask_temp = (df_t.iloc[:, :12] == df_out.iloc[i, :12]).all(axis=1)
    mask_psal = (df_s.iloc[:, :12] == df_out.iloc[i, :12]).all(axis=1)
    # Check that masks are not empty
    if len(mask_temp[mask_temp]) == 1 and len(mask_psal[mask_psal]) == 1:
        # print('Row match found')
        # Add temp and psal data to df_out
        df_out.loc[i, 'TEMP'] = df_t.loc[mask_temp, 'ProfParm'].values[0]
        df_out.loc[i, 'TEMP_flag'] = df_t.loc[mask_temp, 'PP_flag'].values[0]
        df_out.loc[i, 'PSAL'] = df_s.loc[mask_psal, 'ProfParm'].values[0]
        df_out.loc[i, 'PSAL_flag'] = df_s.loc[mask_psal, 'PP_flag'].values[0]

print(df_out.TEMP_flag)

print(len(df_out.loc[df_out.TEMP == -9.]))
print(len(df_out.loc[df_out.PSAL == -9.]))

outname = in_dir + 'MEDS_19940804_19930816_BO_TSO_profiles_source.csv'

df_out.to_csv(outname, index=False)

