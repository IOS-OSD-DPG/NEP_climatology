""" Sept 7, 2021
Generate summary tables for NEP climatology
Organize by instrument, season/year

Do lat/lon check earlier??
"""

import pandas as pd
import numpy as np
import glob


dir1 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
       'value_vs_depth\\1_original\\'

dir8 = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\' \
       'value_vs_depth\\8_gradient_check\\'

files1 = glob.glob(dir1 + '*Oxy*.csv')

file8 = dir8 + 'ALL_Oxy_1991_2020_value_vs_depth_grad_check_done.csv'

# Need to apply lat/lon check to file8 !!!!!!


