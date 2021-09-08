import csv
import numpy as np


def get_standard_levels(fpath_sl):
    # Return array of standard levels from the standard levels text file

    # Initialize list with each element being a row in file_sl
    sl_list = []
    with open(fpath_sl, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            sl_list += row

    # Remove empty elements: '' and ' '
    # Gotta love list comprehension
    sl_list_v2 = [int(x.strip(' ')) for x in sl_list if x not in ['', ' ']]

    # Convert list to array
    sl_arr = np.array(sl_list_v2)
    return sl_arr
