"""Oct. 18, 2021
Calculate the mean triangle length from the unstructured triangle grid"""

import numpy as np
import os
import matplotlib.tri as mtri

indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\MForeman\\'
tri_filename = os.path.join(indir, 'nep35_reord.tri')
grid_filename = os.path.join(indir, 'nep35_reord_latlon_wgeo.ngh')

tri_data = np.genfromtxt(indir + tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3))-1

grid_data = np.genfromtxt(grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                          names=['node', 'lon', 'lat', 'type', 'depth',
                                 's1', 's2', 's3', 's4', 's5', 's6'],
                          delimiter="", skip_header=3)

# create a data dictionary, and write data into dictionary
data_dict = dict()
data_dict['node_number'] = grid_data['node'] - 1  # use node_number as Key
data_dict['depth_in_m'] = grid_data['depth']
data_dict['y_lat'] = grid_data['lat']
data_dict['x_lon'] = grid_data['lon']

# attributes: .mask, .triangles, .edges, .neighbors
tri = mtri.Triangulation(data_dict['x_lon'], data_dict['y_lat'], tri_data)

