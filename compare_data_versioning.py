"""
Compare versions of Argo and IOS data from Sam and Nick
"""

from xarray import open_dataset
import glob
import numpy as np

sam_argo_dir = '/home/hourstonh/Documents/climatology/data/oxy_clim/Coriolis_Argo/'
sam_argo_files = glob.glob(sam_argo_dir + '*.nc', recursive=False)
sam_argo_files.sort()

for f in sam_argo_files:
    sdata = open_dataset(f)
    print(np.where(sdata.DOXY_QC.data == '1'))

sam_argo_files