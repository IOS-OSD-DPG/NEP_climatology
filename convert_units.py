"""
Convert variable units for NEP climatology

Some IOS oxygen data needs converting to umol/kg?
    - Not BOT, PCTD or MCTD
    - So no...
Some MEDS pressure data needs conversion to depth

Convert WOD time data from GMT to UTC?

Check if there are other units that need converting
"""
import glob
from os.path import basename
import gsw
from xarray import open_dataset
from pandas import read_csv


# Convert oxygen ml/l units to umol/kg for IOS
# IOS: DOXMZZ01 is umol/kg; DOXYZZ01 is mL/L

# DOXMZZ01
# Concentration of dissolved oxygen per unit mass
# of the water column. Oxygen may be expressed in
# terms of mass, volume or quantity of substance.

# DOXYZZ01:
# Concentration of dissolved oxygen per unit volume
# of the water column. Oxygen may be expressed in
# terms of mass, volume or quantity of substance.

iosdir = '/home/hourstonh/Documents/climatology/data/oxy_clim/IOS_CIOOS/'
ncfiles = glob.glob(iosdir + '*.nc')

for ncfile in ncfiles:
    ncdata = open_dataset(ncfile)
    print(basename(ncfile))
    print(ncdata.DOXYZZ01.units)
    print(ncdata.DOXMZZ01.units)
    ncdata.close()

# No units need converting as both mL/L and umol/kg are available

# Calculate the solubility of oxygen in [umol/kg]
# Absolute salinity, SA, must be in G/KG
# Conservative temperature, CT, in deg Celsius
# Sea pressure, p, must be in DBAR
# sea_pres = ncdata.PRESPR01.data - 10.1325 # dBar
#umol_kg = gsw.O2sol(SA=, CT=, p=sea_pres, long=ncdata.longitude.data, lat=ncdata.latitude.data)


# Convert pressure to depth for MEDS
# Do for inexact duplicates (*idr.csv) version files
medsdir = '/home/hourstonh/Documents/climatology/data_explore/oxygen/MEDS/bo_profile_extracts/'
medsfiles = glob.glob(medsdir + '*idr.csv')

# Iterate through the meds files to add a pressure column and depth column
for f in medsfiles:
    df = read_csv(f)
    df['Depth'] = df['Depth/Press'].where(df['D_P_Flag'] == 'D')
    df['Press'] = df['Depth/Press'].where(df['D_P_Flag'] == 'P')
    d = df['Depth/Press'].loc[df['D_P_Flag'] == 'D']
    p = df['Depth/Press'].loc[df['D_P_Flag'] == 'P']
    lat_d = df['Lat'].loc[df['D_P_Flag'] == 'D']
    lat_p = df['Lat'].loc[df['D_P_Flag'] == 'P']
    # Take the negative of height since height is positive up
    # and we want depth which is positive down
    df['Depth'].loc[df['D_P_Flag'] == 'P'] = -gsw.z_from_p(p, lat_p)
    df['Press'].loc[df['D_P_Flag'] == 'D'] = gsw.p_from_z(-d, lat_d)
    # Output the updated dataframe to a new csv file
    outname = basename(f)[:-4] + '_p2z.csv'
    df.to_csv(medsdir + outname, index=False)
    

