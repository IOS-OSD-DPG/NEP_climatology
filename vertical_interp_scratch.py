'''
June 25, 2021

Testing an R function for NODC vertical interpolation.
oceApprox() implements Reiniger-Ross interpolation when 4 points are
available, Lagrangian interpolation when only 3 points are available,
and linear interpolation when only 2 points are available.
oceApprox() docs available at:
https://rdrr.io/cran/oce/man/oceApprox.html

The Python package rpy2 is needed for using R packages and functions
in Python. rpy2 docs available at:
https://rpy2.github.io/doc/v3.0.x/html/introduction.html#getting-started

Instructions for installing and using the oceApprox R function in Python:
    Activate conda environment: conda activate conda_env
    Install conda-build: conda install conda-build
    Install Python rpy2 package: conda install rpy2
    Install R oce package: conda install r-oce
    Install R ocedata package: conda install r-ocedata

In Python/Pycharm:
from rpy2 import robjects
from rpy2.robjects.packages import importr

# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

# import R "oce" package
oce = importr('oce')

roceApprox = robjects.r['oceApprox']
'''

from rpy2 import robjects
from rpy2.robjects.packages import importr
from numpy.random import randint, seed
from numpy import linspace, array

# import R's "base" package
# base = importr('base')

# import R's "utils" package
# utils = importr('utils')

# import R "oce" package
oce = importr('oce')

# Get the oceApprox function from the R oce package
roceApprox = robjects.r['oceApprox']

# Set numpy random seed
seed(12345)

# Test case arrays
zpts = linspace(0, 5001, 20) #depth
# opts = randint(0, 436, size=20) #oxygen
opts = zpts**0.25

# Depth levels to interpolate to == 62 levels
zout = linspace(0, 5001, 52)

# Convert the numpy arrays to rpy2 (R) vectors
rzpts = robjects.FloatVector(zpts)
ropts = robjects.FloatVector(opts)
rzout = robjects.FloatVector(zout)

# Print the R representation
print(rzpts.r_repr())

# Need to convert python arrays to R arrays
result = roceApprox(rzpts, ropts, rzout, 'unesco')

# Convert the result back to a numpy array
result_np = array(result)

print(result_np)
