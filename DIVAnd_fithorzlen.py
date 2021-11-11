"""
Copy the method of calling DIVAnd Julia from
https://github.com/gher-ulg/DIVAnd.py/blob/master/DIVAnd/DIVAnd.py

"""

import numpy as np
import julia
from julia import DIVAnd as D


def fithorzlen(x, value, z, smoothz=100, searchz=50, maxnsamp=5000, limitlen=False):
    """
    Function description from https://gher-ulg.github.io/DIVAnd.jl/latest/#DIVAnd.fithorzlen.
    Determines the horizontal correlation length lenxy based on the measurments value at
    the location x (tuple of 3 vectors corresponding to longitude, latitude and depth)
    at the depth levels defined in z.

    Optional arguments:
    :param smoothz: spatial filter for the correlation scale
    :param searchz: vertical search distance (can also be a function of the depth)
    :param maxnsamp: maximum number of samples
    :param limitlen: limit correlation length by mean distance between observations

    Leaving limitfun, epsilon2 and distfun optional parameters out, so that they are set
    to the default values below.
    limitfun (default no function): a function with with the two arguments (depth and
    estimated correlation length) which returns an adjusted correlation length.
    epsilon2 (default is a vector of the same size as value with all elements equal to 1):
    the relative error variance of the observations. Less reliable observation would have
    a larger corresponding value.
    distfun: function computing the distance between the points xi and xj. Per default it
    represent the Euclidian distance.

    :return lenxy: horizontal correlation length
    :return dbinfo: ????
    """
    # epsilon2 = np.ones(len(value))

    # Need to modify structure of tuple for julia
    xjulia = tuple([np.transpose(_) for _ in x])

    # Call the julia function from python
    lenxy, dbinfo = D.fithorzlen(xjulia, value, z)  # , smoothz, searchz, maxnsamp, limitlen)
    return lenxy, dbinfo
