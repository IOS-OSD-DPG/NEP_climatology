"""
Copy the method of calling DIVAnd Julia from
https://github.com/gher-ulg/DIVAnd.py/blob/master/DIVAnd/DIVAnd.py

"""

import numpy as np
import julia
from julia import DIVAnd as D


def DIVAnd_cv(mask, pmn, xi, x, f, corlen, epsilon2, nl, ne, method=0):
    """
    Performs a cross validation to estimate the analysis parameters
    (correlation length and signal-to-noise ratio).
    https://github.com/gher-ulg/DIVAnd.jl/blob/master/src/DIVAnd_cv.jl

    :param mask: binary mask delimiting the domain. true is inside and false outside
    :param pmn: scale factor of the grid. pmn is a tuple with n elements. Every element
    represents the scale factor of the corresponding dimension. Its inverse is the local
    resolution of the grid in a particular dimension.
    :param xi: tuple with n elements. Every element represents a coordinate of the final
    grid on which the observations are interpolated
    :param x: tuple with n elements. Every element represents a coordinate of the observations
    :param f: value of the observations minus the background estimate (m-by-1 array)
    :param corlen: correlation length
    :param epsilon2: error variance of the observations (normalized by the error variance of
    the background field). epsilon2 can be a scalar (all observations have the same error
    variance and their errors are decorrelated), a vector (all observations can have a
    difference error variance and their errors are decorrelated) or a matrix (all observations
    can have a difference error variance and their errors can be correlated). If epsilon2 is a
    scalar, it is thus the inverse of the signal-to-noise ratio
    :param nl: number of testing points around the current value of L. 1 means one additional
    point on both sides of the current L. 0 is allowed and means the parameter is not optimised
    :param ne: number of testing points around the current value of epsilon2. 0 is allowed as
    for nl
    :param method: cross validation estimator method 1: full CV; 2: sampled CV; 3: GCV;
    0: automatic choice between the three possible ones, default value

    :return bestfactorl: best estimate of the multiplication factor to apply to len
    :return bestfactore: best estimate of the multiplication factor to apply to epsilon2
    :return cvvales: the cross validation values calculated
    :return factors: the tested multiplication factors
    :return cvinter: the interpolated cv values for final optimisation
    :returns X2Data, Y2Data: coordinates of sampled cross validation in L,epsilon2 space.
    Normally only used for debugging or plotting
    :returns Xi2D, Yi2d: coordinates of interpolated estimator. Normally only used for
    debugging or plotting
    """
    bestfactorl, bestfactore, cvval, cvvalues, x2Ddata, y2Ddata, cvinter, xi2D, yi2D = D.DIVAnd_cv(
        np.transpose(mask), tuple([np.transpose(_) for _ in pmn]),
        tuple([np.transpose(_) for _ in xi]), x, f, corlen, epsilon2, nl, ne, method)

    return bestfactorl, bestfactore, cvval, cvvalues, x2Ddata, y2Ddata, cvinter, xi2D, yi2D

