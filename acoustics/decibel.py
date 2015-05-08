"""
Decibel
=======

The `decibel` module contains basic functions for decibel arithmetic.

"""

import numpy as np

def dbsum(levels, axis=None):
    """Energetic summation of levels.
    
    :param levels: Sequence of levels.
    :param axis: Axis over which to perform the operation.
    
    .. math:: 10 \\log_{10}{\\sum_i^n{10^{L/10}}}
    
    """
    levels = np.asanyarray(levels)
    return 10.0 * np.log10((10.0**(levels/10.0)).sum(axis=axis))


def dbmean(levels, axis=None):
    """Energetic average of levels.
    
    :param levels: Sequence of levels.
    :param axis: Axis over which to perform the operation.
    
    .. math:: 10 \\log_{10}{\\frac{1}{n}\\sum_i^n{10^{L/10}}}
    
    """
    levels = np.asanyarray(levels)
    return 10.0 * np.log10((10.0**(levels/10.0)).mean(axis=axis))
    

def dbadd(a, b):
    """Energetic addition.
    
    :param a: Single level or sequence of levels.
    :param b: Single level or sequence of levels.
    
    .. math:: 10 \\log_{10}{10^{L_b/10}+10^{L_a/10}}
    
    Energetically adds b to a.
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    return 10.0*np.log10(10.0**(a/10.0)+10.0**(b/10.0))


def dbsub(a, b):
    """Energetic subtraction.
    
    :param a: Single level or sequence of levels.
    :param b: Single level or sequence of levels.
    
    .. math:: 10 \\log_{10}{10^{L_b/10}-10^{L_a/10}}
    
    Energitally subtract b from a.
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    return 10.0*np.log10(10.0**(a/10.0)-10.0**(b/10.0))
