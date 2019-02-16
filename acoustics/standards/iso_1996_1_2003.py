"""
ISO 1996-1:2003
===============

ISO 1996-1:2003 defines the basic quantities to be used for the description of
noise in community environments and describes basic assessment procedures. It
also specifies methods to assess environmental noise and gives guidance on
predicting the potential annoyance response of a community to long-term exposure
from various types of environmental noises. The sound sources can be separate or
in various combinations. Application of the method to predict annoyance response
is limited to areas where people reside and to related long-term land uses.


"""
import numpy as np


def composite_rating_level(levels, hours, adjustment):
    """Composite rating level.

    :params levels: Level per period.
    :params hours: Amount of hours per period.
    :params adjustment: Adjustment per period.

    Composite whole-day rating levels are calculated as

    .. math:: L_R = 10 \\log{\\left[ \\sum_i \\frac{d_i}{24} 10^{(L_i+K_i)/10}  \\right]}

    where :math:`i` is a period. See equation 6 and 7 of the standard.

    .. note:: Summation is done over the last axis.
    """
    levels = np.asarray(levels)
    hours = np.asarray(hours)
    adjustment = np.asarray(adjustment)

    return 10.0 * np.log10((hours / 24.0 * 10.0**((levels + adjustment) / 10.0)).sum(axis=-1))
