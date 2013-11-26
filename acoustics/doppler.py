"""
Doppler shift module.
"""


def velocity_from_doppler_shift(c, f1, f2):
    """
    Calculate velocity based on measured frequency shifts due to Doppler shift.
    The assumption is made that the velocity is constant between the observation times.
    
    .. math:: v = c \cdot \\left( \\frac{f_2 - f_1}{f_2 + f_1} \\right)
    
    
    :param c: Speed of sound :math:`c`.
    :param f1: Lower frequency :math:`f_1`.
    :param f2: Upper frequency :math:`f_2`.
    
    """
    return c * (f2 - f1) / (f2 + f1)
    