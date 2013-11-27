"""
Module for working with octaves.

The following is an example on how to use :class:`acoustics.octave.Octave`.

.. literalinclude:: ../examples/octave.py

"""
import numpy as np


def band_of_frequency(f, order=1, ref=1000.0):
    """
    Calculate the band ``n`` from a given frequency.
    
    :param f: Frequency
    """
    return np.round( ( np.log2(f/ref) - 1.0/order ) * order)



class Octave(object):
    """
    Class to calculate octave center frequencies.
    """
    
    REF = 1000.0
    """Reference center frequency :math:`f_{c,0}`."""
    
    def __init__(self, order=1, interval=None, fmin=None, fmax=None, unique=False):
        
        self.order = order
        """
        Order
        """
        
        if (interval is not None) and (fmin is not None or fmax is not None):
            raise AttributeError
        
        self._interval = interval
        """Interval"""
        
        self._fmin = fmin
        """Minimum frequency of a range."""
        
        self._fmax = fmax
        """Maximum frequency of a range."""
        
        self.unique = unique
        """Whether or not to calculate the requested values for every value of ``interval``."""
        
        
    def _set_fmin(self, x):
        if self.interval is not None:
            pass    # Warning, remove interval first.
        else:
            self._fmin = x
    
    def _set_fmax(self, x):
        if self.interval is not None:
            pass
        else:
            self._fmax = x
    
    def _get_fmin(self):
        if self._fmin is not None:
            return self._fmin
        elif self._interval is not None:
            return self.interval.min()
    
    def _get_fmax(self):
        if self._fmax is not None:
            return self._fmax
        elif self._interval is not None:
            return self.interval.max()
        
    def _set_interval(self, x):
        if self._fmin or self._fmax:
            pass
        else:
            self._interval = x if isinstance(x, np.ndarray) else np.array(x)
    
    def _get_interval(self):
        return self._interval
    
    fmin = property(fget=_get_fmin, fset=_set_fmin)
    """Minimum frequency of an interval."""
    fmax = property(fget=_get_fmax, fset=_set_fmax)
    """Maximum frequency of an interval."""
    interval = property(fget=_get_interval, fset=_set_interval)
    """Interval."""
    
    def _n(self, f):
        """
        Calculate the band ``n`` from a given frequency.
        
        :param f: Frequency
        """
        return np.round( ( np.log2(f/self.REF) - 1.0/self.order ) * self.order)
    
    def _fc(self, n):
        """
        Calculate center frequency of band ``n``.
        
        :param n: band ``n`.
        
        """
        return self.REF * 10.0**(3.0/self.order/10.0) * 2.0**(n/self.order)
    
    def n(self):
        """
        Return band ``n`` for a given frequency.
        """
        if self.interval is not None and self.unique:
            return self._n(self.interval)
        else:
            return np.arange(self._n(self.fmin), self._n(self.fmax)+1)
            
    def center(self):
        """
        Return center frequencies :math:`f_c`.
        
        .. math::  f_c = f_{ref} \cdot 2^{n/N} \\cdot 10^{\\frac{3}{10N}}
        
        """
        n = self.n()
        return self._fc(n)
     
    def bandwidth(self):
        """
        Bandwidth of bands.
        
        .. math:: B = f_u - f_l
        
        """
        return self.upper() - self.lower()
    
    def lower(self):
        """
        Lower frequency limits of bands.
        
        .. math:: f_l = f_c \cdot 2^{\\frac{-1}{2N}}
        
        """
        return self.center() * 2.0**(-1.0/(2.0*self.order))
    
    def upper(self):
        """
        Upper frequency limits of bands.
        
        .. math:: f_u = f_c \cdot 2^{\\frac{+1}{2N}}
        
        """
        return self.center() * 2.0**(+1.0/(2.0*self.order))
    
    
    
###def center_frequency_octave(frequencies, order=1):
    ###"""
    ###Calculate the center frequencies :math:`f_c` of the octaves that (partially) cover ``frequencies``.
    
    ###:param frequencies: An iterable containing frequencies.
    ###:param order: An integer indicating the octave order. E.g., for 1/3-octaves use ``order=3``
    
    ###Center frequencies are calculated using:
    
    ###.. math::  f_c = 1000 \cdot 2^{n/N} \\cdot 10^{\\frac{3}{10N}}
    
    ###"""
    ###n = lambda fc, N: np.log2(fc/1000.0) - 1.0/N    # To calculate the nth octave from a given frequency.
    
    ###n_min = np.floor(n(np.min(frequencies), order))
    ###n_max = np.ceil(n(np.max(frequencies), order))
    ###n = np.arange(n_min, n_max+1)
    ###fc = 1000.0 * 10.0**(3.0/order/10.0) * 2.0**(n)
    ###return fc

