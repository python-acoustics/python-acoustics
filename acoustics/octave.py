"""
Octave
======

Module for working with octaves.

The following is an example on how to use :class:`acoustics.octave.Octave`.

.. literalinclude:: ../examples/octave.py

"""
import numpy as np
import acoustics

#REFERENCE = 1000.0
#"""
#Reference frequency.
#"""

from acoustics.standards.iec_61260_1_2014 import index_of_frequency
from acoustics.standards.iec_61260_1_2014 import REFERENCE_FREQUENCY as REFERENCE


def exact_center_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Exact center frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :return: Exact center frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.index_of_frequency`

    """
    if frequency is not None:
        n = acoustics.standards.iec_61260_1_2014.index_of_frequency(frequency, fraction=fraction, ref=ref)
    return acoustics.standards.iec_61260_1_2014.exact_center_frequency(n, fraction=fraction, ref=ref)


def nominal_center_frequency(frequency=None, fraction=1, n=None):
    """Nominal center frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :returns: The nominal center frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.nominal_center_frequency`

    .. note:: Contrary to the other functions this function silently assumes 1000 Hz reference frequency.

    """
    center = exact_center_frequency(frequency, fraction, n)
    return acoustics.standards.iec_61260_1_2014.nominal_center_frequency(center, fraction)


def lower_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Lower band-edge frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :returns: Lower band-edge frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.lower_frequency`

    """
    center = exact_center_frequency(frequency, fraction, n, ref=ref)
    return acoustics.standards.iec_61260_1_2014.lower_frequency(center, fraction)


def upper_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Upper band-edge frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :returns: Upper band-edge frequency for the given frequency or band index.

    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`acoustics.standards.iec_61260_1_2014.upper_frequency`

    """
    center = exact_center_frequency(frequency, fraction, n, ref=ref)
    return acoustics.standards.iec_61260_1_2014.upper_frequency(center, fraction)


#-- things below will be deprecated?---#

frequency_of_band = acoustics.standards.iec_61260_1_2014.exact_center_frequency
band_of_frequency = index_of_frequency


class Octave:
    """
    Class to calculate octave center frequencies.
    """

    def __init__(self, fraction=1, interval=None, fmin=None, fmax=None, unique=False, reference=REFERENCE):

        self.reference = reference
        """
        Reference center frequency :math:`f_{c,0}`.
        """

        self.fraction = fraction
        """
        Fraction of octave.
        """

        if (interval is not None) and (fmin is not None or fmax is not None):
            raise AttributeError

        self._interval = np.asarray(interval)
        """Interval"""

        self._fmin = fmin
        """Minimum frequency of a range."""

        self._fmax = fmax
        """Maximum frequency of a range."""

        self.unique = unique
        """Whether or not to calculate the requested values for every value of ``interval``."""

    @property
    def fmin(self):
        """Minimum frequency of an interval."""
        if self._fmin is not None:
            return self._fmin
        elif self._interval is not None:
            return self.interval.min()
        else:
            raise ValueError("Incorrect fmin/interval")

    @fmin.setter
    def fmin(self, x):
        if self.interval is not None:
            pass  # Warning, remove interval first.
        else:
            self._fmin = x

    @property
    def fmax(self):
        """Maximum frequency of an interval."""
        if self._fmax is not None:
            return self._fmax
        elif self._interval is not None:
            return self.interval.max()
        else:
            raise ValueError("Incorrect fmax/interval")

    @fmax.setter
    def fmax(self, x):
        if self.interval is not None:
            pass
        else:
            self._fmax = x

    @property
    def interval(self):
        """Interval."""
        return self._interval

    @interval.setter
    def interval(self, x):
        if self._fmin or self._fmax:
            pass
        else:
            self._interval = np.asarray(x)

    def _n(self, f):
        """
        Calculate the band ``n`` from a given frequency.

        :param f: Frequency

        See also :func:`band_of_frequency`.
        """
        return band_of_frequency(f, fraction=self.fraction, ref=self.reference)

    def _fc(self, n):
        """
        Calculate center frequency of band ``n``.

        :param n: band ``n`.

        See also :func:`frequency_of_band`.
        """
        return frequency_of_band(n, fraction=self.fraction, ref=self.reference)

    @property
    def n(self):
        """
        Return band ``n`` for a given frequency.
        """
        if self.interval is not None and self.unique:
            return self._n(self.interval)
        else:
            return np.arange(self._n(self.fmin), self._n(self.fmax) + 1)

    @property
    def center(self):
        r"""
        Return center frequencies :math:`f_c`.

        .. math::  f_c = f_{ref} \cdot 2^{n/N} \cdot 10^{\frac{3}{10N}}

        """
        n = self.n
        return self._fc(n)

    @property
    def bandwidth(self):
        """
        Bandwidth of bands.

        .. math:: B = f_u - f_l

        """
        return self.upper - self.lower

    @property
    def lower(self):
        r"""
        Lower frequency limits of bands.

        .. math:: f_l = f_c \cdot 2^{\frac{-1}{2N}}

        See also :func:`lower_frequency`.
        """
        return lower_frequency(self.center, self.fraction)

    @property
    def upper(self):
        r"""
        Upper frequency limits of bands.

        .. math:: f_u = f_c \cdot 2^{\frac{+1}{2N}}

        See also :func:`upper_frequency`.
        """
        return upper_frequency(self.center, self.fraction)


__all__ = [
    'exact_center_frequency',
    'nominal_center_frequency',
    'lower_frequency',
    'upper_frequency',
    'index_of_frequency',
    'Octave',
    'frequency_of_band',
    'band_of_frequency',  # These three will be deprecated?
]
