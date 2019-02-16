"""
Descriptors
============

The descriptors module offers all kinds of acoustics related descriptors.

.. toctree::
    :maxdepth: 2

Descriptors from ISO/TR 25417:2007
**********************************

Descriptors from :mod:`acoustics.standards.iso_tr_25417_2007`.

.. autoattribute:: acoustics.descriptors.REFERENCE_PRESSURE
.. autofunction:: acoustics.descriptors.sound_pressure_level
.. autofunction:: acoustics.descriptors.equivalent_sound_pressure_level
.. autofunction:: acoustics.descriptors.peak_sound_pressure
.. autofunction:: acoustics.descriptors.peak_sound_pressure_level
.. autoattribute:: acoustics.descriptors.REFERENCE_SOUND_EXPOSURE
.. autofunction:: acoustics.descriptors.sound_exposure
.. autofunction:: acoustics.descriptors.sound_exposure_level
.. autoattribute:: acoustics.descriptors.REFERENCE_POWER
.. autofunction:: acoustics.descriptors.sound_power_level
.. autofunction:: acoustics.descriptors.sound_energy
.. autofunction:: acoustics.descriptors.sound_energy_level
.. autoattribute:: acoustics.descriptors.REFERENCE_ENERGY
.. autofunction:: acoustics.descriptors.sound_intensity
.. autofunction:: acoustics.descriptors.time_averaged_sound_intensity
.. autoattribute:: acoustics.descriptors.REFERENCE_INTENSITY
.. autofunction:: acoustics.descriptors.time_averaged_sound_intensity_level
.. autofunction:: acoustics.descriptors.normal_time_averaged_sound_intensity
.. autofunction:: acoustics.descriptors.normal_time_averaged_sound_intensity_level


Other descriptors
*****************

"""
import numpy as np

from acoustics.standards.iso_tr_25417_2007 import (
    REFERENCE_PRESSURE,
    sound_pressure_level,
    equivalent_sound_pressure_level,
    peak_sound_pressure,
    peak_sound_pressure_level,
    REFERENCE_SOUND_EXPOSURE,
    sound_exposure,
    sound_exposure_level,
    REFERENCE_POWER,
    sound_power_level,
    sound_energy,
    REFERENCE_ENERGY,
    sound_energy_level,
    sound_intensity,
    time_averaged_sound_intensity,
    REFERENCE_INTENSITY,
    time_averaged_sound_intensity_level,
    normal_time_averaged_sound_intensity,
    normal_time_averaged_sound_intensity_level,
)

from acoustics.standards.iso_1996_1_2003 import composite_rating_level


def _leq(levels, time):
    levels = np.asarray(levels)
    return 10.0 * np.log10((1.0 / time) * np.sum(10.0**(levels / 10.0)))


def leq(levels, int_time=1.0):
    """
    Equivalent level :math:`L_{eq}`.

    :param levels: Levels as function of time.
    :param int_time: Integration time. Default value is 1.0 second.
    :returns: Equivalent level L_{eq}.

    Sum of levels in dB.
    """
    levels = np.asarray(levels)
    time = levels.size * int_time
    return _leq(levels, time)


def sel(levels):
    """
    Sound Exposure Level from ``levels`` (NumPy array).
    """
    levels = np.asarray(levels)
    return _leq(levels, 1.0)


def lw(W, Wref=1.0e-12):
    """
    Sound power level :math:`L_{w}` for sound power :math:`W` and reference power :math:`W_{ref}`.

    :param W: Sound power :math:`W`.
    :param Wref: Reference power :math:`W_{ref}`. Default value is :math:`10^{12}` watt.
    """
    W = np.asarray(W)
    return 10.0 * np.log10(W / Wref)


def lden(lday, levening, lnight, hours=(12.0, 4.0, 8.0), adjustment=(0.0, 5.0, 10.0)):
    """
    Calculate :math:`L_{den}` from :math:`L_{day}`, :math:`L_{evening}` and :math:`L_{night}`.

    :param lday: Equivalent level during day period :math:`L_{day}`.
    :param levening: Equivalent level during evening period :math:`L_{evening}`.
    :param lnight: Equivalent level during night period :math:`L_{night}`.
    :param hours: Hours per period.
    :param adjustment: Correction factor per period.
    :returns: :math:`L_{den}`

    .. seealso:: :func:`acoustics.standards.iso_1996_1_2003.composite_rating_level`
    """
    lday = np.asarray(lday)
    levening = np.asarray(levening)
    lnight = np.asarray(lnight)
    return composite_rating_level(np.vstack((lday, levening, lnight)).T, hours, adjustment)


def ldn(lday, lnight, hours=(15.0, 9.0), adjustment=(0.0, 10.0)):
    """
    Calculate :math:`L_{dn}` from :math:`L_{day}` and :math:`L_{night}`.

    :param lday: Equivalent level during day period :math:`L_{day}`.
    :param lnight: Equivalent level during night period :math:`L_{night}`.
    :param hours: Hours per period.
    :param adjustment: Correction factor per period.
    :returns: :math:`L_{dn}`

    .. seealso:: :func:`acoustics.standards.iso_1996_1_2003.composite_rating_level`
    """
    lday = np.asarray(lday)
    lnight = np.asarray(lnight)
    return composite_rating_level(np.vstack((lday, lnight)).T, hours, adjustment)


__all__ = [
    # Following we take from another module
    "REFERENCE_PRESSURE",
    "sound_pressure_level",
    "equivalent_sound_pressure_level",
    "peak_sound_pressure",
    "peak_sound_pressure_level",
    "REFERENCE_SOUND_EXPOSURE",
    "sound_exposure",
    "sound_exposure_level",
    "REFERENCE_POWER",
    "sound_power_level",
    "sound_energy",
    "REFERENCE_ENERGY",
    "sound_energy_level",
    "sound_intensity",
    "time_averaged_sound_intensity",
    "REFERENCE_INTENSITY",
    "time_averaged_sound_intensity_level",
    "normal_time_averaged_sound_intensity",
    "normal_time_averaged_sound_intensity_level",
    # Following are locally defined
    "leq",
    "sel",
    "lw",
    "lden",
    "ldn",
]
