"""
ISO/TR 25417 2007
=================

ISO/TR 25417:2007 specifies definitions of acoustical quantities and terms used
in noise measurement documents prepared by ISO Technical Committee TC 43,
Acoustics, Subcommittee SC 1, Noise, together with their symbols and units, with
the principal aim of harmonizing the terminology used [ISO25417]_.

.. [ISO24517] http://www.iso.org/iso/home/store/catalogue_tc/catalogue_detail.htm?csnumber=42915

.. inheritance-diagram:: acoustics.standards.iso_tr_25417_2007

"""
import numpy as np

REFERENCE_PRESSURE = 2.0e-5
"""
Reference value of the sound pressure :math:`p_0` is :math:`2 \cdot 10^{-5}` Pa.
"""


def sound_pressure_level(pressure, reference_pressure=REFERENCE_PRESSURE):
    """
    Sound pressure level :math:`L_p` in dB.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param reference_pressure: Reference value :math:`p_0`.

    .. math:: L_p = 10 \\log_{10}{ \\left( \\frac{p^2}{p_0^2} \\right)}

    See section 2.2.
    """
    return 10.0 * np.log10(pressure**2.0 / reference_pressure**2.0)


def equivalent_sound_pressure_level(pressure, reference_pressure=REFERENCE_PRESSURE, axis=-1):
    """
    Time-averaged sound pressure level :math:`L_{p,T}` or equivalent-continious sound pressure level :math:`L_{p,eqT}` in dB.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param reference_pressure: Reference value :math:`p_0`.
    :param axis: Axis.

    .. math:: L_{p,T} = L_{p,eqT} = 10.0 \\log_{10}{ \\left( \\frac{\\frac{1}{T} \\int_{t_1}^{t_2} p^2 (t) \\mathrm{d} t  }{p_0^2} \\right)}

    See section 2.3.
    """
    return 10.0 * np.log10((pressure**2.0).mean(axis=axis) / reference_pressure**2.0)


def max_sound_pressure_level(pressure, reference_pressure=REFERENCE_PRESSURE, axis=-1):
    """
    Maximum time-averaged sound pressure level :math:`L_{F,max}` in dB.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param reference_pressure: Reference value :math:`p_0`.
    :param axis: Axis.

    .. math:: \mathrm{max}{(L_{p})}

    """
    return sound_pressure_level(pressure, reference_pressure=reference_pressure).max(axis=axis)


def peak_sound_pressure(pressure, axis=-1):
    """
    Peak sound pressure :math:`p_{peak}` is the greatest absolute sound pressure during a certain time interval.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param axis: Axis.

    .. math:: p_{peak} = \\mathrm{max}(|p|)

    """
    return np.abs(pressure).max(axis=axis)


def peak_sound_pressure_level(pressure, reference_pressure=REFERENCE_PRESSURE, axis=-1):
    """
    Peak sound pressure level :math:`L_{p,peak}` in dB.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param reference_pressure: Reference value :math:`p_0`.
    :param axis: Axis.

    .. math:: L_{p,peak} = 10.0 \\log \\frac{p_{peak}^2.0}{p_0^2}

    """
    return 10.0 * np.log10(peak_sound_pressure(pressure, axis=axis)**2.0 / reference_pressure**2.0)


REFERENCE_SOUND_EXPOSURE = 4.0e-10
"""
Reference value of the sound exposure :math:`E_0` is :math:`4 \cdot 10^{-12} \\mathrm{Pa}^2\\mathrm{s}`.
"""


def sound_exposure(pressure, fs, axis=-1):
    """
    Sound exposure :math:`E_{T}`.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param fs: Sample frequency :math:`f_s`.
    :param axis: Axis.

    .. math:: E_T = \\int_{t_1}^{t_2} p^2 (t) \\mathrm{d}t

    """
    return (pressure**2.0 / fs).sum(axis=axis)


def sound_exposure_level(pressure, fs, reference_sound_exposure=REFERENCE_SOUND_EXPOSURE, axis=-1):
    """
    Sound exposure level :math:`L_{E,T}` in dB.

    :param pressure: Instantaneous sound pressure :math:`p`.
    :param fs: Sample frequency :math:`f_s`.
    :param sound_exposure: Sound exposure :math:`E_{T}`.
    :param reference_sound_exposure: Reference value :math:`E_{0}`

    .. math:: L_{E,T} = 10 \\log_{10}{ \\frac{E_T}{E_0}  }

    """
    return 10.0 * np.log10(sound_exposure(pressure, fs, axis=axis) / reference_sound_exposure)


REFERENCE_POWER = 1.0e-12
"""
Reference value of the sound power :math:`P_0` is 1 pW.
"""


def sound_power_level(power, reference_power=REFERENCE_POWER):
    """
    Sound power level :math:`L_{W}`.

    :param power: Sound power :math:`P`.
    :param reference_power: Reference sound power :math:`P_0`.

    .. math:: 10 \\log_{10}{ \\frac{P}{P_0}  }

    """
    return 10.0 * np.log10(power / reference_power)


def sound_energy(power, axis=-1):
    """
    Sound energy :math:`J`..

    :param power: Sound power :math:`P`.

    .. math:: J = \\int_{t_1}^{t_2} P(t) \\mathrm{d} t
    """
    return power.sum(axis=axis)


REFERENCE_ENERGY = 1.0e-12
"""
Reference value of the sound energy :math:`J_0` is 1 pJ.
"""


def sound_energy_level(energy, reference_energy=REFERENCE_ENERGY):
    """
    Sound energy level L_{J} in dB.

    :param energy: Sound energy :math:`J`.
    :param reference_energy: Reference sound energy :math:`J_0`.

    .. math:: L_{J} = 10 \\log_{10}{ \\frac{J}{J_0} }

    """
    return np.log10(energy / reference_energy)


def sound_intensity(pressure, velocity):
    """
    Sound intensity :math:`\\mathbf{i}`.

    :param pressure: Sound pressure :math:`p(t)`.
    :param velocity: Particle velocity :math:`\\mathbf{u}(t)`.

    .. math:: \\mathbf{i} = p(t) \cdot \\mathbf{u}(t)

    """
    return pressure * velocity


REFERENCE_INTENSITY = 1.0e-12
"""
Reference value of the sound intensity :math:`I_0` is :math:`\\mathrm{1 pW/m^2}`.
"""


def time_averaged_sound_intensity(intensity, axis=-1):
    """
    Time-averaged sound intensity :math:`\\mathbf{I}_T`.

    :param intensity: Sound intensity :math:`\\mathbf{i}`.
    :param axis: Axis.

    .. math:: \\mathbf{I}_T = \\frac{1}{T} \\int_{t_1}^{t_2} \\mathbf{i}(t)

    """
    return intensity.mean(axis=axis)


def time_averaged_sound_intensity_level(time_averaged_sound_intensity, reference_intensity=REFERENCE_INTENSITY,
                                        axis=-1):
    """
    Time-averaged sound intensity level :math:`L_{I,T}`.

    :param time_averaged_sound_intensity: Time-averaged sound intensity :math:`\\mathbf{I}_T`.
    :param reference_intensity: Reference sound intensity :math:`I_0`.

    .. math:: L_{I,T} = 10 \\log_{10} { \\frac{|\\mathbf{I}_T|}{I_0} }

    """
    return 10.0 * np.log10(np.linalg.norm(time_averaged_sound_intensity, axis=axis) / reference_intensity)


def normal_time_averaged_sound_intensity(time_averaged_sound_intensity, unit_normal_vector):
    """
    Normal time-averaged sound intensity :math:`I_{n,T}`.

    :param time_averaged_sound_intensity: Time-averaged sound intensity :math:`\\mathbf{I}_T`.
    :param unit_normal_vector: Unit normal vector :math:`\\mathbf{n}`.

    .. math:: I_{n,T} = \\mathbf{I}_T \\cdot \\mathbf{n}

    """
    return time_averaged_sound_intensity.dot(unit_normal_vector)


def normal_time_averaged_sound_intensity_level(normal_time_averaged_sound_intensity,
                                               reference_intensity=REFERENCE_INTENSITY):
    """
    Normal time-averaged sound intensity level :math:`L_{In,T}` in dB.

    :param normal_time_averaged_sound_intensity: Normal time-averaged sound intensity :math:`I{n,T}`.
    :param reference_intensity: Reference sound intensity :math:`I_0`.

    .. math:: I_{n,T} = 10 \\log_{10} { \\frac{|I_{n,T}|}{I_0}}

    """
    return 10.0 * np.log10(np.abs(normal_time_averaged_sound_intensity / reference_intensity))
