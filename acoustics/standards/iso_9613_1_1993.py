"""
ISO 9613-1:1993
===============

ISO 9613-1:1993 specifies an analytical method of calculating the attenuation of sound
as a result of atmospheric absorption for a variety of meteorological conditions.

"""

import numpy as np

SOUNDSPEED = 343.2
"""
Speed of sound.
"""

REFERENCE_TEMPERATURE = 293.15
"""
Reference temperature.
"""

REFERENCE_PRESSURE = 101.325
"""
International Standard Atmosphere in kilopascal.
"""

TRIPLE_TEMPERATURE = 273.16
""".
Triple point isotherm temperature.
"""


def soundspeed(temperature, reference_temperature=REFERENCE_TEMPERATURE):
    """
    Speed of sound :math:`c`.

    :param temperature: Ambient temperature :math:`T_0`
    :param reference_temperature: Reference temperature :math:`T`


    The speed of sound is calculated using

    ..  math:: c = 343.2 \\left( \\frac{T}{T_0} \\right)

    """
    return 343.2 * np.sqrt(temperature / reference_temperature)


def saturation_pressure(temperature, reference_pressure=REFERENCE_PRESSURE, triple_temperature=TRIPLE_TEMPERATURE):
    """
    Saturation vapour pressure :math:`p_{sat}`.

    :param temperature: Ambient temperature :math:`T`
    :param reference_pressure: Reference pressure :math:`p_r`
    :param triple_temperature: Triple point temperature water :math:`T_{01}`

    The saturation vapour pressure is calculated using

    .. math:: p_{sat} = 10^C \cdot p_r

    with exponent :math:`C` given by

    .. math:: C = -6.8346 \cdot \\left( \\frac{T_{01}}{T} \\right)^{1.261}  + 4.6151

    """
    return reference_pressure * 10.0**(-6.8346 * (triple_temperature / temperature)**(1.261) + 4.6151)


def molar_concentration_water_vapour(relative_humidity, saturation_pressure, pressure):
    """
    Molar concentration of water vapour :math:`h`.

    :param relative_humidity: Relative humidity :math:`h_r`
    :param saturation_pressure: Saturation pressure :math:`p_{sat}`
    :param pressure: Ambient pressure :math:`p`

    The molar concentration of water vapour is calculated using

    .. math:: h = h_r  \\frac{p_{sat}}{p_a}

    """
    return relative_humidity * saturation_pressure / pressure


def relaxation_frequency_oxygen(pressure, h, reference_pressure=REFERENCE_PRESSURE):
    """
    Relaxation frequency of oxygen :math:`f_{r,O}`.

    :param pressure: Ambient pressure :math:`p_a`
    :param reference_pressure: Reference pressure :math:`p_r`
    :param h: Molar concentration of water vapour :math:`h`

    The relaxation frequency of oxygen is calculated using

    .. math:: f_{r,O} = \\frac{p_a}{p_r} \\left( 24 + 4.04 \cdot 10^4 h \\frac{0.02 + h}{0.391 + h}  \\right)

    """
    return pressure / reference_pressure * (24.0 + 4.04 * 10.0**4.0 * h * (0.02 + h) / (0.391 + h))


def relaxation_frequency_nitrogen(pressure, temperature, h, reference_pressure=REFERENCE_PRESSURE,
                                  reference_temperature=REFERENCE_TEMPERATURE):
    """
    Relaxation frequency of nitrogen :math:`f_{r,N}`.

    :param pressure: Ambient pressure :math:`p_a`
    :param temperature: Ambient temperature :math:`T`
    :param h: Molar concentration of water vapour :math:`h`
    :param reference_pressure: Reference pressure :math:`p_{ref}`
    :param reference_temperature: Reference temperature :math:`T_{ref}`

    The relaxation frequency of nitrogen is calculated using

    .. math:: f_{r,N} = \\frac{p_a}{p_r} \\left( \\frac{T}{T_0} \\right)^{-1/2} \cdot \\left( 9 + 280 h \exp{\\left\{ -4.170 \\left[ \\left(\\frac{T}{T_0} \\right)^{-1/3} -1 \\right] \\right\} } \\right)

    """
    return pressure / reference_pressure * (temperature / reference_temperature)**(-0.5) * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((temperature / reference_temperature)**(-1.0 / 3.0) - 1.0)))


def attenuation_coefficient(pressure, temperature, reference_pressure, reference_temperature,
                            relaxation_frequency_nitrogen, relaxation_frequency_oxygen, frequency):
    """
    Attenuation coefficient :math:`\\alpha` describing atmospheric absorption in dB/m for the specified ``frequency``.

    :param pressure: Ambient pressure :math:`T`
    :param temperature: Ambient temperature :math:`T`
    :param reference_pressure: Reference pressure :math:`p_{ref}`
    :param reference_temperature: Reference temperature :math:`T_{ref}`
    :param relaxation_frequency_nitrogen: Relaxation frequency of nitrogen :math:`f_{r,N}`.
    :param relaxation_frequency_oxygen: Relaxation frequency of oxygen :math:`f_{r,O}`.
    :param frequency: Frequencies to calculate :math:`\\alpha` for.

    """
    return 8.686 * frequency**2.0 * (
        (1.84 * 10.0**(-11.0) * (reference_pressure / pressure) * (temperature / reference_temperature)**
         (0.5)) + (temperature / reference_temperature)**
        (-2.5) * (0.01275 * np.exp(-2239.1 / temperature) * (relaxation_frequency_oxygen +
                                                             (frequency**2.0 / relaxation_frequency_oxygen))**
                  (-1.0) + 0.1068 * np.exp(-3352.0 / temperature) *
                  (relaxation_frequency_nitrogen + (frequency**2.0 / relaxation_frequency_nitrogen))**(-1.0)))
