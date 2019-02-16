"""
Doppler
=======

Doppler shift module.
"""
SOUNDSPEED = 343.0
"""Speed of sound
"""


def velocity_from_doppler_shift(f1, f2, c=SOUNDSPEED):
    r"""Calculate velocity based on measured frequency shifts due to Doppler shift.

    :param c: Speed of sound :math:`c`.
    :param f1: Lower frequency :math:`f_1`.
    :param f2: Upper frequency :math:`f_2`.

    .. math:: v = c \cdot \left( \frac{f_2 - f_1}{f_2 + f_1} \right)

    The assumption is made that the velocity is constant between the observation times.

    """
    return c * (f2 - f1) / (f2 + f1)


def frequency_shift(frequency, velocity_source, velocity_receiver, soundspeed=SOUNDSPEED):
    r"""Frequency shift due to Doppler effect.

    :param frequency: Emitted frequency :math:`f`.
    :param velocity_source: Velocity of source :math:`v_s`.
        Positive if the source is moving away from the receiver (and negative in the other direction).
    :param velocity_receiver: Velocity of receiver :math:`v_r`.
        Positive if the receiver is moving towards the source (and negative in the other direction);
    :param soundspeed: Speed of sound :math:`c`.

    .. math:: f = \frac{c + v_r}{c + v_s} f_0

    """
    return (soundspeed + velocity_receiver) / (soundspeed + velocity_source) * frequency
