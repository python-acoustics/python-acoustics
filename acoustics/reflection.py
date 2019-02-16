"""
Reflection
==========

The reflection module contains functions for calculating reflection factors and impedances.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc  # pylint: disable=no-name-in-module

SPECIFIC_HEAT_RATIO = 1.4
r"""Specific heat ratio of air :math:`\gamma`."""
POROSITY_DECREASE = 120.0
r"""Rate of exponential decrease of porosity with depth :math:`\alpha`."""
SOUNDSPEED = 343.0
"""Speed of sound in air :math:`c`."""
DENSITY = 1.296
r"""Density of air :math:`\rho`."""


class Boundary:
    """
    An object describing a boundary.
    """

    def __init__(  # pylint: disable=too-many-instance-attributes
            self,
            frequency,
            flow_resistivity,
            density=DENSITY,
            soundspeed=SOUNDSPEED,
            porosity_decrease=POROSITY_DECREASE,
            specific_heat_ratio=SPECIFIC_HEAT_RATIO,
            angle=None,
            distance=None,
            impedance_model='db',
            reflection_model='plane',
    ):

        self.frequency = frequency
        """
        Frequency. Single value or vector for a frequency range.
        """

        self.flow_resistivity = flow_resistivity
        r"""
        Flow resistivity :math:`\sigma`.
        """

        self.density = density
        r"""
        Density of air :math:`\rho`.

        .. note::

            This value is only required for when calculating the impedance according to Attenborough's model.
            See :func:`impedance_attenborough`.

        """

        self.soundspeed = soundspeed
        """
        Speed of sound in air :math:`c`.

        .. note::

            This value is required when calculating the impedance according to Attenborough's model or when
            calculating the spherical wave reflection factor. See respectively :func:`impedance_attenborough`
            and :func:`reflection_factor_spherical_wave`.

        """

        self.porosity_decrease = porosity_decrease
        r"""
        Rate of exponential decrease of porosity with depth :math:`\alpha`.

        .. note::

            This value is only required for when calculating the impedance according to Attenborough's model.
            See :func:`impedance_attenborough`.

        """

        self.specific_heat_ratio = specific_heat_ratio
        r"""
        Ratio of specific heats :math:`\gamma` for air.

        .. note::

            This value is only required for when calculating the impedance according to Attenborough's model.
            See :func:`impedance_attenborough`.

        """

        self.angle = angle
        r"""
        Angle of incidence :math:`\theta`.

        """

        self.distance = distance
        """
        Path length of the reflected ray :math:`r`.

        .. note::

            This value is only required for when calculating the spherical wave reflection factor.
            See :func:`reflection_factor_spherical_wave`.

        """

        self.impedance_model = impedance_model
        """
        Impedance model.

        Possibilities are ``'db'`` and ``'att'`` for respectively :func:`impedance_delany_and_bazley` and :func:`impedance_attenborough`.
        """

        self.reflection_model = reflection_model
        """
        Reflection factor model.

        Possibilities are ``'plane'`` and ``'spherical'``` for respectively :func:`reflection_factor_plane_wave` and :func:`reflection_factor_spherical_wave`.
        """

    @property
    def wavenumber(self):
        r"""
        Wavenumber :math:`k`.

        .. math:: k = \frac{2 \pi f}{c}
        """
        return 2.0 * np.pi * self.frequency / self.soundspeed

    @property
    def impedance(self):
        """
        Impedance according to chosen impedance model defined using :meth:`impedance_model`.
        """
        if self.impedance_model == 'db':
            return impedance_delany_and_bazley(self.frequency, self.flow_resistivity)
        if self.impedance_model == 'att':
            return impedance_attenborough(
                self.frequency,
                self.flow_resistivity,
                self.density,
                self.soundspeed,
                self.porosity_decrease,
                self.specific_heat_ratio,
            )
        else:
            raise ValueError("Incorrect impedance model.")

    @property
    def reflection_factor(self):
        """
        Reflection factor according to chosen reflection factor model defined using :meth:`reflection_model`.
        """
        if self.angle is None:
            raise AttributeError('Cannot calculate reflection factor. self.angle has not been specified.')

        if self.reflection_model == 'plane':
            return reflection_factor_plane_wave(*np.meshgrid(self.impedance, self.angle))
        elif self.reflection_model == 'spherical':
            if self.distance is None:
                raise AttributeError('Cannot calculate reflection factor. self.distance has not been specified.')
            else:
                return reflection_factor_spherical_wave(
                    *np.meshgrid(self.impedance, self.angle),
                    distance=self.distance,
                    wavenumber=self.wavenumber,
                )
        else:
            raise RuntimeError("Oops...")

    def plot_impedance(self, filename=None):
        """
        Plot magnitude and phase of the impedance as function of frequency.
        """

        fig = plt.figure()

        ax0 = fig.add_subplot(211)
        ax0.set_title('Magnitude of impedance')
        ax0.semilogx(self.frequency, np.abs(self.impedance))
        ax0.set_xlabel(r'$f$ in Hz')
        ax0.set_ylabel(r'$\left|Z\right|$')
        ax0.grid()

        ax0 = fig.add_subplot(212)
        ax0.set_title('Angle of impedance')
        ax0.semilogx(self.frequency, np.angle(self.impedance))
        ax0.set_xlabel(r'$f$ in Hz')
        ax0.set_ylabel(r'$\angle Z$')
        ax0.grid()

        plt.tight_layout()

        if filename:
            fig.savefig(filename, transparant=True)
        return fig

    def plot_reflection_factor(self, filename=None):
        """
        Plot reflection factor.
        """

        if self.frequency is None:
            raise ValueError("No frequency specified.")
        if self.angle is None:
            raise ValueError("No angle specified.")

        try:
            n_f = len(self.frequency)
        except TypeError:
            n_f = 1
        try:
            n_a = len(self.angle)
        except TypeError:
            n_a = 1

        if n_f == 1 and n_a == 1:
            raise ValueError("Either frequency or angle needs to be a vector.")

        elif n_f == 1 or n_a == 1:
            if n_f == 1 and n_a > 1:  # Show R as function of angle for a single frequency.
                xlabel = r"$\theta$ in degrees"
            elif n_f > 1 and n_a == 1:  # Show R as function of frequency for a single angle.
                xlabel = r"$f$ in Hz"
            R = self.reflection_factor
            fig = plt.figure()

            ax0 = fig.add_subplot(211)
            ax0.set_title("Magnitude of reflection factor")
            ax0.semilogx(self.frequency, np.abs(R))
            ax0.set_xlabel(xlabel)
            ax0.set_ylabel(r'$\left|R\right|$')
            ax0.grid()

            ax1 = fig.add_subplot(212)
            ax1.set_title("Phase of reflection factor")
            ax1.semilogx(self.frequency, np.angle(R))
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(r'$\angle R$')
            ax1.grid()

        elif n_f > 1 and n_a > 1:  # Show 3D or pcolor
            R = self.reflection_factor
            fig = plt.figure()

            #grid = AxesGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1, cbar_mode='each', cbar_location='right')
            ax0 = fig.add_subplot(211)
            #ax0 = grid[0]
            ax0.set_title("Magnitude of reflection factor")
            ax0.pcolormesh(self.frequency, self.angle * 180.0 / np.pi, np.abs(R))
            #ax0.pcolor(self.angle, self.frequency, np.abs(R))
            #ax0.set_xlabel(xlabel)
            #ax0.set_ylabel(r'$\left|R\right|$')
            ax0.grid()

            ax1 = fig.add_subplot(212)
            #ax1 = grid[1]
            ax1.set_title("Phase of reflection factor")
            ax1.pcolormesh(self.frequency, self.angle * 180.0 / np.pi, np.angle(R))
            #ax1.pcolor(self.angle, self.frequency, np.angle(R))
            #ax0.set_xlabel(xlabel)
            #ax0.set_ylabel(r'$\angle R$')
            ax1.grid()

        else:
            raise RuntimeError("Oops...")

        #plt.tight_layout()

        if filename:
            fig.savefig(filename, transparant=True)
        else:
            return fig


def reflection_factor_plane_wave(impedance, angle):
    r"""
    Plane wave reflection factor :math:`R`.

    :param impedance: Normalized impedance :math:`Z`.
    :param angle: Angle of incidence :math:`\theta`.


    The plane wave reflection factor :math:`R` is given by

    .. math:: R = \frac{Z\cos{\theta}-1}{Z\cos{\theta}+1}

    where :math:`Z` is the normalized impedance and :math:`\theta` the angle of incidence.

    """
    return (impedance * np.cos(angle) - 1.0) / (impedance * np.cos(angle) + 1.0)


def numerical_distance(impedance, angle, distance, wavenumber):
    r"""
    Numerical distance :math:`w`.

    :param impedance: Normalized impedance :math:`Z`.
    :param angle: Angle of incidence :math:`\theta`.
    :param distance: Path length of the reflected ray :math:`r`.
    :param wavenumber: Wavenumber :math:`k`.

    The numerical distance :math:`w` is given by

    .. math:: w = \sqrt{-j k r  \left( 1 + \frac{1}{Z} \cos{\theta} - \sqrt{1 - \left( \frac{1}{Z} \right)^2} \sin{\theta} \right) }

    """
    return np.sqrt(-1j * wavenumber * distance *
                   (1.0 + 1.0 / impedance * np.cos(angle) - np.sqrt(1.0 - (1.0 / impedance)**2.0) * np.sin(angle)))


def reflection_factor_spherical_wave(impedance, angle, distance, wavenumber):
    r"""
    Spherical wave reflection factor :math:`Q`.

    :param impedance: Normalized impedance :math:`Z`.
    :param angle: Angle of incidence :math:`\theta`.
    :param distance: Path length of the reflected ray :math:`r`.
    :param wavenumber: Wavenumber :math:`k`.

    The spherical wave relfection factor :math:`Q` is given by

    .. math:: Q = R \left(1 - R \right) F

    where :math:`R` is the plane wave reflection factor as calculated in :func:`reflection_factor_plane_wave` and :math:`F` is given by

    .. math:: F = 1 - j \sqrt{ \pi} w e^{-w^2} \mathrm{erfc} \left( j w \right)

    where :math:`w` is the numerical distance as calculated in :func:`numerical_distance`.
    """
    w = numerical_distance(impedance, angle, distance, wavenumber)

    F = 1.0 - 1j * np.sqrt(np.pi) * w * np.exp(-w**2.0) * erfc(1j * w)

    plane_factor = reflection_factor_plane_wave(impedance, angle)
    return plane_factor * (1.0 - plane_factor) * F


def impedance_delany_and_bazley(frequency, flow_resistivity):
    r"""
    Normalised specific acoustic impedance according to the empirical one-parameter model by Delany and Bazley.

    :param frequency: Frequency :math:`f`.
    :param flow_resistivity: Flow resistivity :math:`\sigma`.

    The impedance :math:`Z` is given by

    .. math:: Z = 1 + 9.08 \left( \frac{1000f}{\sigma}\right)^{-0.75} - 11.9 j \left( \frac{1000f}{\sigma}\right)^{-0.73}

    """
    return 1.0 + 9.08 * (1000.0 * frequency / flow_resistivity)**(-0.75) - 1j * 11.9 * (
        1000.0 * frequency / flow_resistivity)**(-0.73)


def impedance_attenborough(
        frequency,
        flow_resistivity,
        density=DENSITY,
        soundspeed=SOUNDSPEED,
        porosity_decrease=POROSITY_DECREASE,
        specific_heat_ratio=SPECIFIC_HEAT_RATIO,
):
    r"""
    Normalised specific acoustics impedance according to the two-parameter model by Attenborough.

    :param frequency: Frequency :math:`f`.
    :param flow_resistivity: Flow resistivity :math:`\sigma`.
    :param soundspeed: Speed of sound in air :math:`c`.
    :param density: Density of air :math:`\rho`.
    :param porosity_decrease: Rate of exponential decrease of porosity with depth :math:`\alpha`.
    :param specific_heat_ratio: Ratio of specific heats :math:`\gamma` for air.


    The impedance :math:`Z` is given by

    .. math:: Z = \frac{\left( 1-j\right) \sqrt{\sigma/f}}{\sqrt{\pi \gamma_0 \rho_0}} - \frac{jc\alpha}{8 \pi \gamma_0 f}

    """
    return (1.0 - 1j) * np.sqrt(flow_resistivity / frequency) / np.sqrt(
        np.pi * specific_heat_ratio * density) - 1j * soundspeed * porosity_decrease / (
            8.0 * np.pi * specific_heat_ratio * frequency)
