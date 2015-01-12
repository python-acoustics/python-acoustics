"""
Atmosphere
==========

The atmosphere module contains functions and classes related to atmospheric acoustics and is based on :mod:`acoustics.standards.iso_9613_1_1993`.

From ISO 9613-1 1993
********************

Constants
---------

.. autoattribute:: acoustics.standards.iso_9613_1_1993.SOUNDSPEED
.. autoattribute:: acoustics.standards.iso_9613_1_1993.REFERENCE_TEMPERATURE
.. autoattribute:: acoustics.standards.iso_9613_1_1993.REFERENCE_PRESSURE
.. autoattribute:: acoustics.standards.iso_9613_1_1993.TRIPLE_TEMPERATURE

Functions
---------

.. autofunction:: acoustics.standards.iso_9613_1_1993.soundspeed
.. autofunction:: acoustics.standards.iso_9613_1_1993.saturation_pressure
.. autofunction:: acoustics.standards.iso_9613_1_1993.molar_concentration_water_vapour
.. autofunction:: acoustics.standards.iso_9613_1_1993.relaxation_frequency_nitrogen
.. autofunction:: acoustics.standards.iso_9613_1_1993.relaxation_frequency_oxygen
.. autofunction:: acoustics.standards.iso_9613_1_1993.attenuation_coefficient


Atmosphere class
****************

"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

try:
    from pyfftw.interfaces.numpy_fft import ifft       # Performs much better than numpy's fftpack
except ImportError:                                    # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import ifft

from acoustics.standards.iso_9613_1_1993 import *
                                                

class Atmosphere(object):
    """
    Class describing atmospheric conditions.
    """
    
    REF_TEMP = 293.15
    """Reference temperature"""
    
    REF_PRESSURE = 101.325
    """International Standard Atmosphere in kilopascal"""
    
    TRIPLE_TEMP = 273.16
    """Triple point isotherm temperature."""

    def __init__(self, 
                 temperature=REFERENCE_TEMPERATURE, 
                 pressure=REFERENCE_PRESSURE,
                 relative_humidity=0.0, 
                 reference_temperature=REFERENCE_TEMPERATURE,
                 reference_pressure=REFERENCE_PRESSURE,
                 triple_temperature=TRIPLE_TEMPERATURE):
        """

        :param temperature: Temperature
        :param pressure: Pressure
        :param relative_humidity: Relative humidity
        :param reference_temperature: Reference temperature.
        :param reference_pressure: Reference pressure.
        :param triple_temperature: Triple temperature.
        """
        
        self.temperature = temperature
        """Ambient temperature :math:`T`."""
        
        self.pressure = pressure
        """Ambient pressure :math:`p_a`."""
        
        self.relative_humidity = relative_humidity
        """Relative humidity"""
        
        self.reference_temperature = reference_temperature
        """
        Reference temperature.
        """
        
        self.reference_pressure = reference_pressure
        """
        Reference pressure.
        """
        
        self.triple_temperature = triple_temperature
        """
        Triple temperature.
        """
    
    @property
    def soundspeed(self):
        """
        Speed of sound :math:`c` calculated using :func:`soundspeed`.
        """
        return soundspeed(self.temperature, self.reference_temperature)
        
    @property
    def saturation_pressure(self):
        """
        Saturation pressure :math:`p_{sat}` calculated using :func:`acoustics.standards.iso_9613_1_1993.saturation_pressure`.
        """
        return saturation_pressure(self.temperature, self.reference_pressure, self.triple_temperature)
    
    @property
    def molar_concentration_water_vapour(self):
        """
        Molar concentration of water vapour :math:`h` calculated using :func:`molar_concentration_water_vapour`.
        """
        return molar_concentration_water_vapour(self.relative_humidity, self.saturation_pressure, self.pressure)
        
    @property
    def relaxation_frequency_nitrogen(self):
        """
        Resonance frequency of nitrogen :math:`f_{r,N}` calculated using :func:`relaxation_frequency_nitrogen`.
        """
        return relaxation_frequency_nitrogen(self.pressure, self.temperature, self.molar_concentration_water_vapour, self.reference_pressure, self.reference_temperature)
    
    @property
    def relaxation_frequency_oxygen(self):
        """
        Resonance frequency of oxygen :math:`f_{r,O}` calculated using :func:`relaxation_frequency_oxygen`.
        """
        return relaxation_frequency_oxygen(self.pressure, self.molar_concentration_water_vapour, self.reference_pressure)
    
    def attenuation_coefficient(self, frequency):
        """
        Attenuation coefficient :math:`\\alpha` describing atmospheric absorption in dB/m as function of ``frequency``.
        
        :param frequency: Frequencies to be considered.
        """
        return attenuation_coefficient(self.pressure, self.temperature, self.reference_pressure, self.reference_temperature, self.relaxation_frequency_nitrogen, self.relaxation_frequency_oxygen, frequency)
        
    ###def atmospheric_absorption(self, signal):
        ###"""
        ###Calculate the signal for the atmospheric absorption.
        
        ###:param signal:
        ###:type signal: `Auraliser.Signal.Signal`
        ###"""
        
        ###"""Octaves object"""
        ###o = Acoustics.Octave.Octave(fmin=1.0, fmax=signal.sample_frequency/2.0, order=24)
        ###frequencies = o.center()
        
        ###t = np.arange(0, len(signal))               # Time signal
        
        ###alpha = self.atmosphere.attenuation_coefficient(frequencies)
        
        ###A = 10.0**(alpha/10.0)      # Convert from decibel/m to linear/m
        
        ###A *= o.bandwidth()          # Integrate over the frequency band
        
        ###phi = np.random.randn(len(frequencies))     # Random phase
        ####phi = np.zeros(len(frequencies))
        
        ###d = self.geometry.distance  # Distance. Final Multiplier for signal strength.
        
        ###print  A * np.sin(2.0 * np.pi * np.outer(t, frequencies)+ phi)
        
        ###"""Absorption time signal."""
        ###s = d * np.sum( A * np.sin(2.0 * np.pi * np.outer(t, frequencies)+ phi))
        
        ####s /= np.sum(o.bandwidth())

        ###return s
    
    ##def absorption_coefficient(self, frequencies):
        ##"""
        ##Calculate the absorption coefficient in dB/m for the given frequencies.
        
        ##According to ISO9613-1:1993.
        
        ##:param frequencies: Frequencies to calculate alpha for.
        ##:type frequencies: :class:`np.ndarray`
        ##"""
        
        ###T = np.array(self.temperature, dtype='float128')                        # Ambient temperature
        ###p_a = np.array(self.pressure, dtype='float128')                         # Ambient pressure
        ###h = np.array(self.molar_concentration_water_vapour, dtype='float128')   # Ambient molar...
        
        ###p_r = np.array(self.REF_PRESSURE, dtype='float128')                     # Reference pressure
        ###T0 = np.array(self.REF_TEMP, dtype='float128')                          # Reference temperature
        
        ###f = np.array(frequencies, dtype='float128')
        
        
        ##T = self.temperature                        # Ambient temperature
        ##p_a = self.pressure                         # Ambient pressure
        ##h = self.molar_concentration_water_vapour   # Ambient molar...
        
        ##p_r = self.REF_PRESSURE                     # Reference pressure
        ##T0 = self.REF_TEMP                          # Reference temperature
        
        ##f = frequencies
        
        
        ##"""Relaxation frequency of oxygen."""
        ##f_rO = p_a / p_r * ( 24.0 + 4.04 * 10.0**4.0 * h * (0.02 + h) / (0.391 + h) )
        
        ##"""Relaxation frequency of nitrogen."""
        ##f_rN = p_a / p_r * (T/T0)**(-0.5) * (9.0 + 280.0 * h * np.exp(-4.170 * ((T/T0)**(-1.0/3.0) - 1.0 ) ) )
        
        ##alpha = 8.686 * f**2.0 * ( ( 1.84 * 10.0**(-11.0) * (p_r/p_a) * (T/T0)**(0.5)) + (T/T0)**(-2.5) * ( 0.01275 * np.exp(-2239.1 / T) * (f_rO + (f**2.0/f_rO))**(-1.0) + 0.1068 * np.exp(-3352.0/T) * (f_rN + (f**2.0/f_rN))**(-1.0) ) )
        
        ##return alpha
    
    def ir_attenuation_coefficient(self, d, fs=44100, N=2048, sign=+1):
        """
        Calculate the impulse response due to air absorption.
        
        :param fs: Sample frequency
        :param d: Distance
        :param N: Blocks
        :param sign: Multiply (+1) or divide (-1) by transfer function. Multiplication is used for applying the absorption while -1 is used for undoing the absorption.
        """ 
        
        d = d if isinstance(d, np.ndarray) else np.array([d])
        
        f = np.linspace(0.0, fs/2.0, N/2.0) # Frequency vector. A bin for every signal sample.
        
        tf = np.zeros((len(d), len(f)), dtype='complex')                          # Transfer function needs to be complex, and same size.
        tf += 10.0**( float(sign) * d.reshape((-1,1)) * self.attenuation_coefficient(f) / 20.0  )  # Calculate the actual transfer function.
        
        #print('TF: ' + str(tf.shape))
        
        #tf = np.concatenate( ( tf, np.conj(tf[::-1]) ))                 
        tf = np.hstack( (tf, np.conj(tf[::-1, :]))) # Positive frequencies first, and then mirrored the conjugate negative frequencies.
        
        #print('TF reshaped: ' + str(tf.shape))
        
        #n = 2**int(np.ceil(np.log2(len(tf))))   # Blocksize for the IFFT. Zeros are padded.

        ir = ifft( tf , n=N)     # Obtain the impulse response through the IFFT.
        
        ir = np.hstack((ir[:, N/2:N], ir[:, 0:N/2]))
        
        #ir = np.real(ir[0:N/2])
        
        ir = np.real(ir).T
        
        return ir   # Note that the reduction is a factor two too much! Too much energy loss now that we use a double-sided spectrum.
    
    def plot_ir_attenuation_coefficient(self, fs, N, d, filename=None):
        """
        Plot the impulse response of the attenuation due to atmospheric absorption.
        The impulse response is calculated using :meth:`ir_attenuation_coefficient`.
        
        :param filename: Filename
        :param fs: Sample frequency
        :param N: Blocks
        :param d: Distance
        
        """
        fig = plt.figure()
        
        ax0 = fig.add_subplot(111)
        ax0.set_title('Impulse response atmospheric attenuation')
        
        ir = self.ir_attenuation_coefficient(fs, N, d)
        
        xsignal = np.arange(0.0, len(ir)) / fs
        ax0.plot(xsignal, ir)
        ax0.set_xlabel(r'$t$ in s')
        ax0.set_ylabel(r'Some')
        ax0.set_yscale('log')
        ax0.grid()
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()
    
    
    
    def plot_attenuation_coefficient(self, frequency):
        """
        Plot the attenuation coefficient :math:`\\alpha` as function of frequency and write the figure to ``filename``.
        
        :param filename: Filename
        :param frequency: Frequencies
        
        .. note:: The attenuation coefficient is plotted in dB/km!
        
        """
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        
        ax0.plot(frequency, self.attenuation_coefficient(frequency)*1000.0)
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.set_xlabel(r'$f$ in Hz')
        ax0.set_ylabel(r'$\alpha$ in dB/km')
        
        ax0.legend()
        ax0.grid()
        
        return fig
