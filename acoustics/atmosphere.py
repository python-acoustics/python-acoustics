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

        :param temperature: Temperature in kelvin
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
        
        
    #def ir_attenuation_coefficient(self, distances, fs=44100.0, n_blocks=2048, sign=-1):
        #"""
        #Calculate the impulse response due to air absorption.
        
        #:param fs: Sample frequency
        #:param distances: Distances
        #:param blocks: Blocks
        #:param sign: Multiply (+1) or divide (-1) by transfer function. Multiplication is used for applying the absorption while -1 is used for undoing the absorption.
        #""" 
        #distances = np.atleast_1d(distances)

        #f = np.fft.rfftfreq(n_blocks, 1./fs)

        #tf = np.zeros((len(distances), len(f)), dtype='float64')                          # Transfer function needs to be complex, and same size.
        #tf += 10.0**( float(sign) * distances.reshape((-1,1)) * self.attenuation_coefficient(f) / 20.0  )  # Calculate the actual transfer function.

        #ir = np.fft.ifftshift( np.fft.irfft(tf, n=n_blocks) )
        #print(ir.shape)
        #return ir.real.T / 2.0   
    
    
    def ir_attenuation_coefficient(self, distances, fs=44100., n_blocks=2048, sign=-1):
        """
        Calculate the impulse response due to air absorption.
        
        :param fs: Sample frequency
        :param distances: Distances
        :param N: Blocks
        :param sign: Multiply (+1) or divide (-1) by transfer function. Multiplication is used for applying the absorption while -1 is used for undoing the absorption.
        """ 
        distances = np.atleast_1d(distances)
        
        f = np.fft.rfftfreq(n_blocks, 1./fs)
        
        # Transfer function needs to be complex, and same size. Complex?? Mehh
        tf = np.zeros((len(distances), len(f)), dtype='float64')
        # Calculate the actual transfer function.    
        tf += 10.0**( float(sign) * distances.reshape((-1,1)) * self.attenuation_coefficient(f) / 20.0  )
        # Positive frequencies first, and then mirrored the conjugate negative frequencies.
        tf = np.hstack( (tf, np.conj(tf[::-1, :]))) 

        # Obtain the impulse response through the IFFT.
        ir = ifft( tf , n=n_blocks)     
        ir = np.hstack((ir[:, n_blocks/2:n_blocks], ir[:, 0:n_blocks/2]))
        
        ir = np.real(ir).T

        return ir   # Note that the reduction is a factor two too much! Too much energy loss now that we use a double-sided spectrum.
    
    def plot_ir_attenuation_coefficient(self, distances, fs=44100., n_blocks=2048):
        """
        Plot the impulse response of the attenuation due to atmospheric absorption.
        The impulse response is calculated using :meth:`ir_attenuation_coefficient`.
        
        :param filename: Filename
        :param fs: Sample frequency
        :param N: Blocks
        :param d: Distance
        
        """
        ir = self.ir_attenuation_coefficient(distances, fs, n_blocks)
        
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Impulse response atmospheric attenuation')
        xsignal = np.arange(0.0, len(ir)) / fs
        ax0.plot(xsignal, ir)
        ax0.set_xlabel(r'$t$ in s')
        ax0.set_ylabel(r'Some')
        ax0.set_yscale('log')
        ax0.grid()
        return fig
    
            
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
