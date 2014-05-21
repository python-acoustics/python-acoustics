"""
Module containing functions and a classes related to atmospheric acoustics.
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

try:
    from pyfftw.interfaces.numpy_fft import ifft       # Performs much better than numpy's fftpack
except ImportError:                                    # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import ifft

SOUNDSPEED = 343.2
"""
Speed of sound.
"""

def soundspeed(ref_temp, temp):
    """
    Speed of sound :math:`c`.
    
    :param ref_temp: Reference temperature :math:`T`
    :param temp: Ambient temperature :math:`T_0`
    
    According to ISO9613-1:1993.
    
    ..  math:: c = 343.2 \\left( \\frac{T}{T_0} \\right)
    
    """
    return 343.2 * np.sqrt(temp / ref_temp)
    
def saturation_pressure(ref_pressure, triple_temp, temp):
    """
    Saturation vapour pressure :math:`p_{sat}`.
    
    :param ref_pressure: Reference pressure :math:`p_r`
    :param triple_temp: Triple point temperature water :math:`T_{01}`
    :param temp: Ambient temperature :math:`T`
    
    According to ISO9613-1:1993.
    
    .. math:: p_{sat} = 10^C \cdot p_r
    
    with exponent :math:`C` given by
    
    .. math:: C = -6.8346 \cdot \\left( \\frac{T_{01}}{T} \\right)^{1.261}  + 4.6151
    
    """
    return ref_pressure * 10.0** (-6.8346 *(triple_temp/temp)**(1.261) + 4.6151)

def molar_concentration_water_vapour(relative_humidity, saturation_pressure, pressure):
    """
    Molar concentration of water vapour :math:`h`.
    
    :param relative_humidity: Relative humidity :math:`h_r`
    :param saturation_pressure: Saturation pressure :math:`p_{sat}`
    :param pressure: Ambient pressure :math:`p`
    
    According to ISO9613-1:1993.
    
    .. math:: h = h_r  \\frac{p_{sat}}{p_a}
    
    """
    return relative_humidity * saturation_pressure / pressure

def relaxation_frequency_oxygen(pressure, ref_pressure, h):
    """
    Relaxation frequency of oxygen :math:`f_{r,O}`.
    
    :param pressure: Ambient pressure :math:`p_a`
    :param ref_pressure: Reference pressure :math:`p_r`
    :param h: Molar concentration of water vapour :math:`h`
    
    According to ISO9613-1:1993.
    
    .. math:: f_{r,O} = \\frac{p_a}{p_r} \\left( 24 + 4.04 \cdot 10^4 h \\frac{0.02 + h}{0.391 + h}  \\right)
    
    """
    return pressure / ref_pressure * ( 24.0 + 4.04 * 10.0**4.0 * h * (0.02 + h) / (0.391 + h) )

def relaxation_frequency_nitrogen(pressure, ref_pressure, temperature, ref_temperature, h):        
    """
    Relaxation frequency of nitrogen :math:`f_{r,N}`.
    
    :param pressure: Ambient pressure :math:`p_a`
    :param ref_pressure: Reference pressure :math:`p_{ref}`
    :param temperature: Ambient temperature :math:`T`
    :param ref_temperature: Reference temperature :math:`T_{ref}`
    :param h: Molar concentration of water vapour :math:`h`
    
    According to ISO9613-1:1993.
    
    .. math:: f_{r,N} = \\frac{p_a}{p_r} \\left( \\frac{T}{T_0} \\right)^{-1/2} \cdot \\left( 9 + 280 h \exp{\\left\{ -4.170 \\left[ \\left(\\frac{T}{T_0} \\right)^{-1/3} -1 \\right] \\right\} } \\right)
    
    """
    return pressure / ref_pressure * (temperature/ref_temperature)**(-0.5) * (9.0 + 280.0 * h * np.exp(-4.170 * ((temperature/ref_temperature)**(-1.0/3.0) - 1.0 ) ) )

def attenuation_coefficient(pressure, reference_pressure, temperature, reference_temperature, relaxation_frequency_nitrogen, relaxation_frequency_oxygen, frequency):
    """
    Attenuation coefficient :math:`\\alpha` describing atmospheric absorption in dB/m for the specified ``frequency``.
    
    :param temperature: Ambient temperature :math:`T`
    :param pressure: Ambient pressure :math:`T`
    
    :param frequency: Frequencies to calculate :math:`\\alpha` for.
    
    According to ISO9613-1:1993.
    """
    return 8.686 * frequency**2.0 * ( ( 1.84 * 10.0**(-11.0) * (reference_pressure/pressure) * (temperature/reference_temperature)**(0.5)) + (temperature/reference_temperature)**(-2.5) * ( 0.01275 * np.exp(-2239.1 / temperature) * (relaxation_frequency_oxygen + (frequency**2.0/relaxation_frequency_oxygen))**(-1.0) + 0.1068 * np.exp(-3352.0/temperature) * (relaxation_frequency_nitrogen + (frequency**2.0/relaxation_frequency_nitrogen))**(-1.0) ) )


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

    def __init__(self, temperature=293.15, pressure=101.325, relative_humidity=0.0):
        """
        Constructor
        
        :param temperature: Temperature
        :param pressure: Pressure
        :param relative_humidity: Relative humidity
        """
        
        ###self.__class__.temperature.add_callback(self, self._update)
        ###self.__class__.pressure.add_callback(self, self._update)
        ###self.__class__.relative_humidity.add_callback(self, self._update)
        
        self.temperature = temperature
        """Ambient temperature :math:`T`."""
        
        self.pressure = pressure
        """Ambient pressure :math:`p_a`."""
        
        self.relative_humidity = relative_humidity
        """Relative humidity"""
        
    @property
    def soundspeed(self):
        """
        Speed of sound :math:`c` calculated using :func:`soundspeed`.
        """
        return soundspeed(self.temperature, self.REF_TEMP)
        
    @property
    def saturation_pressure(self):
        """
        Saturation pressure :math:`p_{sat}` calculated using :func:`saturation_pressure`.
        """
        return saturation_pressure(self.REF_PRESSURE, self.TRIPLE_TEMP, self.temperature)
    
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
        return relaxation_frequency_nitrogen(self.pressure, self.REF_PRESSURE, self.temperature, self.REF_TEMP, self.molar_concentration_water_vapour)
    
    @property
    def relaxation_frequency_oxygen(self):
        """
        Resonance frequency of oxygen :math:`f_{r,O}` calculated using :func:`relaxation_frequency_oxygen`.
        """
        return relaxation_frequency_oxygen(self.pressure, self.REF_PRESSURE, self.molar_concentration_water_vapour)
    
    
    def attenuation_coefficient(self, frequency):
        """
        Attenuation coefficient :math:`\\alpha` describing atmospheric absorption in dB/m as function of ``frequency``.
        
        :param frequency: Frequencies to be considered.
        """
        return attenuation_coefficient(self.pressure, self.REF_PRESSURE, self.temperature, self.REF_TEMP, self.relaxation_frequency_nitrogen, self.relaxation_frequency_oxygen, frequency)
        
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
    
    
    
    def plot_attenuation_coefficient(self, frequency, filename=None):
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
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()
            
