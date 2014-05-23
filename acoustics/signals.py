"""
Signals
=======

The signals module provides classes to work with signals in time and frequency domain.

.. inheritance-diagram:: acoustics.signals
"""

import numpy as np
import six
import abc
import matplotlib.pyplot as plt

from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

from quantity import Quantity, Unit, get_quantity


integration_times = {
    'fast'      : 0.125,
    'slow'      : 1.000,
    'impulse'   : 0.035, 
    'F'         : 0.125,
    'S'         : 1.000,
    'I'         : 0.035,
    }
"""
Standard integration times.
"""

def get_integration_time(x):
    """
    Get integration time.
    
    :param x: String or float.
    
    """
    try: 
        return integration_times[x]
    except KeyError:
        try:
            return integration_times[x.lower()]
        except AttributeError:
            try:
                return float(x)
            except TypeError:
                raise TypeError("Invalid integration time.")

            
@six.add_metaclass(abc.ABCMeta)
class Signal(object):
    """
    Signal.
    """
        
    def __init__(self, data, fs, quantity):
        """
        
        :param data: Data.
        :param fs: Sample frequency.
        
        """
        
        self._data = data
        """
        Data.
        """
        self.sample_frequency = fs
        """
        Sample frequency.
        """
        
        self.quantity = quantity
        """
        Quantity. Instance of :class:`Quantity`.
        """
    
    @property
    def quantity(self):
        """
        Quantity.
        """
        return self._quantity
    
    @quantity.setter
    def quantity(self, x):
        if isinstance(x, Quantity):
            self._quantity = x
        else:
            self._quantity = get_quantity(x)
        
    @abc.abstractproperty
    def samples(self):
        """
        Amount of samples.
        """
        pass
    
    @property
    def fs(self):
        """
        Samples frequency. Alias for :meth:`sample_frequency`.
        """
        return self.sample_frequency
    
    @property
    def duration(self):
        """
        Duration of signal in seconds.
        """
        return self.samples * self.sample_frequency
    
    
    @property
    def times(self):
        """
        Timestamps.
        """
        return np.arange(0, len(self)) * 1.0/self.sample_frequency
    
    @abc.abstractmethod
    def calibrate(self, value, level=True):
        """
        Calibrate with value.
        """
        pass


    @abc.abstractmethod
    def plot(self):
        """
        Plot response.
        """
        pass

@six.add_metaclass(abc.ABCMeta)
class Scalar(Signal):
    """
    Scalar quantity signal.
    """
    
    @property
    def samples(self):
        return self._data.shape[0]


@six.add_metaclass(abc.ABCMeta)
class Vector(Signal):
    """
    Vector quantity signal.
    """

    @property 
    def samples(self):
        return self._data.shape[1]
     
    @property
    def x(self):
        return self._data[0]
    
    @property
    def y(self):
        return self._data[1]
    
    @property
    def z(self):
        return self._data[2]

@six.add_metaclass(abc.ABCMeta)
class Linear(Signal):
    """
    Data is linearly scaled.
    """
    
    linear = True
    level = False



    def level(self):
        if self.dynamic:
            return 20.0 * np.log10(self._data / self.quantity.reference)
        else:
            return 10.0 * np.log10(self._data / self.quantity.reference)
    
    def peak(self):
        """
        Peak value.
        
        See :func:`acoustics.standards.iso_tr_25417_2007`.
        """
        return acoustics.standards.iso_tr_25417_2007(self._data)
    
    

@six.add_metaclass(abc.ABCMeta)
class Level(Signal):
    """
    Data is logarithmically scaled, i.e. in decibel.
    """
    
    linear = False
    level = True
    
    def linear(self):
        if self.dynamic:
            return 10.0**(self._data/20.0)
        else:
            return 10.0**(self._data/10.0)
     

    def peak(self):
        """
        Peak sound pressure level.
        """
        raise NotImplementedError

@six.add_metaclass(abc.ABCMeta)
class Instantaneous(Signal):
    """
    Instantaneous values.
    """
    
    def __init__(self, data, fs, quantity='pressure'):
        
        super(self, Instantaneous).__init__(self, data, fs, quantity)

    
    def spectral(self):
        """
        Convert to spectral values.
        """

    def integrate(self, integration_time='fast'):
        """
        Integrate over time.
        """
        raise NotImplementedError
        return Integrated(integrated_data, 1.0/integration_time, integration_time='fast')
   
    @classmethod
    def from_wav(cls, filename, quantity='pressure'):
        """
        Load signal from WAV file.
        """
        fs, data = wavread(filename)  
        data = data.astype(np.float32, copy=False)
        data /= np.max(np.abs(data))
        return cls(data, fs=fs, quantity=quantity)
    
    def to_wav(self, filename):
        """
        Write signal to WAV file.
        
        :param filename: Filename
        
        .. warning:: The WAV file will have 16-bit depth!
        
        """
        wavfile.write(filename, self.sample_frequency, self.data/np.abs(self.data).max())
        
        
    def calibrate(self, value, level=True):
        
        if level:
            raise NotImplementedError
        else:
            raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class Integrated(Signal):
    """
    Integrated values.
    """
    
    def __init__(self, data, fs=None, integration_time='fast', weighting=None):
        
        if fs is None:
            try:
                fs = 1.0 / integration_time
            except TypeError:
                raise ValueError("Need at least either fs or integration time.")
            
        super(self, Integrated).__init__(data, fs)
        
        self.integration_time = integration_time
        
        self.weighting = weighting
        """
        Frequency weighting.
        """
    
    @property
    def time_averaged(self):
        """
        Time-averaged signal or not.
        """
        return 1.0/self.sample_frequency > self.integration_time
    
    @property
    def integration_time(self):
        """
        Integration time.
        """
        return self._integration_time
    
    @integration_time.setter
    def integration_time(self, x):
        self._integration_time = get_integration_time(x)

    def apply_weighting(self, weighting=None):
        """
        Apply frequency weighting.
        """
        raise NotImplementedError
        return self.__class__()
    
 
class InstantaneousScalar(Instantaneous, Scalar):
    """
    Instantaneous values of a scalar quantity.
    """
    pass

class InstantaneousVector(Instantaneous, Vector):
    """
    Instantaneous values of a vector quantity.
    """
    pass

class LinearScalar(Integrated, Linear, Scalar):
    """
    Integrated values of a scalar quantity. Expressed on a linear scale.
    """
    
    
    def plot(self):
        """
        Plot response.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot(self.times, self._data)
        ax.set_xlabel('$t$ in s')
        ax.set_ylabel('{} in {}'.format(self.quantity.symbol_latex, 
                                         self.quantity.unit.symbol_latex))
        return fig
    

class LinearVector(Integrated, Linear, Vector):
    """
    Integrated values of a vector quantity. Expressed on a linear scale.
    """
    pass

class LevelScalar(Integrated, Level, Scalar):
    """
    Integrated values of a scalar quantity. Expressed on a logarithmic scale.
    """
    
    
    def plot(self):
        """
        Plot response.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot(self.times, self._data)
        ax.set_xlabel('$t$ in s')
        ax.set_ylabel('{} in dB re. {} {}'.format(self.quantity.symbol_latex, 
                                                   self.quantity.reference, 
                                                   self.quantity.unit.symbol_latex))
        return fig
    

class LevelVector(Integrated, Level, Vector):
    """
    Integrated values of a scalar quantity. Expressed on a logarithmic scale.
    """
    pass

    
class TimeSpectrum(object):
    """
    Time series of spectral values.
    """
    
    def __init__(self, data, fs, quantity, frequencies):
        raise NotImplementedError

    
    
class Spectrum(object):
    """
    Spectrum.
    """
    



