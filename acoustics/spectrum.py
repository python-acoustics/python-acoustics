import six
import numpy as np
import abc
import matplotlib.pyplot as plt
import acoustics.octave
import operator
from copy import deepcopy as copy

REFERENCE = 1000


def _apply_operator(a, b, op):
    
    try:
        same = np.all(a.center == b.center)
    except AttributeError: # does not have center attribute
        out = copy(a)
        out._data = op(a._data, b)
        return out
        #return type(a)(op(a._data, b), fstart=a.fstart, fstop=a.fstop, scale=a.scale)
    else:
        if a.scale != b.scale:
            return TypeError("Scales do not match.")
        elif not same:
            return TypeError("Frequency bins do not match.")
        else:
        #if same and a.scale==b.scale: # same center frequencies
            out = copy(a)
            out._data = op(a._data, b._data)
            return out
            #return type(a)(op(a._data, b._data), fstart=a.fstart, fstop=a.fstop, scale=a.scale)
        #else:
            #return TypeError("Spectra do not match.")


    
@six.add_metaclass(abc.ABCMeta)
class Spectrum(object):
    """
    Abstract spectrum class.
    """
    
    
    def __init__(self, data, fstart, fstop, scale='lin'):
        
        """
                
        :param data: Spectral values.
        :param fstart: Lowest center frequency.
        :param fstop: Highest center frequency.
        :param scale: Linear or logarithmic. Options are 'lin' and 'log'.
        """
        
        if isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError("Wrong dimensions.")
        
        self._data = data
        """
        Actual data.
        """
        
        self.fstart = fstart
        """
        Lowest center frequency.
        """
        
        self.fstop = fstop
        """
        
        Highest center frequency.
        """
        
        self.scale = scale
        """
        Linear or logarithmic amplitudes.
        """
    
    def __add__(self, other):
        return _apply_operator(self, other, operator.add)
    
    def __sub__(self, other):
        return _apply_operator(self, other, operator.sub)
    
    def __mul__(self, other):
        return _apply_operator(self, other, operator.mul)
    
    def __div__(self, other):
        return _apply_operator(self, other, operator.div)
    
    def __truediv__(self, other):
        return _apply_operator(self, other, operator.truediv)
    
    def __floordiv__(self, other):
        return _apply_operator(self, other, operator.floordiv)
    
    def __mod__(self, other):
        return _apply_operator(self, other, operator.mod)
    
    def __pow__(self, other):
        return _apply_operator(self, other, operator.pow)
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        for i in self._data:
            yield i
    
    def __repr__(self):
        return "spectrum({})".format(self._data.__str__())
    
    def __str__(self):
        return self._data.__str__()
    
    @abc.abstractproperty
    def center(self):
        pass
    
    @abc.abstractproperty
    def upper(self):
        pass
    
    @abc.abstractproperty
    def lower(self):
        pass

    @abc.abstractproperty
    def bandwidth(self):
        pass
    
    def max(self):
        """
        Maximum value in spectrum.
        """
        return self._data.max()
    
    def min(self):
        """
        Minimum value in spectrum.
        """
        return self._data.min()
    
    def argmax(self):
        """
        Index of maximum value in spectrum.
        """
        return self._data.argmax()
    
    def argmin(self):
        """
        Index of minimum value in spectrum.
        """
        return self._data.argmin()
    
    def dynamic_range(self):
        """
        Dynamic range.
        """
        if self.scale=='log':
            return self.max() - self.min()
        elif self.scale=='lin':
            return self.max()/self.min()

    def asarray(self):
        """
        Spectrum as `np.ndarray`. 
        
        .. warning:: Creates a view, not a copy!
        
        """
        return self._data

    #@classmethod
    #def empty(cls, 


class Equalband(Spectrum):
    """
    Equal bandwidth spectrum. Generally used for narrowband data.
    """
        
    def __init__(self, data, center=None, fstart=None, fstop=None, bandwidth=None):
        """
        :param data: Spectral values.
        :param center: Center frequencies.
        :param fstart: Lowest center frequency.
        :param fstop: Highest center frequency.
        :param reference: Reference center frequency.
        """
        
        if center is not None:
            if len(center) != len(data):
                raise ValueError("Amount of center frequencies does not match with data.")
            fstart = center[0]
            fstop = center[-1]
            u = np.unique(np.gradient(center))
            if len(u)==1:
                bandwidth = u
            else:
                raise ValueError("Given center frequencies are not equally spaced.")
        elif fstart and fstop:
            bandwidth = (fstop - fstart) / len(data)
        elif fstart and bandwidth:
            fstop = fstart + len(data) * bandwidth
        elif fstop and bandwidth:
            fstart = fstop - len(data) * bandwidth
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart, fstop, bandwidth.")
        
        super(Equalband, self).__init__(data, fstart, fstop)
        
        self._bandwidth = bandwidth

    @property
    def center(self):
        return self.fstart + np.arange(len(self)) * self.bandwidth + self.bandwidth/2.0
        
    @property
    def upper(self):
        return self.fstart + np.arange(len(self)) * self.bandwidth + self.bandwidth
    
    @property
    def lower(self):
        return self.fstart + np.arange(len(self)) * self.bandwidth

    @property
    def bandwidth(self):
        return self._bandwidth
    
    def real(self):
        """
        Real part of the spectrum.
        """
        return self._data.real

    def imag(self):
        """
        Imaginary part of the spectrum.
        """
        return self._data.imag

    def conjugate(self):
        """
        Complex conjugate spectrum.
        """
        return self.__class__(self._data.conjugate(), fstart=self.fstart, fstop=self.stop)

    def plot(self, filename=None):
        """
        Plot spectrum.
        """
        fig = plt.figure()
        ax0 = fig.add_subplot(211)
        a = ax0.plot(self.center, np.abs(self._data))
        ax0.set_xlabel('$f$ in Hz')
        ax0.set_ylabel('$|x|$')
        ax0.grid()
        
        ax1 = fig.add_subplot(212)
        p = ax1.plot(self.center, np.angle(self._data))
        ax1.set_xlabel('$f$ in Hz')
        ax1.set_ylabel('$\angle$ in rad') 
        ax1.grid()
        
        if filename:
            fig.savefig(filename)
        else:
            return fig

class Octave(Spectrum):
    """
    Fractional-octave band spectrum.
    """
    
    def __init__(self, data, center=None, fstart=None, fstop=None, fraction=1, reference=REFERENCE):
        """
        :param data: Spectral values.
        :param center: Center frequencies.
        :param fstart: Lowest center frequency.
        :param fstop: Highest center frequency.
        :param fraction: Fraction of octave.
        :param reference: Reference center frequency.
        """
        
        if center is not None:
            if len(center) != len(data):
                raise ValueError("Amount of center frequencies does not match with data.")
            fstart = center[0]
            fstop = center[-1]
        
        elif fstart and fstop:
            pass
        elif fstart and not fstop:
            nstart = acoustics.octave.band_of_frequency(fstart, order=fraction, ref=reference)
            nstop = nstart + len(data)-1
            fstop = acoustics.octave.frequency_of_band(nstop, order=fraction, ref=reference)
        elif not fstart and fstop:
            nstop = acoustics.octave.band_of_frequency(fstop, order=fraction, ref=reference)
            nstart = nstop - len(data)+1
            fstart = acoustics.octave.band_of_frequency(nstart, order=fraction, ref=reference)
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart and/or fstop.")    
        
        super(Octave, self).__init__(data, fstart, fstop)
        
        self.fraction = fraction
        self.reference = reference

    @property
    def center(self):
        return acoustics.octave.Octave(order=self.fraction, 
                                       fmin=self.fstart, 
                                       fmax=self.fstop, 
                                       reference=self.reference).center()
    
    @property
    def upper(self):
        return acoustics.octave.upper_frequency(self.center, self.fraction)
        #return self.center * 2.0**(+1.0/(2.0*self.fraction))
    
    @property
    def lower(self):
        return acoustics.octave.lower_frequency(self.center, self.fraction)
        #return self.center * 2.0**(-1.0/(2.0*self.fraction))

    @property
    def bandwidth(self):
        return self.upper - self.lower

    def as_octaves(self, fraction=1):
        """
        Convert to fractional-octaves.
        
        :param fraction: Fraction.
        
        """
        
        if fraction <= self.fraction:
            raise NotImplementedError
        else:
            raise ValueError("Incorrect fraction. Cannot increase resolution.")
    
    
    def plot(self, filename=None):
        """
        Plot spectrum.
        """
        center = self.center
        
        width=1.0
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = ax.bar(range(len(center)), self._data, width=width)
        ax.set_xlabel('$f$ in Hz')
        ax.set_xticks(np.arange(len(center)) + width/2)
        ax.set_xticklabels( center.astype('int') )
        ax.grid()
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
        
