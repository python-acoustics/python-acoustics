"""
Generator
=========

The generator module provides signal generators.

The following functions calculate ``N`` samples and return an array containing the samples.

For indefinitely long iteration over the samples, consider using the output of these functions in :func:`itertools.cycle`.

Noise
*****

Different types of noise are available. The following table lists the color 
of noise and how the power and power density change per octave.

====== ===== =============
Color  Power Power density
====== ===== =============
White  +3 dB  0 dB
Pink    0 dB -3 dB
Blue   +6 dB +3 dB
Brown  -3 dB -6 dB
Violet +9 dB +6 dB
====== ===== =============

.. autofunction:: white
.. autofunction:: pink
.. autofunction:: blue
.. autofunction:: brown
.. autofunction:: violet

Waveforms
*********

For related functions, check :mod:`scipy.signal`.


"""

import numpy as np
import random
import itertools
#import scipy.signal.sawtooth

try:
    from pyfftw.interfaces.numpy_fft import rfft, irfft       # Performs much better than numpy's fftpack
except ImportError:                                    # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import rfft, irfft

from .signal import normalise


def noise(N, color='white'):
    """Noise generator.
    
    :param N: Amount of samples.
    :param color: Color of noise.
    
    """
    try:
        return noise_generators[color](N)
    except KeyError:
        raise ValueError("Incorrect color.")



def white(N):
    """
    White noise.
    
    :param N: Amount of samples.
    
    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band, 
    and therefore increases with 3 dB per octave.
    """
    return np.random.randn(N)



def pink(N):
    """
    Pink noise. 
    
    :param N: Amount of samples.
    
    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    
    """
    # This method uses the filter with the following coefficients.
    #b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    #a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    #return lfilter(B, A, np.random.randn(N))
    
    # Another way would be using the FFT
    x = np.random.randn(N)
    X = rfft(x) / N
    S = np.sqrt(np.arange(len(X))+1.) # +1 to avoid divide by zero
    y = (irfft(X/S)).real[0:N]
    return normalise(y)


def blue(N):
    """
    Blue noise. 
    
    :param N: Amount of samples.
    
    Power increases with 6 dB per octave.
    Power density increases with 3 dB per octave. 
    
    """
    x = np.random.randn(N)
    X = rfft(x) / N
    S = np.sqrt(np.arange(len(X)))# Filter
    y = (irfft(X*S)).real[0:N]
    return normalise(y)


def brown(N):
    """
    Violet noise.
    
    :param N: Amount of samples.
    
    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave. 

    """
    x = np.random.randn(N)
    X = rfft(x) / N
    S = (np.arange(len(X))+1)# Filter
    y = (irfft(X/S)).real[0:N]
    return normalise(y)


def violet(N):
    """
    Violet noise. Power increases with 6 dB per octave. 
    
    :param N: Amount of samples.
    
    Power increases with +9 dB per octave.
    Power density increases with +6 dB per octave. 
    
    """
    x = np.random.randn(N)
    X = rfft(x) / N
    S = (np.arange(len(X)))# Filter
    y = (irfft(X*S)).real[0:N]
    return normalise(y)


noise_generators = {
    'white'  : white,
    'pink'   : pink,
    'blue'   : blue,
    'brown'  : brown,
    'violet' : violet,
    }


def noise_frequency_dependent(N, frequencies, gains):
    """Frequency-dependent noise signal.
    
    :param N: Amount of samples.
    :param frequencies: Normalized frequencies.
    :param gains: Gains.
    :param gains: Gain for given frequency.
    :type gains: List of tuples or 2xN array. The first item in the 



#def sawtooth(f, fs, width=1):
    #"""
    #Sawtooth. Return f sawteeth per second.
    
    
    #"""
    #t = np.arange()

#def white_generator(N=44100):
    #"""
    #White noise generator. Repeat every ``N`` samples.
    #"""
    #return itertools.cycle(white(N))

#def pink_generator(N=44100):
    #"""
    #Pink noise generator. Repeat every ``N`` samples.
    #"""
    #return itertools.cycle(white(N))

#def brown_generator(N=44100):
    #"""
    #Brown noise generator. Repeat every ``N`` samples.
    #"""
    #return itertools.cycle(brown(N))

#def generate(iterable):
    #"""
    #Return indefinetely long generator of iterable.
    
    #.. note:: :func:`itertools.cycle`
    #"""
    #return itertools.cycle(iterable)


#def sawtooth_generator(f, fs=44100):
    #"""
    #Sawtooth generator. Repeat....
    #"""
    #raise NotImplementedError
    

def heaviside(N):
    """Heaviside.
    
    Returns the value 0 for `x < 0`, 1 for `x > 0`, and 1/2 for `x = 0`.
    """
    return 0.5 * (np.sign(N) + 1)
    
