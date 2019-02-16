"""
Generator
=========

The generator module provides signal generators.

The following functions calculate ``N`` samples and return an array containing the samples.

For indefinitely long iteration over the samples, consider using the output of these functions
in :func:`itertools.cycle`.

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

The colored noise is created by generating pseudo-random numbers using
:func:`np.random.randn` and then multiplying these with a curve tyical for the color.
Afterwards, an inverse DFT is performed using :func:`np.fft.irfft`.
Finally, the noise is normalized using :func:`acoustics.signal.normalize`.

All colors
----------

.. autofunction:: noise
.. autofunction:: noise_generator

Per color
---------

.. autofunction:: white
.. autofunction:: pink
.. autofunction:: blue
.. autofunction:: brown
.. autofunction:: violet


Other
*****

.. autofunction:: heaviside

For related functions, check :mod:`scipy.signal`.


"""
import itertools
import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import irfft  # Performs much better than numpy's fftpack
except ImportError:  # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import irfft  # pylint: disable=ungrouped-imports

from .signal import normalize


def noise(N, color='white', state=None):
    """Noise generator.

    :param N: Amount of samples.
    :param color: Color of noise.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    """
    try:
        return _noise_generators[color](N, state)
    except KeyError:
        raise ValueError("Incorrect color.")


def white(N, state=None):
    """
    White noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)


def pink(N, state=None):
    """
    Pink noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.

    """
    # This method uses the filter with the following coefficients.
    #b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    #a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    #return lfilter(B, A, np.random.randn(N))
    # Another way would be using the FFT
    #x = np.random.randn(N)
    #X = rfft(x) / N
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def blue(N, state=None):
    """
    Blue noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    Power increases with 6 dB per octave.
    Power density increases with 3 dB per octave.

    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def brown(N, state=None):
    """
    Violet noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave.

    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (np.arange(len(X)) + 1)  # Filter
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def violet(N, state=None):
    """
    Violet noise. Power increases with 6 dB per octave.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    Power increases with +9 dB per octave.
    Power density increases with +6 dB per octave.

    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


_noise_generators = {
    'white': white,
    'pink': pink,
    'blue': blue,
    'brown': brown,
    'violet': violet,
}


def noise_generator(N=44100, color='white', state=None):
    """Noise generator.

    :param N: Amount of unique samples to generate.
    :param color: Color of noise.

    Generate `N` amount of unique samples and cycle over these samples.

    """
    #yield from itertools.cycle(noise(N, color)) # Python 3.3
    for sample in itertools.cycle(noise(N, color, state)):
        yield sample


def heaviside(N):
    """Heaviside.

    Returns the value 0 for `x < 0`, 1 for `x > 0`, and 1/2 for `x = 0`.
    """
    return 0.5 * (np.sign(N) + 1)


__all__ = ['noise', 'white', 'pink', 'blue', 'brown', 'violet', 'noise_generator', 'heaviside']
