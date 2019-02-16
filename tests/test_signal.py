"""
Tests for :func:`acoustics.signal`
"""
from acoustics.signal import convolve as convolveLTV
from scipy.signal import convolve as convolveLTI
import numpy as np
import itertools

from acoustics.signal import *  #decibel_to_neper, neper_to_decibel, ir2fr, zero_crossings
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_approx_equal

import pytest


@pytest.mark.parametrize("u,h", [
    (np.array([1, 2, 3, 4, 3, 2, 1], dtype='float'), np.array([1, 2, 3, 4], dtype='float')),
    (np.array([1, 2, 3, 4, 3, 2, 1, 1], dtype='float'), np.array([1, 2, 3, 4, 5], dtype='float')),
])
def test_convolve_lti(u, h):
    """Test whether :func:`acoustics.signal.convolve` behaves properly when
    performing a convolution with a time-invariant system.
    """
    H = np.tile(h, (len(u), 1)).T

    np.testing.assert_array_almost_equal(convolveLTV(u, H), convolveLTI(u, h))
    np.testing.assert_array_almost_equal(convolveLTV(u, H, mode='full'), convolveLTI(u, h, mode='full'))
    np.testing.assert_array_almost_equal(convolveLTV(u, H, mode='valid'), convolveLTI(u, h, mode='valid'))
    # This function and test needs some investigation. Disabling for now.
    #np.testing.assert_array_almost_equal(convolveLTV(u,H,mode='same'), convolveLTI(u,h,mode='same'))


def test_convolve_ltv():
    """Test whether :func:`acoustics.signal.convolve` behaves properly when
    performing a convolution with a time-variant system.
    """
    """Input signal"""
    u = np.array([1, 1, 1])
    """Impulse responses where each column represents an impulse response."""
    C = np.array([[1, 0, 0], [2, 1, 1]])
    """The result calculated manually."""
    y_manual = np.array([1, 2, 1, 1])

    y_ltv = convolveLTV(u, C)
    np.testing.assert_array_equal(y_ltv, y_manual)


def test_decibel_to_neper():
    assert (decibel_to_neper(1.0) == 0.11512925464970229)


def test_neper_to_decibel():
    assert (neper_to_decibel(1.0) == 8.685889638065035)


def test_ir2fr():
    """
    Test whether the frequency vector is correct.
    """

    t = 1.0
    fs = 100.0
    f = 20.0
    ts = np.arange(0, t, 1. / fs)

    A = 5.0

    x = A * np.sin(2. * np.pi * f * ts)

    fv, fr = ir2fr(x, fs)

    assert_array_almost_equal(fv[np.abs(fr).argmax()], f)

    assert_array_almost_equal(np.abs(fr).max(), A)


class TestEqualBand:  #(unittest.TestCase):
    """
    Test :class:`acoustics.signal.EqualBand`.
    """

    def test_construction_1(self):
        """Using center."""
        x = np.arange(10.0, 20.0, 2.0)
        b = EqualBand(x)
        assert_array_equal(b.center, x)

    def test_construction_2(self):
        """Using fstart, fstop and fbands"""
        x = np.arange(10.0, 20.0, 2.0)
        fstart = x[0]
        fstop = x[-1]
        nbands = len(x)
        b = EqualBand(fstart=fstart, fstop=fstop, nbands=nbands)
        assert_array_equal(b.center, x)

    def test_construction_3(self):
        """Using fstart, fstop and bandwidth"""
        x = np.arange(10.0, 20.0, 2.0)
        fstart = x[0]
        fstop = x[-1]
        bandwidth = np.diff(x)[0]
        b = EqualBand(fstart=fstart, fstop=fstop, bandwidth=bandwidth)
        assert_array_equal(b.center, x)

    def test_construction_4(self):
        # Using fstart, bandwidth and bands
        x = np.arange(10.0, 20.0, 2.0)
        fstart = x[0]
        bandwidth = np.diff(x)[0]
        nbands = len(x)
        b = EqualBand(fstart=fstart, nbands=nbands, bandwidth=bandwidth)
        assert_array_equal(b.center, x)

    def test_construction_5(self):
        # Using fstop, bandwidth and bands
        x = np.arange(10.0, 20.0, 2.0)
        fstop = x[-1]
        bandwidth = np.diff(x)[0]
        nbands = len(x)
        b = EqualBand(fstop=fstop, nbands=nbands, bandwidth=bandwidth)
        assert_array_equal(b.center, x)

    def test_selection(self):

        eb = EqualBand(fstart=0.0, fstop=10.0, nbands=100)
        assert type(eb[3] == type(eb))
        assert type(eb[3:10] == type(eb))


class Test_integrate_bands():
    """
    Test :func:`acoustics.signal.test_integrate_bands`.
    """

    def test_narrowband_to_octave(self):

        nb = EqualBand(np.arange(100, 900, 200.))
        x = np.ones(len(nb))
        ob = OctaveBand(([125., 250, 500.]))
        y = integrate_bands(x, nb, ob)
        assert_array_equal(y, np.array([1, 1, 2]))


def test_zero_crossings():

    duration = 2.0
    fs = 44100.0
    f = 1000.0
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0 * np.pi * f * t)

    z = zero_crossings(x)

    # Amount of zero crossings.
    assert (len(z) == f * duration * 2)

    # Position of zero crossings.
    y = np.linspace(0, samples, len(z), endpoint=False).astype(int)
    assert ((np.abs(z - y) <= 1).all())


def test_ms():

    duration = 2.0
    fs = 8000.0
    f = 1000.0
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0 * np.pi * f * t)

    assert (np.abs(ms(x) - 0.5) < 1e-9)

    x *= 4.0

    assert (np.abs(ms(x) - 8.0) < 1e-9)


def test_rms():

    duration = 2.0
    fs = 8000.0
    f = 1000.0
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0 * np.pi * f * t)

    assert (np.abs(rms(x) - np.sqrt(0.5)) < 1e-9)

    x *= 4.0

    assert (np.abs(rms(x) - np.sqrt(8.0)) < 1e-9)


@pytest.fixture(params=[4000.0, 20000.0, 44100.0])
def fs(request):
    return request.param


@pytest.fixture(params=[1.0, 2.0, 3.0])
def amplitude(request):
    return request.param


@pytest.fixture(params=[100.0, 200.0, 300.0])
def frequency(request):
    return request.param


def test_amplitude_envelope(amplitude, frequency, fs):
    """Test amplitude envelope.
    """
    duration = 5.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    signal = amplitude * np.sin(2.0 * np.pi * frequency * t)

    out = amplitude_envelope(signal, fs)
    # Rounding is necessary. We take the first element because occasionally
    # there is also a zero.
    amplitude_determined = np.unique(np.round(out), 6)[0]

    assert (amplitude == amplitude_determined)


#def test_instantaneous_frequency(amplitude, frequency, fs):

#duration = 5.0
#samples = int(fs*duration)
#t = np.arange(samples) / fs

#signal = amplitude * np.sin(2.0*np.pi*frequency*t)

#out = instantaneous_frequency(signal, fs)
## Rounding is necessary. We take the first element because occasionally
## there is also a zero.
#frequency_determined = np.unique(np.round(out), 0)

#assert( frequency == frequency_determined )


@pytest.mark.parametrize("channels", [1, 2, 5])
def test_bandpass(channels):
    fs = 88200
    duration = 2
    samples = duration * fs

    signal = np.random.randn(channels, samples)

    # We check whether it computes, and whether channels is in right dimension
    result = bandpass_octaves(signal, fs, order=8, purge=False)
    assert result[1].shape[-2] == channels

    result = bandpass_third_octaves(signal, fs, order=8, purge=False)
    assert result[1].shape[-2] == channels

    # We need to define frequencies explicitly
    with pytest.raises(TypeError):
        bandpass_fractional_octaves(signal, fs)

    frequencies = OctaveBand(fstart=100.0, fstop=2000.0, fraction=12)
    result = bandpass_fractional_octaves(signal, fs, frequencies, order=8, purge=False)
    assert result[1].shape[-2] == channels

    frequencies = EqualBand(center=[100.0, 200.0, 300.0], bandwidth=20.0)
    result = bandpass_frequencies(signal, fs, frequencies)
    assert result[1].shape[-2] == channels


@pytest.fixture(params=[1, 3, 6, 12, 24])
def fraction(request):
    return request.param


@pytest.fixture
def ob(fraction):
    return OctaveBand(fstart=10.0, fstop=1000, fraction=fraction)


class TestOctaveBand:
    def test_unique(self, ob):
        """Test whether we don't have duplicate values."""
        assert len(ob.center) == len(np.unique(ob.center))
        assert len(ob.lower) == len(np.unique(ob.lower))
        assert len(ob.upper) == len(np.unique(ob.upper))
        assert len(ob.nominal) == len(np.unique(ob.nominal))
        assert len(ob.bandwidth) == len(np.unique(ob.bandwidth))
