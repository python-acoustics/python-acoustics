"""
Tests for :func:`acoustics.signal`
"""

from acoustics.signal import convolve as convolveLTV
from acoustics.signal import EqualBand, OctaveBand, integrate_bands
from scipy.signal import convolve as convolveLTI
import numpy as np
import itertools

from acoustics.signal import * #decibel_to_neper, neper_to_decibel, ir2fr, zero_crossings
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_approx_equal


class TestConvolve:
    
    def test_LTI(self):
        """
        Test whether it gives correct results for the LTI case.
        """

        """Input signals."""
        signals = [np.array([1,2,3,4,3,2,1], dtype='float'),
                   np.array([1,2,3,4,3,2,1,1], dtype='float'),
                   ]
        
        """Filters"""
        filters = [np.array([1,2,3,4], dtype='float'),
                   np.array([1,2,3,4,5], dtype='float'),
                   ]
        
        """Test for every combination of input signal and filter."""
        for u, h in itertools.product(signals, filters):

            H = np.tile(h, (len(u), 1)).T
        
            #"""The array C represents here a linear time-invariant system."""
            #y_ltv = convolveLTV(u, H)
            #y_lti = convolveLTI(u, h)
            #"""Check whether the output is indeed the same."""
            #np.testing.assert_array_equal(y_ltv, y_lti)
        
            np.testing.assert_array_almost_equal(convolveLTV(u,H), convolveLTI(u,h))
            np.testing.assert_array_almost_equal(convolveLTV(u,H,mode='full'), convolveLTI(u,h,mode='full'))
            np.testing.assert_array_almost_equal(convolveLTV(u,H,mode='valid'), convolveLTI(u,h,mode='valid'))
            np.testing.assert_array_almost_equal(convolveLTV(u,H,mode='same'), convolveLTI(u,h,mode='same'))

            
    def test_LTV(self):
        """
        Test whether it gives correct results for the LTV case.
        """
        
        """Input signal"""
        u = np.array([1, 1, 1])
        
        """Impulse responses where each column represents an impulse response."""
        C = np.array([
            [1, 0, 0],
            [2, 1, 1]
            ])
        
        """The result calculated manually."""
        y_manual = np.array([1, 2, 1, 1])
        
        y_ltv = convolveLTV(u, C)
        np.testing.assert_array_equal(y_ltv, y_manual)
        
        
def test_decibel_to_neper():
    assert( decibel_to_neper(1.0) == 0.11512925464970229)


def test_neper_to_decibel():
    assert( neper_to_decibel(1.0) == 8.685889638065035)
    
def test_ir2fr():
    """
    Test whether the frequency vector is correct.
    """
    
    t = 1.0
    fs = 100.0
    f = 20.0
    ts = np.arange(0, t, 1./fs)
    
    A = 5.0
    
    x = A * np.sin(2. * np.pi * f * ts)
    
    fv, fr = ir2fr(x, fs)
    
    assert_array_almost_equal(fv[np.abs(fr).argmax()], f)

    assert_array_almost_equal(np.abs(fr).max(), A)
    
    
class TestEqualBand:#(unittest.TestCase):
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
    


class Test_zero_crossings():
    
    def test_sine(self):
        
        duration = 2.0
        fs = 44100.0
        f = 1000.0
        samples = int(duration*fs)
        t = np.arange(samples) / fs
        x = np.sin(2.0*np.pi*f*t)
        
        z = zero_crossings(x)
        
        """Amount of zero crossings."""
        assert(len(z)==f*duration*2)
        

        y = np.arange(0, samples, round(fs/f/2), dtype='int64')
        print(y)    
        assert((np.abs(z-y) <= 1).all())
        
        
def test_ms():
    
    duration = 2.0
    fs = 8000.0
    f = 1000.0
    samples = int(duration*fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0*np.pi*f*t)
    
    assert(np.abs( ms(x) - 0.5) < 1e-9 ) 
    
    x *= 4.0
    
    assert(np.abs( ms(x) - 8.0) < 1e-9 )
        
        
def test_rms():
    
    duration = 2.0
    fs = 8000.0
    f = 1000.0
    samples = int(duration*fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0*np.pi*f*t)
    
    assert(np.abs( rms(x) - np.sqrt(0.5) ) < 1e-9 )
    
    x *= 4.0
    
    assert(np.abs( rms(x) - np.sqrt(8.0) ) < 1e-9 )
    
    
    
    
#if __name__ == '__main__':
    #unittest.main()
