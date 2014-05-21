"""
Tests for :func:`Acoustics.LTV.convolve`
"""
import unittest

from acoustics.signal import convolve as convolveLTV
from scipy.signal import convolve as convolveLTI
import numpy as np
import itertools

from acoustics.signal import decibel_to_neper, neper_to_decibel, ir2fr
from numpy.testing import assert_almost_equal, assert_array_almost_equal


class ConvolveCase(unittest.TestCase):
    
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
    
if __name__ == '__main__':
    unittest.main()