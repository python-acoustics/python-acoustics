"""
Tests for :func:`Acoustics.LTV.convolve`
"""
import unittest

from acoustics.signal import convolve as convolveLTV
from scipy.signal import convolve as convolveLTI
import numpy as np
import itertools

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
        
    
if __name__ == '__main__':
    unittest.main()