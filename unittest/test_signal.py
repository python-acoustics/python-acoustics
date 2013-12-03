"""
Tests for :func:`Acoustics.LTV.convolve`
"""
import unittest

from acoustics.signal import convolve as convolveLTV
from scipy.signal import convolve as convolveLTI
import numpy as np

class ConvolveCase(unittest.TestCase):
    
    def test_LTI(self):
        """
        Test whether it gives correct results for the LTI case.
        """
        
        """Input signal"""
        u = np.array([1, 2, 3, 4, 3, 2, 1], dtype='float')
        
        """Impulse responses where each column represents an impulse response."""
        C = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4]
            ], dtype='float')

        """The array C represents here a linear time-invariant system."""
        y_ltv = convolveLTV(u, C)
        y_lti = convolveLTI(u, C[:,0])
        """Check whether the output is indeed the same."""
        np.testing.assert_array_equal(y_ltv, y_lti)
    
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