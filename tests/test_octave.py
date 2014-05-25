"""
Tests for :class:`Acoustics.Octave.Octave`
"""
import unittest

from acoustics.octave import Octave
import numpy as np

class OctaveCase(unittest.TestCase):
    
    
    def setUp(self):
        """
        Code run before each unit test.
        """
    
    def tearDown(self):
        """
        Code run after each test.
        """
        
    def test_interval(self):
        emin = 1.0
        emax = 4.0
        f = np.logspace(emin, emax, 50)
        
        o = Octave(interval=f)
        
        self.assertEqual(o.fmin, 10.0**emin)
        self.assertEqual(o.fmax, 10.0**emax)
        self.assertEqual(len(o.n), len(o.center))
        
        o.unique = True
        self.assertEqual(len(o.n), len(f))
    
    def test_minmax(self):
        fmin = 250.0
        fmax = 1000.0
        
        o = Octave(fmin=fmin, fmax=fmax)
        
        self.assertEqual(len(o.center), 3) # 250, 500, and 1000 Hz
        self.assertEqual(len(o.n), 3)
        
if __name__ == '__main__':
    unittest.main()
