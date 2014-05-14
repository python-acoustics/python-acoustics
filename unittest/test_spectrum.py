import unittest
import numpy as np

from acoustics.spectrum import Equalband, Octave


class EqualbandCase(unittest.TestCase):
    
    def test_construction(self):
        
        data = np.ones(10)
        
        s = Equalband(data, fstart=10.0, fstop=110.0)
        
        


class OctaveCase(unittest.TestCase):
    
    pass



if __name__ == '__main__':
    unittest.main()
