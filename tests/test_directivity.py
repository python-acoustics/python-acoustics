
import unittest

from acoustics.directivity import *

import numpy as np

class PatternsTest(unittest.TestCase):
    
    
    def test_cardioid(self):
        pass
        
    
    def test_figure_eight(self):
        
        self.assertEqual(figure_eight(0.0), +1.0)
        self.assertAlmostEqual(figure_eight(1./2.*np.pi), 0.0)
        self.assertEqual(figure_eight(np.pi), +1.0)
        self.assertAlmostEqual(figure_eight(3./2.*np.pi), 0.0)
        self.assertEqual(figure_eight(2.*np.pi), +1.0)
        
        


class OmniTest(unittest.TestCase):
    pass

class CardioidTest(unittest.TestCase):
    pass

class FigureEight(unittest.TestCase):
    pass

class CustomTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
    
    
    
    