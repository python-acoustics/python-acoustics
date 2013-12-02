import unittest

from acoustics.atmosphere import Atmosphere

class AtmosphereCase(unittest.TestCase):
    
    
    def test_standard_atmosphere(self):
        
        a = Atmosphere()
        
        self.assertEqual(a.temperature, 293.15)
        self.assertEqual(a.pressure, 101.325)
        self.assertEqual(a.relative_humidity, 0.0)
        self.assertEqual(a.soundspeed, 343.2)
    
    
if __name__ == '__main__':
    unittest.main()
