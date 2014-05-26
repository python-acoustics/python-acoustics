import numpy as np
from acoustics.generator import white, pink, blue, brown, violet

from acoustics.signal import octaves

class Test_white:
    """
    Test :func:`acoustics.generator.white`.
    """
    
    def test_length(self):
        N = 1000
        assert(len(white(N))==N)
        
        N = 1001
        assert(len(white(N))==N)

    def test_power(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(white(samples), fs); 
        change = np.diff(L).mean().round(0)
        assert(change==+3.)
    
    def test_power_density(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(white(samples), fs, density=True); 
        change = np.diff(L).mean().round(0)
        assert(change==0.)
    
    
    
class Test_pink:
    """
    Test :func:`acoustics.generator.pink`.
    """
    
    def test_length(self):
        N = 1000
        assert(len(pink(N))==N)
        
        # Need to fix length!!
        #N = 1001
        #assert(len(pink(N))==N)
    
    def test_power(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(pink(samples), fs); 
        change = np.diff(L).mean().round(0)
        assert(change==0.)
   
    def test_power_density(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(pink(samples), fs, density=True); 
        change = np.diff(L).mean().round(0)
        assert(change==-3.)
        
        
class Test_blue:
    """
    Test :func:`acoustics.generator.blue`.
    """
    
    def test_length(self):
        N = 1000
        assert(len(blue(N))==N)
        
        # Need to fix length!!
        #N = 1001
        #assert(len(blue(N))==N)
    
    def test_power(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(blue(samples), fs); 
        change = np.diff(L).mean().round(0)
        assert(change==+6.)

    def test_power_density(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(blue(samples), fs, density=True); 
        change = np.diff(L).mean().round(0)
        assert(change==+3.)
        
        
class Test_brown:
    """
    Test :func:`acoustics.generator.brown`.
    """
    
    def test_length(self):
        N = 1000
        assert(len(brown(N))==N)
        
        # Need to fix length!!
        #N = 1001
        #assert(len(brown(N))==N)
    
    def test_power(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(brown(samples), fs); 
        change = np.diff(L).mean().round(0)
        assert(change==-3.)
    
    def test_power_density(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(brown(samples), fs, density=True); 
        change = np.diff(L).mean().round(0)
        assert(change==-6.)


class Test_violet:
    """
    Test :func:`acoustics.generator.violet`.
    """
    
    def test_length(self):
        N = 1000
        assert(len(violet(N))==N)
        
        # Need to fix length!!
        #N = 1001
        #assert(len(violet(N))==N)
    
    def test_power(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(violet(samples), fs); 
        change = np.diff(L).mean().round(0)
        assert(change==+9.)
    
    def test_power_density(self):
        fs = 44100
        samples = 44100 * 10
        _, L = octaves(violet(samples), fs, density=True); 
        change = np.diff(L).mean().round(0)
        assert(change==+6.)



