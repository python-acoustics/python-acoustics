import numpy as np
from acoustics.turbulence import Gaussian2DTemp, VonKarman2DTemp, Comparison, Field2D

def main():
    
    mu_0 = np.sqrt(10.0**(-6))
    correlation_length = 1.0    # Typical correlation length for Gaussian spectrum.
    
    x = 20.0
    y = 0.0
    z = 40.0
    
    plane = (1,0,1)
    
    #f_resolution = wavenumber_resolution / (2.0*np.pi)
    
    spatial_resolution = 0.05
    
    
    N = 100
    
    min_wavenumber = 0.01
    max_wavenumber = 10.0
    
    
    wavenumber_resolution = (max_wavenumber - min_wavenumber) / N
    
    
    """Create an object to describe an Gaussian turbulence spectrum."""
    g = Gaussian2DTemp(plane=plane, a=correlation_length, mu_0=mu_0, wavenumber_resolution=wavenumber_resolution, max_mode_order=N)
    
    """Create an object to describe a VonKarman turbulence spectrum."""
    s = VonKarman2DTemp(plane=plane, a=correlation_length, mu_0=mu_0, wavenumber_resolution=wavenumber_resolution, max_mode_order=N)
    
    g.plot_mode_amplitudes('Gaussian2DTemp_mode_amplitudes.png')
    s.plot_mode_amplitudes('VonKarman2DTemp_mode_amplitudes.png')

    c = Comparison([g, s])
    
    c.plot_mode_amplitudes('Gaussian2DTemp_and_VonKarman2DTemp_mode_amplitudes.png')
    
    
    field_g = Field2D(x=x, y=y, z=z, spatial_resolution=spatial_resolution, spectrum=g)
    field_s = Field2D(x=x, y=y, z=z, spatial_resolution=spatial_resolution, spectrum=s)
    
    field_g.generate().plot('Gaussian2DTemp_field.png')
    field_s.generate().plot('VonKarman2DTemp_field.png')
    
    

    
    
    
    
    
    
    
if __name__ == '__main__':
    main()