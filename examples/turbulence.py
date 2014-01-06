import numpy as np
from acoustics.turbulence import Gaussian2DTemp, VonKarman2DTemp, Comparison

def main():
    
    mu_0 = np.sqrt(10.0**(-6))
    correlation_length = 1.0    # Typical correlation length for Gaussian spectrum.
    
    x = 20.0
    y = 0.0
    z = 40.0
    
    #f_resolution = wavenumber_resolution / (2.0*np.pi)
    
    spatial_resolution = 0.05
    
    
    N = 100
    
    min_wavenumber = 0.01
    max_wavenumber = 10.0
    
    
    wavenumber_resolution = (max_wavenumber - min_wavenumber) / N
    
    
    """Create an object to describe an Gaussian turbulence spectrum."""
    g = Gaussian2DTemp(x=x, y=y, z=z, spatial_resolution=spatial_resolution, a=correlation_length, mu_0=mu_0, wavenumber_resolution=wavenumber_resolution, max_mode_order=N)
    
    """Create an object to describe a VonKarman turbulence spectrum."""
    s = VonKarman2DTemp(x=x, y=y, z=z, spatial_resolution=spatial_resolution, a=correlation_length, mu_0=mu_0, wavenumber_resolution=wavenumber_resolution, max_mode_order=N)
    
    g.plot_field('Gaussian2DTemp_field.png')
    s.plot_field('VonKarman2DTemp_field.png')
    
    g.plot_mode_amplitudes('Gaussian2DTemp_mode_amplitudes.png')
    s.plot_mode_amplitudes('VonKarman2DTemp_mode_amplitudes.png')

    c = Comparison([g, s])
    
    c.plot_mode_amplitudes('Gaussian2DTemp_and_VonKarman2DTemp_mode_amplitudes.png')

    
    
    
    
    
    
    
if __name__ == '__main__':
    main()