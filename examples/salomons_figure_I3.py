"""
This script reproduces figure I.3 from Salomons.
"""

from acoustics.turbulence import Gaussian2DTempWind, VonKarman2DTempWind


def main():
    
    
    a = 1.0         # Correlation length
    
    
    sigma_T = np.sqrt(1e-5) / 2.0
    T_0 = 1.0
    mu_0 = np.sqrt(1e-5)
    sigma_nu = 0.0
    K_0 = 10.0
    L = 2.0 * np.pi / K_0 
    C_T = np.sqrt(6e-7)
    c_0 = 1.0
    C_v = np.sqrt(2e-6)
    l_0 = 0.001
    k_max = 5.48 / l_0
    k_x = 0.0
    
    g = Gaussian3DTempWind()
    
    
    




if __name__ == '__main__':
    main()