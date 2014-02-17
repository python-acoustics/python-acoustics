"""
Turbulence in the atmosphere affects wave propagation.
This module contains implementations of models that can be used to create random turbulence fields.

References are made to the book 'Computational Atmospheric Acoustics' by 'Erik M. Salomons', published in 2001.
"""


import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import iv as bessel  # Modified Bessel function of the first kind.

from ._turbulence import *


class Gaussian1DTemp(GaussianTemp, Spectrum1D):
    """
    One-dimensional Gaussian turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod    
    def spectral_density_function(k, a, mu_0):
        """
        One-dimensional spectral density :math:`V(k)`.
        
        :param k: :math:`k`
        :param a: :math:`a`
        :param mu_0: :math:`\\mu_0`

        The spectral density function is given by
        
        .. math:: V(k) = \\mu_0^2 \\frac{a}{2\\sqrt{\\pi}} \\exp{(-k^2a^2/4)}
        
        See Salomons, equation I.30.
        """
        return mu_0**2.0 * a / (2.0 * np.sqrt(np.pi)) * np.exp(-k**2.0 * a**2.0 / 4)
    

class Kolmogorov1DTemp(KolmogorovTemp, Spectrum1D):
    """
    One-dimensional Kolmogorov turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, C, p=2.0/3.0):
        """
        One-dimensional spectral density density :math:`V(k)`.
        
        :param k: :math:`k`
        :param C: :math:`C`
        :param p: :math:`p = 2/3`
        
        The spectral density function is given by
        
        .. math:: V(k) = C^2 \\frac{\\Gamma(p+1)}{2\\pi} \\sin{\\left(\\frac{1}{2}\\pi p\\right)} \\left| k \\right|^{-p-1}
        
        See Salomons, equation I.35.
        """
        return C**2.0 * gamma(p+1.0)/(2.0*np.pi) * np.sin(0.5*np.pi*p)*np.abs(k)**(-p-1.0)
    

class VonKarman1DTemp(VonKarmanTemp, Spectrum1D):
    """
    One-dimensional Von Karman turbulence spectrum supporting temperature fluctuations.
    """

    @staticmethod
    def spectral_density_function(k, a, mu_0):
        """
        One-dimensional spectral density function :math:`V(k)`.
        
        :param r: :math:`r`
        :param a: :math:`a`
        :param mu_0: :math:`\\mu_0`

        The spectral density function is given by
        
        .. math:: V(k) = \\mu_0 \\frac{\\Gamma(5/6)}{\\Gamma(1/3) \\pi^{1/2}} \\frac{a}{\\left( 1 + k^2 a^2 \\right)^{5/6}}
        
        See Salomons, equation I.41.
        """
        return mu_0 * gamma(5.0/6.0) / (gamma(1.0/3.0)*np.sqrt(np.pi)) * a / (1.0 + k**2.0 * a**2.0)**(5.0/6.0)
        

class Gaussian2DTemp(GaussianTemp, Spectrum2D):
    """
    Two-dimensional Gaussian turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, a, mu_0):
        """
        Two-dimensional spectral density :math:`F(k)`.
        
        .. math:: F(k) =  \\mu_0^2 \\frac{ a^2 }{4 \\pi} \\exp{(-k^2 a^2 / 4)}
        
        See Salomons, equation I.31.
        """
        return mu_0**2.0 * a**2.0 / (4.0 *np.pi) * np.exp(-k**2.0 * a**2.0 / 4)


class Kolmogorov2DTemp(KolmogorovTemp, Spectrum2D):
    """
    Two-dimensional Kolmogorov turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, C, p=2.0/3.0):
        """
        Two-dimensional spectraldensity density :math:`F(k)`.
        
        .. math:: F(k) = C^2 \\frac{\\Gamma^2(0.5 p + 1) 2^p}{2 \\pi^2} \\sin{\\left(\\frac{1}{2}\\pi p\\right)} \\left| k \\right|^{-p-2}
        
        See Salomons, equation I.36.
        """
        return C**2.0 * gamma(0.5*p+1.0)*2.0**p/(2.0*np.pi**2.0) * np.sin(0.5*np.pi*p)*np.abs(k)**(-p-2.0)    


class VonKarman2DTemp(VonKarmanTemp, Spectrum2D):
    """
    Two-dimensional Von Karman turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, a, mu_0):
        """
        Two-dimensional spectral density function :math:`F(k)`.
        
        .. math:: F(k) = \\mu_0^2 \\frac{\\Gamma(8/6)}{\\Gamma(1/3) \\pi} \\frac{a^2}{\\left( 1 + k^2 a^2 \\right)^{8/6}}
        
        See Salomons, equation I.42.
        """
        return mu_0**2.0 * gamma(8.0/6.0) / (gamma(1.0/3.0)*np.pi) * a**2 / (1.0 + k**2.0 * a**2.0)**(8.0/6.0)
    

class Gaussian3DTemp(GaussianTemp, Spectrum3D):
    """
    Three-dimensional Gaussian turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, a, mu_0):
        """
        Three-dimensional spectral density :math:`\\Phi(k)`.
        
        .. math:: \\Phi(k) = \\mu_0^2 \\frac{a^3}{8\\pi^{3/2}} \\exp{(-k^2 a^2 /4)}
        
        See Salomons, equation I.32.
        """
        return mu_0**2.0 * a**3.0 * np.exp(-k**2.0*a**2.0 / 4)
    
    
class Kolmogorov3DTemp(KolmogorovTemp, Spectrum3D):
    """
    Three-dimensional Kolmogorov turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, C, p=2.0/3.0):
        """
        Three-dimensional spectral density density :math:`\\Phi(k)`.
        
        .. math:: \\Phi(k) = C^2 \\frac{\\Gamma(p+2)}{4\\pi^2} \\sin{\\left(\\frac{1}{2}\\pi p\\right)} \\left| k \\right|^{-p-3}
        
        See Salomons, equation I.37.
        """
        return C**2.0 * gamma(p+2.0)/(4.0*np.pi**2.0) * np.sin(0.5*np.pi*p)*np.abs(k)**(-p-3.0)
    
class VonKarman3DTemp(VonKarmanTemp, Spectrum3D):
    """
    Three-dimensional Von Karman turbulence spectrum supporting temperature fluctuations.
    """
    
    @staticmethod
    def spectral_density_function(k, a, mu_0):
        """
        Three-dimensional spectral density function :math:`\\Phi(k)`.
        
        .. math:: \\Phi(k) = \\mu_0 \\frac{\\Gamma(11/6)}{\\Gamma(1/3) \\pi^{3/2}} \\frac{a^3}{\\left( 1 + k^2 a^2 \\right)^{11/6}}
        
        See Salomons, equation I.43.
        """
        return mu_0 * gamma(11.0/6.0) / (gamma(1.0/3.0)*np.pi**(1.5)) * a**3 / (1.0 + k**2.0 * a**2.0)**(11.0/6.0)
    


class Gaussian2DTempWind(GaussianTempWind, Spectrum2D):
    """
    Two-dimensional Gaussian turbulence spectrum supporting temperature and wind fluctuations.
    """
    
    
    
    
    @staticmethod
    def spectral_density_function(k, theta, plane, a, sigma_T, T_0, sigma_mu, c_0):
        """
        Two-dimensional spectral density function :math:`F(k)`.
        
        
        :param k: :math:`k`
        :param plane: Tuple indicating which planes to consider.
        
        
        The spectral density is calculated according to
        
        .. math:: F(k_x, k_y) = F(k_x, k_z) = \\frac{a^2}{4 \\pi} \\left( \\frac{\\sigma_T^2}{4 T_0^2} + \\frac{\\sigma_{\\mu}^2  [k_z^2 a^2 + 2]  }{4 c_0^2} \\right) \\exp{(-k^2 a^2 / 4)}
        
        or
        
        .. math:: F(k_y, k_z) = \\frac{a^2}{4 \\pi} \\left( \\frac{\\sigma_T^2}{4 T_0^2} + \\frac{\\sigma_{\\mu}^2  [k^2 a^2 + 2]  }{4 c_0^2} \\right) \\exp{(-k^2 a^2 / 4)}
        
        depending on the chosen plane.
        
        
        See Salomons, page 215, and equation I.49 and I.50.
        """
        
        if plane == (1,0,1):    # xz-plane
            k_x = k * np.cos(theta)
            k_z = k * np.sin(theta)
            k = (k_x**2.0 + k_z**2.0)**(0.5)
            return a**2.0/(4.0*np.pi) * ( (sigma_T/(2.0*T_0))**2.0 + sigma_mu**2.0/(4.0*c_0**2.0)*(k_z**2.0*a**2.0+1)) * np.exp(-k**2.0*a**2.0/4)
        
        elif plane == (1,1,0):  # xy-plane
            k_x = k * np.cos(theta)
            k_y = k * np.sin(theta)
            k = (k_x**2.0 + k_y**2.0)**(0.5)
            return a**2.0/(4.0*np.pi) * ( (sigma_T/(2.0*T_0))**2.0 + sigma_mu**2.0/(4.0*c_0**2.0)*(k_y**2.0*a**2.0+1)) * np.exp(-k**2.0*a**2.0/4)
        
        elif plane == (0,1,1):  # yz-plane
            k_y = k * np.cos(theta)
            k_z = k * np.sin(theta)
            k = (k_y**2.0 + k_z**2.0)**(0.5)
            return a**2.0/(4.0*np.pi) * ( (sigma_T/(2.0*T_0))**2.0 + sigma_mu**2.0/(4.0*c_0**2.0)*(k**2.0*a**2.0+1)) * np.exp(-k**2.0*a**2.0/4)
        
        else:
            raise ValueError("Incorrect wavenumbers given.")
        
        

class Kolmogorov2DTempWind(KolmogorovTempWind, Spectrum2D):
    """
    Two-dimensional Kolmogorov turbulence spectrum support temperature and wind fluctuations.
    """
    pass

class VonKarman2DTempWind(VonKarmanTempWind, Spectrum2D):
    """
    Two-dimensional Von Karman turbulence spectrum supporting temperature and wind fluctuations.
    """

    @staticmethod
    def spectral_density_function(k, theta, plane, c_0, T_0, C_v, C_T, L, A):
        """
        Two-dimensional spectral density function :math:`F(k)`.
        
        :param k: Wavenumber :math:`k`
        :param c_0: :math:`c_0`
        :param T_0: :math:`T_0`
        :param C_v: :math:`C_v`
        :param C_T: :math:`C_T`
        :param L: :math:`L`
        
        See Salomons, equation I.53.
        """
        K_0 = 2.0 * np.pi / L
        
        if plane == (1,0,1):    # xz-plane
            k_var = k*np.sin(theta)
        
        elif plane == (1,1,0):  # xy-plane
            k_var = k*np.sin(theta)
        
        elif plane == (0,1,1):  # yz-plane
            k_var = k
        
        f1 = A / (k**2.0 + K_0**2.0 )**(8.0/6.0)
        f2 = gamma(1.0/2.0)*gamma(8.0/6.0) / gamma(11.0/6.0) * C_T**2.0/(4.0*T_0**2.0)
        f3 = gamma(3.0/2.0)*gamma(8.0/6.0)/gamma(17.0/6.0) + k_var**2.0/(k**2.0+K_0**2.0) * gamma(1.0/2.0)*gamma(14.0/6.0)/gamma(17.0/6.0)
        f4 = 22.0*C_v**2.0/(12.0*c_0**2.0)
        
        return f1 * (f2 + f3 * f4) 

class Gaussian3DTempWind(GaussianTempWind, Spectrum3D):
    """
    Three-dimensional Von Karman turbulence spectrum supporting temperature and wind fluctiations.
    """
    
    #@staticmethod
    #def spectral_density_function():
        #"""
        #Three-dimensional spectral density function :math:`\\Phi(k_x, k_y, k_z)`.
        
        #See Salomons, I.51.
        #"""
        #raise NotImplementedError
    
class Comparison(object):
    """
    Compare turbulence spectra.
    """
    
    def __init__(self, items):
        
        self.items = items
        """
        Turbulence spectra.
        """
    
    def plot_mode_amplitudes(self, filename=None):
        """
        Create a plot of the mode amplitudes for all turbulence spectra.
        """
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        
        for item in self.items:
            ax.plot(item.wavenumber, item.mode_amplitude(), label=item.__class__.__name__)
        
        ax.set_xlabel(r'$k$ in $\mathrm{m}^{-1}$')
        ax.set_ylabel(r'$G$')
        ax.grid()
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()

    
    
    def plot_spectral_density(self, filename=None):
        """
        Plot the spectral density.
        """

        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        
        for item in self.items:
            ax.loglog(item.wavenumber, item.spectral_density(), label=item.__class__.__name__)
        
        ax.set_xlabel(r'$k$ in $\mathrm{m}^{-1}$')
        ax.set_ylabel(r'$F$')
        ax.grid()
        ax.legend()
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()
    
    
    
class Field2D(object):
    """
    Refractive index field.
    """
    
    mu = None
    """
    Refractive index.
    """
    
    def __init__(self, x, y, z, spatial_resolution, spectrum):
        
        self.x = x
        """
        Size of field in x-direction.
        """
        self.y = y
        """
        Size of field in y-direction.
        """
        self.z = z
        """
        Size of field in z-direction.
        """
        self.spatial_resolution = spatial_resolution
        """
        Spatial resolution.
        """
        self.spectrum = spectrum
        """
        Spectrum.
        """
    
    def randomize(self):
        """
        Create new random values. This is a shortcut to :meth:`Spectrum2D.randomize()`.
        """
        self.spectrum.randomize()
        return self
    
    #@numba.autojit
    def generate(self):
        """
        Create a random realization of the refractive-index fluctuations. To actually create a random field, call :meth:`randomize` first.
        
        .. math:: \\mu(r) = \\sqrt{4\\pi \\Delta k} \\sum_n \\cos{\\left( \\mathbf{k}_n \cdot \\mathbf{r} + \\alpha_n \\right)} \\sqrt{F(\\mathbf{k_n} k_n}
        
        """
    
        r = self.x
        z = self.z
        
        r = np.arange(np.ceil(r/self.spatial_resolution)) * self.spatial_resolution
        z = np.arange(np.ceil(z/self.spatial_resolution)) * self.spatial_resolution
        
        #r = np.arange(0.0, r, self.spatial_resolution)
        #z = np.arange(0.0, z, self.spatial_resolution)
        
        delta_k = self.spectrum.wavenumber_resolution
        
        mu = list()
        
        mode_amplitudes = self.spectrum.mode_amplitude()
        
        for n, G, theta_n, alpha_n in zip(self.spectrum.modes, mode_amplitudes, self.spectrum.theta, self.spectrum.alpha):
            
            k_n = n * delta_k

            k_nr = k_n * np.cos(theta_n)    # Wavenumber component
            k_nz = k_n * np.sin(theta_n)    # Wavenumber component
            
            #r_mesh, z_mesh = np.meshgrid(r, z, indexing='ij')
            r_mesh, z_mesh = np.meshgrid(r, z)
            r_mesh = r_mesh.T
            z_mesh = z_mesh.T
            
            #k_n_v = np.vstack( , k_n * np.sin(theta))
            
            mu_n = G * np.cos(r_mesh * k_nr + z_mesh * k_nz + alpha_n)
            mu.append(mu_n)
        
        self.mu = sum(mu)
        return self
        
    
    def plot(self, filename=None):
        """
        Plot the field.
        """
        
        if self.mu is None:
            raise ValueError("Need to calculate the refractive index first.")
        
        r = self.x
        z = self.z
        r = np.arange(np.ceil(r/self.spatial_resolution)) * self.spatial_resolution
        z = np.arange(np.ceil(z/self.spatial_resolution)) * self.spatial_resolution
        
        fig = plt.figure(figsize=(16, 12), dpi=80)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_title("Refractive-index field")
        plot = ax.pcolormesh(r, z, self.mu.T)
        ax.set_xlabel(r'$r$ in m')
        ax.set_ylabel(r'$z$ in m')
        c = fig.colorbar(plot)
        c.set_label(r'Refractive-index fluctuation $\mu$')
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()




