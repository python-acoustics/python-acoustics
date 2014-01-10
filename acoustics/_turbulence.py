"""
This file contains all abstract base classes related to atmospheric turbulence.
"""

import numpy as np
import matplotlib.pyplot as plt
import abc
import six

from scipy.special import gamma
from scipy.special import iv as bessel  # Modified Bessel function of the first kind.


class abstractstaticmethod(staticmethod):
    """
    Abstract static method.
    """
    __slots__ = ()
    def __init__(self, function):
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


    
@six.add_metaclass(abc.ABCMeta)
class Spectrum(object):
    """
    Abstract turbulence spectrum.
    """
    
    @property
    def x(self):
        """
        Length of field :math:`x`.
        """
        return self._sizes[0]
    
    
    @x.setter
    def x(self, x):
        self._sizes[0] = x
    
    @property
    def y(self):
        """
        Depth of field :math:`y`
        """
        return self._sizes[1]

    @y.setter
    def y(self, x):
        self._sizes[1] = x
    
    
    
    @property
    def z(self):
        """
        Height of field :math:`z`.
        """
        return self._sizes[2]
    
    @z.setter
    def z(self, x):
        self._sizes[2] = x
    
        
    #@property
    #def dimensions(self):
        #return np.count_nonzero(self._sizes)

    wavenumber_resolution = None
    """
    Wavenumber resolution
    """
    
    spatial_resolution = None
    """
    Spatial resolution.
    """
    
    max_mode_order = None
    """
    Maximum amount of modes to consider.
    """
    
    
    _required_attributes = ['x', 'y', 'z', 'wavenumber_resolution', 'spatial_resolution', 'max_mode_order']
    
    def _construct(self, attributes, *args, **kwargs):
        for attr in attributes:
            try:
                setattr(self, attr, kwargs[attr])
            except KeyError:
                raise ValueError(attr)
                #raise ValueError("Requires " + attr + ".")
        
    def __init__(self, *args, **kwargs):
        
        self._sizes = np.zeros(3)
        """
        Length of the field in each direction.
        """
        
        missing = list()
        
        for cls in reversed(self.__class__.__mro__):
            try:
                self._construct(getattr(cls, '_required_attributes'), args, **kwargs)
            except AttributeError as error:
                pass
            except ValueError as error:
                missing.append(str(error))
        if missing:
            raise ValueError("Missing arguments: " + str(set(missing)))
        
        
        #super(Spectrum, self).__init__(*args, **kwargs)
        
        ##=0.0, z=0.0, d=0.0, N=10, a=1.0, mu_0=0.0010):
        
        #for arg in args.iteritems():
            #print arg
        
        #for kwarg in kwargs.iteritems():
            #print kwarg
        
        #attributes = ['wavenumber', 'spatial_resolution', 'max_mode_order']
        #self._construct(attributes, args, kwargs)
        
        
        #self._sizes = np.zeros(3)
        #"""
        #Length of the field in each direction.
        #"""
        
        #self.a = a
        #"""
        #Correlation length.
        #"""
        
        #self.wavenumber = wavenumber
        #"""
        #Wavenumbers to use for the calculation.
        #"""
        
    @property
    def modes(self):
        """
        Vector of modes.
        """
        return np.arange(0, self.max_mode_order)
    
    @property
    def wavenumber(self):
        """
        Vector of wavenumbers.
        """
        return self.wavenumber_resolution * self.modes

    @abc.abstractmethod
    def mode_amplitude():
        pass

    @abc.abstractmethod
    def field(self):
        pass

    @abc.abstractmethod
    def spectral_density(self):
        """
        Spectral density of this object.
        """
        pass
    
    @abstractstaticmethod
    def spectral_density_function():
        """
        The spectral density function that is used in this model.
        """
        pass
    
    
    ##def correlation(self):
        ##"""
        ##Correlation for this object.
        ##"""
        ##return self.correlation_function(self.mu_0, self.a, self.r)
    
    ##def structure(self):
        ##"""
        ##Structure for this object.
        ##"""
        ##return self.structure_function(self.mu_0, self.a, self.r)

    ##def spectral_density(self):
        ##"""
        ##Spectral density for this object.
        ##"""
        ##return self.spectral_density_function(self.mu_0, self.a, self.wavenumber)
        #return getattr(self, 'spectral_density_function_'+self.ndim+'d')(self.mu_0, self.a, self.wavenumber)   




@six.add_metaclass(abc.ABCMeta)
class Spectrum1D(Spectrum):
    """
    Abstract class for one-dimensional turbulence spectra.
    """
    
    NDIM = 1
    """
    Amount of dimensions.
    """
    
    
@six.add_metaclass(abc.ABCMeta)
class Spectrum2D(Spectrum):
    """
    Abstract class for two-dimensional turbulence spectra.
    """
    
    NDIM = 2
    """
    Amount of dimensions.
    """
    
    #def __init__(self, *args, **kwargs):
        
        #super(Spectrum2D, self).__init__(*args, **kwargs)
        #attributes = ['r', 'z']
        #self._construct(attributes, args, kwargs)
        
    
    #def __init__(self, wavenumber, max_mode_order, r, z, *args, **kwargs):
        #Spectrum.__init__(self, wavenumber, max_mode_order)

    _max_mode_order = None
    
    @property
    def wavenumber_resolution(self):
        return self._wavenumber_resolution
    
    @wavenumber_resolution.setter
    def wavenumber_resolution(self, x):
        self._wavenumber_resolution = x
        self.randomize()
    
    @property
    def max_mode_order(self):
        return self._max_mode_order
    
    @max_mode_order.setter
    def max_mode_order(self, x):
        self._max_mode_order = x
        self.randomize()
    
    
    @property
    def plane(self):
        """
        Tuple indicating the plane that is modelled.
        """
        return self._sizes.astype(bool)
        
    
    def mode_amplitude(self):
        """
        Mode amplitudes :math:`G(\\mathbf{k})`.
        
        :rtype: A `n`-dimensional array where `n` is equal to the amount of dimensions of `k_n`.
        
        The mode amplitudes are calculating using
        
        .. math:: G (\\mathbf{k}_n ) = \\sqrt{4 \\pi \\Delta k F(\\mathbf{k}_n) \\mathbf{k}_n} 
        
        where :math:`\\mathbf{k}_n` are the wavenumber, :math:`\\Delta k` the wavenumber resolution, 
        and :math:`F` the spectral density.
        
        See Salomons, below equation J.24.
        
        """
        n = np.arange(0, self.max_mode_order)
        return np.sqrt(4.0 * np.pi * self.wavenumber_resolution  * self.spectral_density()  * self.wavenumber)
    
    #def _field(self):
        """Numba version??"""
        
        #r_vector = np.arange(0.0, self.r, self.spatial_resolution)
        #z_vector = np.arange(0.0, self.z, self.spatial_resolution)
            
        #shape = (len(self.r_vector), len(self.z_vector), len(self.wavenumber))
        #mu = np.zeros(shape)
        
        #for n in range(self.max_mode_order):
            #for r in r_vector:
                #for z in z_vector:
                    #for k in self.wavenumber:
                        #mu += 
                        
        #return mu
    
    
    def randomize(self):
        """
        Create new random values for :math:`\\theta_n` and :math:`\\alpha_n`.
        
        :rtype: self
        
        .. warning:: This function should always be called before :meth:`field` when a new random field should be generated.
        
        .. note:: This function is called whenever :attr:`max_mode_order` or :attr:`wavenumber_resolution` is changed.
        
        """
        self.alpha = np.random.random_sample(self.max_mode_order) * np.pi # Create random alpha_n
        self.theta = np.random.random_sample(self.max_mode_order) * np.pi # Create random alpha_n
        return self
    
    #@numba.autojit
    def field(self):
        """
        Create a random realization of the refractive-index fluctuations. To actually create a random field, call :meth:`randomize` first.
        
        .. math:: \\mu(r) = \\sqrt{4\\pi \\Delta k} \\sum_n \\cos{\\left( \\mathbf{k}_n \cdot \\mathbf{r} + \\alpha_n \\right)} \\sqrt{F(\\mathbf{k_n} k_n}
        
        """
    
        
        r, z = self._sizes[self.plane]
        r = np.arange(0.0, r, self.spatial_resolution)
        z = np.arange(0.0, z, self.spatial_resolution)
        
        delta_k = self.wavenumber_resolution
        
        
        mu = list()
        
        mode_amplitudes = self.mode_amplitude()
        
        for n, G, theta_n, alpha_n in zip(self.modes, mode_amplitudes, self.theta, self.alpha):
            
            #alpha_n = np.random.random_sample() * np.pi # Create random alpha_n
            #theta_n = np.random.random_sample() * np.pi # Create random theta_n
            k_n = n * delta_k

            k_nr = k_n * np.cos(theta_n)    # Wavenumber component
            k_nz = k_n * np.sin(theta_n)    # Wavenumber component
            
            r_mesh, z_mesh = np.meshgrid(r, z)
            
            #k_n_v = np.vstack( , k_n * np.sin(theta))
            
            
            mu_n = G * np.cos(r_mesh * k_nr + z_mesh * k_nz + alpha_n)
            mu.append(mu_n)
        
        return sum(mu)
    
    def plot_field(self, filename=None):
        """
        Calculate and plot a random field.
        """
        r, z = self._sizes[self.plane]
        r = np.arange(0.0, r, self.spatial_resolution)
        z = np.arange(0.0, z, self.spatial_resolution)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Refractive-index field")
        plot = ax.pcolormesh(r, z, self.field())
        ax.set_xlabel(r'$r$ in m')
        ax.set_ylabel(r'$z$ in m')
        c = fig.colorbar(plot)
        c.set_label(r'Refractive-index fluctuation $\mu$')
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()

    def plot_mode_amplitudes(self, filename=None):
        """
        Calculate and plot mode amplitudes.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.set_title("Mode {}".format(n))

        ax.plot(self.wavenumber, self.mode_amplitude())
        ax.set_xlabel(r'$k$ in $\mathrm{m}^{-1}$')
        ax.set_ylabel(r'$G$')
        ax.grid()
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()


    def plot_correlation(self):
        """
        Plot the correlation function.
        """
        pass
    
    def plot_structure(self):
        """
        Plot the structure function.
        """
        pass
    
    def plot_spectral_density(self, filename=None):
        """
        Plot the spectral density.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(self.wavenumber, self.spectral_density())
        ax.set_xlabel(r'$k$ in $\mathrm{m}^{-1}$')
        ax.set_ylabel(r'$F$')
        ax.grid()
        
        if filename:
            fig.savefig(filename)
        else:
            fig.show()
        
        

@six.add_metaclass(abc.ABCMeta)
class Spectrum3D(Spectrum):
    """
    Abstract class for one-dimensional turbulence spectra.
    """
    
    NDIM = 3
    """
    Amount of dimensions.
    """


@six.add_metaclass(abc.ABCMeta)
class GaussianTemp(Spectrum):
    """
    Abstract class for Gaussian spectrum when only temperature fluctuations are considered.
    """
    
    _required_attributes = ['a', 'mu_0']
    
    
    a = None
    """
    Characteristic length :math:`a`.
    """
    
    mu_0 = None
    """
    The standard deviation of the refractive-index fluctuation :math:`\\mu` is :math:`\\mu_0`.
    """
    
    def spectral_density(self):
        return self.spectral_density_function(self.wavenumber, self.a, self.mu_0)
    
    
    @staticmethod
    def correlation_function(r, a, mu_0):
        """
        Correlation function :math:`B(r)`.
        
        :param r: :math:`r`
        :param a: :math:`a`
        :param mu_0: :math:`\\mu_0`
        
        The correlation function is given by
        
        .. math:: B(r) = \\mu_0^2 \\exp{(-r^2/a^2)}
        
        See Salomons, equation I.28.
        """
        return mu_0**2.0 * np.exp(-r**2.0/a**2.0)

    @staticmethod
    def structure_function(r, a, mu_0):
        """
        Structure function :math:`D(r)`.
        
        :param r: :math:`r`
        :param a: :math:`a`
        :param mu_0: :math:`\\mu_0`

        The structure function is given by
        
        .. math:: D(r) = 2 \\mu_0^2 \\left[ 1 - \\exp{(-r^2/a^2)} \\right]
        
        See Salomons, equation I.29.
        """
        return 2.0 * mu_0**2.0 * (1.0 - np.exp(-r**2/a**2))
    

@six.add_metaclass(abc.ABCMeta)
class KolmogorovTemp(object):
    """
    Abstract class for Kolmogorov spectrum when only temperature fluctuations are considered.
    """
    
    def spectral_density(self):
        return self.spectral_density_function(self.wavenumber, self.C)
    
    
    @staticmethod
    def correlation_function():
        """
        Correlation function is not defined for Kolmogorov spectrum.
        """
        raise AttributeError("Correlation function is not defined for Kolmogorov spectrum.")

    @staticmethod
    def structure_function(r, C, p=2.0/3.0):
        """
        Structure function :math:`D(r)`.
        
        :param r: :math:`r`
        :param C: :math:`C`
        
        The structure function is given by
        
        .. math:: D(r) = C^2 r^p
        
        where :math:`p = 2/3`.
        
        See Salomons, equation I.34.
        """
        return C**2.0 * r**p
    

@six.add_metaclass(abc.ABCMeta)
class VonKarmanTemp(Spectrum):
    """
    Abstract class for Von Karman spectrum when only temperature fluctuations are considered.
    """ 
    
    _required_attributes = ['a', 'mu_0']
    
    
    a = None
    """
    Characteristic length :math:`a`.
    """
    
    mu_0 = None
    """
    The standard deviation of the refractive-index fluctuation :math:`\\mu` is :math:`\\mu_0`.
    """
    
    def spectral_density(self):
        return self.spectral_density_function(self.wavenumber, self.a, self.mu_0)
    
    
    @staticmethod
    def correlation_function(r, a, mu_0):
        """
        Correlation function :math:`B(r)`.
        
        :param r: :math:`r`
        :param a: :math:`a`
        :param mu_0: :math:`\\mu_0`

        The correlation function is given by
        
        .. math:: B(r) = \\mu_0^2 \\frac{2^{2/3}}{\\Gamma(1/3)} \\left(\\frac{r}{a}\\right)^{1/3} K_{1/3} \\left(\\frac{r}{a}\\right)
        
        See Salomons, equation I.39.
        """
        return mu_0**2.0 * 2.0**(2.0/3.0) / gamma(1.0/3.0) * (r/a)**(1.0/3.0) * bessel(1.0/3.0, r/a)
    
    @staticmethod
    def structure_function(r, a, mu_0, smaller_than_factor=0.1):
        """
        Structure function :math:`D(r)`.
        
        :param r: :math:`r`
        :param a: :math:`a`
        :param mu_0: :math:`\\mu_0`
        :param smaller_than_factor: Factor
        
        
        .. math:: D(r) = 2 \\mu_0^2 \\left[ 1 - \\frac{2^{2/3}}{\\Gamma(1/3)} \\left(\\frac{r}{a}\\right)^{1/3} K_{1/3} \\left(\\frac{r}{a}\\right) \\right]
        
        When :math:`a \\ll r`, or 'r < smaller_than_factor * a'
        
        .. math:: D(r) \\approx \\mu_0^2 \\frac{\\sqrt{\\pi}}{\\Gamma(7/6)} \\left( \\frac{r}{a} \\right)^{2/3} 
        
        See Salomons, equation I.40.
        """
        return (r < smaller_than_factor * a) * \
               ( mu_0**2.0 * np.sqrt(np.pi)/gamma(7.0/6.0) * (r/a)**(2.0/3.0)  ) + \
               (r >= smaller_than_factor * a) * \
               ( mu_0**2.0 * (1.0 -  2.0**(2.0/3.0) / gamma(1.0/3.0) * (r/a)**(1.0/3.0) * bessel(1.0/3.0, r/a) )  )
                
        #if r < smaller_than_factor * a:
            #return mu_0**2.0 * np.sqrt(np.pi)/gamma(7.0/6.0) * (r/a)**(2.0/3.0)
        #else:
            #return mu_0**2.0 * (1.0 -  2.0**(2.0/3.0) / gamma(1.0/3.0) * (r/a)**(1.0/3.0) * bessel(1.0/3.0, r/a) )
    



@six.add_metaclass(abc.ABCMeta)
class GaussianTempWind(object):
    """
    Abstract class for Gaussian spectrum when both temperature and wind fluctuations are considered.
    """
    
    _required_attributes = ['a', 'sigma_T', 'T_0', 'sigma_nu', 'c_0']
    
    
    a = None
    """
    Characteristic length :math:`a`.
    """
    
    
    @staticmethod
    def r(x, y, z):
        """
        Distance :math:`r`.
        
        :param x: x
        :param y: y
        :param z: z
        
        .. math:: r = \\sqrt{x^2 + y^2 + z^2}
        
        """
        return (x**2.0 + y**2.0 + z**2.0)
        
    
    @staticmethod
    def rho(y, z):
        """
        Distance :math:`\\rho`.
        
        :param y: y
        :param z: z
        
        .. math:: \\rho = \\sqrt{y^2 + z^2}
        
        """
        return (x**2.0 + y**2.0)**0.5
    
    
    def spectral_density(self):
        return self.spectral_density_function(self.wavenumber, self.theta, tuple(self.plane), self.a, self.sigma_T, self.T_0, self.sigma_nu, self.c_0)
    
    @staticmethod
    def correlation_function(r, a, sigma_T, T_0, sigma_nu, c_0, rho):
        """
        Correlation function :math:`B(r)`.
        
        
        .. math:: B(x,y,z) = \\left[ \\frac{\\sigma_T^2}{4 T_0)^2} + \\frac{\\sigma_{\\nu}^2}{c_0^2} \\left( 1 - \\frac{\\rho^2}{a^2}  \\right)  \\right]  \\exp{\\left( -r^2/a^2 \\right)}
        
        See Salomons, equation I.48.
        """
        return (sigma_T/(2.0*T_0))**2.0 + (sigma_nu/c_0)**2.0 * (1.0 - (rho/a)**2.0) * np.exp(-(r/a)**2.0)

    


@six.add_metaclass(abc.ABCMeta)
class KolmogorovTempWind(object):
    """
    Abstract class for Kolmogorov spectrum when both temperature and wind fluctuations are considered.
    """
    pass

@six.add_metaclass(abc.ABCMeta)
class VonKarmanTempWind(object):
    """
    Abstract class for Von Karman spectrum when both temperature and wind fluctuations are considered.
    """
    
    _required_attributes = ['c_0', 'T_0', 'C_v', 'C_T', 'L']

    CONSTANT_A = 5.0 / (18.0 * np.pi * gamma(1.0/3.0))
    """
    Constant `A`.

    .. math:: A = 5 / [ 18 \\pi \\Gamma(1/3) ] \\approx 0.0330

    """

    def spectral_density(self):
        return self.spectral_density_function(self.wavenumber, self.theta, tuple(self.plane), self.c_0, self.T_0, self.C_v, self.C_T, self.L, self.CONSTANT_A)
        






