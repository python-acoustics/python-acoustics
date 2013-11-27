"""
Class and functions to work with directivity.

The following conventions are used within this module:

* The angle :math:`\\theta` has a range :math:`[0, \\pi]`.
* The angle :math:`\\phi` has a range :math:`[-pi, pi]`.

"""
from __future__ import division

import numpy as np
import abc
from scipy.interpolate import interp2d as interpolate
from mayavi import mlab

def cardioid(theta, a=1.0, k=1.0):
    """
    A cardioid pattern.
    """
    return np.abs( a + a * np.cos(k*theta) )

def figure_eight(theta):
    """
    A figure-of-eight pattern.
    """
    return np.abs( np.cos(theta) )


def spherical_to_cartesian(r, theta , phi):
    """
    Convert spherical coordinates to cartesian coordinates.
    
    .. math:: x = r \\sin{\\theta}\\cos{\\phi}
    .. math:: y = r \\sin{\\theta}\\sin{\\phi}
    .. math:: z = r \\cos{\\theta}
    """
    return (
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
        )
    
def cartesian_to_spherical(x, y, z):
    """
    Convert cartesian coordinates to spherical coordinates.
    
    .. math:: r = \\sqrt{\\left( x^2 + y^2 + z^2 \\right)}
    .. math:: \\theta = \\arccos{\\frac{z}{r}}
    .. math:: \\phi = \\arccos{\\frac{y}{x}}
    """
    r = np.sqrt(np.sum(np.abs(np.vstack((x,y,z)))**2.0, axis=0))
    return r, np.arccos(z/r), np.arctan(y/x)


#def spherical_to_cartesian(spherical):
    #"""
    #Convert spherical coordinates to cartesian coordinates.
    
    #.. math:: x = r \\sin{\\theta}\\cos{\\phi}
    #.. math:: y = r \\sin{\\theta}\\sin{\\phi}
    #.. math:: z = r \\cos{\\theta}
    #"""
    #r = spherical[0,:]
    #theta = spherical[1,:]
    #phi = spherical[2,:]
    #return np.vstack((
        #r * np.sin(theta) * np.cos(phi),
        #r * np.sin(theta) * np.sin(phi),
        #r * np.cos(theta)
        #))
    

#def cartesian_to_spherical(cartesian):
    #"""
    #Convert cartesian coordinates to spherical coordinates.
    
    #.. math:: r = \\sqrt{\\left( x^2 + y^2 + z^2 \\right)}
    #.. math:: \\theta = \\arccos{\\frac{z}{r}}
    #.. math:: \\phi = \\arccos{\\frac{y}{x}}
    #"""
    #x = spherical[0,:]
    #y = spherical[1,:]
    #z = spherical[2,:]
    #r = np.sqrt(np.sum(np.abs(cartesian)**2, axis=0)),
    #return np.vstack((
        #r,
        #np.arccos(z/r),
        #np.arctan(y/x)
        #))


class Directivity(object):
    """
    Abstract directivity class.
    
    This class defines several methods to be implemented by subclasses.
    """
    
    
    def __init__(self, rotation=None):
        
        self.rotation = rotation if rotation else np.array([1.0, 0.0, 0.0]) # X, Y, Z rotation
        """
        Rotation of the directivity pattern.
        """

    @abc.abstractmethod
    def _directivity(self, theta, phi):
        """
        This function should return the directivity as function of :math:`\\theta` and :math:`\\phi`.
        """
        pass

    def _undo_rotation(self, theta, phi):
        """
        Undo rotation.
        """
        pass
        

    def using_spherical(self, r, theta, phi, include_rotation=True):
        """
        Return the directivity for given spherical coordinates.
        
        :param x: Spherical coordinate.
        """
        
        """
        Correct for rotation!!!!
        """
        return self._directivity(theta, phi)    
    
    def using_cartesian(self, x, y, z, include_rotation=True):
        """
        Return the directivity for given cartesian coordinates.
        
        :param x: Cartesian coordinate.
        """
        return self.using_spherical(*cartesian_to_spherical(x, y, z))
    
    def plot(self, filename=None, include_rotation=True):
        """
        Directivity plot.
        
        :param filename: Filename
        :param include_rotation: Apply the rotation to the directivity. By default the rotation is applied in this figure.
        """
        phi = np.linspace(-np.pi, +np.pi, 50)
        theta = np.linspace(0.0, np.pi, 50)
        
        theta_n, phi_n = np.meshgrid(theta, phi)    # Create a 2-D mesh
    
        d = self._directivity(theta_n, phi_n.ravel)
        
        #fig = plt.figure()
        #ax0 = fig.add_subplot(111, projection='3d')
        #ax0.set_title('Directivity')
        
        #ax0.pcolormesh(u*180.0/np.pi, v*180.0/np.pi, r)
        (x, y, z) = spherical_to_cartesian(d, theta_n, phi_n)
        
        fig = mlab.figure()
         
        s = mlab.mesh(x,y,z)
        
        fig.add(s)
        
        mlab.axes()
        mlab.outline()
        mlab.show()
        
        
        #ax0.plot_wireframe(x*180.0/np.pi, y*180.0/np.pi, z)
        #ax0.set_xlabel(r'Latitude $u$ in degree')
        #ax0.set_ylabel(r'Longitude $v$ in degree')
        #ax0.grid()
        
        #if filename:
            #fig.savefig(filename)
    


class Omni(Directivity):
    """
    Class to work with omni-directional directivity.
    """
    
    def _directivity(self, theta, phi):
        """
        Directivity
        """
        return np.ones_like(theta)
    

class Cardioid(Directivity):
    """
    Cardioid directivity.
    """
    
    def _directivity(self, theta, phi):
        """
        Directivity
        """
        return cardioid(theta)
    
class FigureEight(Directivity):
    """
    Figure of eight directivity.
    """
    
    def _directivity(self, theta, phi):
        """
        Directivity
        """
        return figure_eight(theta)
        

class Custom(Directivity):
    """
    A class to work with directivity.
    """
    
    def __init__(self, theta=None, phi=None, r=None):
        """
        Constructor.
        """
        
        self.theta = phi
        """
        Latitude. 1-D array.
        """
        self.phi = phi
        """
        Longitude. 1-D array.
        """
        self.r = r
        """
        Magnitude or radius. 2-D array.
        """
    
    def _directivity(self, theta, phi):
        """
        Custom directivity.
        
        Interpolate the directivity given longitude and latitude vectors.
        """
        f = interpolate(self.theta, self.phi, self.r)
        
        return f(theta, phi)
