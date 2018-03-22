"""
Perform a convolution with a linear time-variant system.
"""

import numpy as np
from acoustics.signal import convolve as convolveLTV
from scipy.signal import convolve as convolveLTI

def main():
    
    """Input signal"""
    u = np.array([1, 2, 3, 4, 3, 2, 1], dtype='float')
    
    """Impulse responses where each column represents an impulse response."""
    C = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4]
        ], dtype='float')

    """The array C represents here a linear time-invariant system."""
    y_ltv = convolveLTV(u, C)
    y_lti = convolveLTI(u, C[:,0])
    """Check whether the output is indeed the same."""
    print( np.all(y_ltv == y_lti) )
    print('LTV:' + str(y_ltv))
    print('LTI:' + str(y_lti))
    
    """Let's now use a time-variant system."""
    C = np.array([
        [1, 2, 3, 4, 3, 2, 1],
        [2, 3, 4, 5, 4, 3, 2],
        [3, 4, 5, 6, 5, 4, 3],
        [4, 5, 6, 7, 6, 5, 4]
        ], dtype='float')
    
    y_ltv = convolveLTV(u, C)
    y_lti = convolveLTI(u, C[:,0])  # We use now only the first impulser response for the LTI.
    """Is the result still equal?"""
    print( np.all( y_ltv == y_lti ) )
    print('LTV:' + str(y_ltv))
    print('LTI:' + str(y_lti))
    


if __name__ == '__main__':
    main()