"""
IEC 61672-1:2013
================

IEC 61672-1:2013 gives electroacoustical performance specifications for three 
kinds of sound measuring instruments [IEC61672]_:

- time-weighting sound level meters that measure exponential-time-weighted, frequency-weighted sound levels;
- integrating-averaging sound level meters that measure time-averaged, frequency-weighted sound levels; and
- integrating sound level meters that measure frequency-weighted sound exposure levels. 

.. [IEC61672] http://webstore.iec.ch/webstore/webstore.nsf/artnum/048669!opendocument

.. inheritance-diagram:: acoustics.standards.iec_61672_1_2013

"""

NOMINAL_FREQUENCIES = np.array([10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 
                                100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 
                                800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 
                                6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
                                ])
"""
Nominal frequencies.
"""

WEIGHTING_A = np.array([-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, 
                        -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9,
                        -0.8, 0.0, +0.6, +1.0, +1.2, +1.3, +1.2, +1.0, +0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
                        ])
"""
Frequency weighting A.
"""

WEIGHTING_C = np.array([-14.3, -11.2,-8.5, -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5, -0.3, 
                        -0.2, -0.1, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,0,  0.0,  
                        -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.4, -6.2, -8.5, -11.2]) 
"""
Frequency weighting C.
"""

WEIGHTING_Z = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
"""
Frequency weighting Z.
"""
