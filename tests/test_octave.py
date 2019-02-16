"""
Tests for :class:`Acoustics.Octave.Octave`
"""

from acoustics.octave import Octave
import numpy as np


class TestOctave():
    def test_interval(self):
        emin = 1.0
        emax = 4.0
        f = np.logspace(emin, emax, 50)

        o = Octave(interval=f)

        assert (o.fmin == 10.0**emin)
        assert (o.fmax == 10.0**emax)
        assert (len(o.n) == len(o.center))

        o.unique = True
        assert (len(o.n) == len(f))

    def test_minmax(self):
        fmin = 250.0
        fmax = 1000.0

        o = Octave(fmin=fmin, fmax=fmax)

        assert (len(o.center) == 3)  # 250, 500, and 1000 Hz
        assert (len(o.n) == 3)
