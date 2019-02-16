import numpy as np
from acoustics.atmosphere import Atmosphere

import sys
sys.path.append('..')
from get_data_path import data_path


class TestAtmosphere:
    def test_standard_atmosphere(self):

        a = Atmosphere()
        """Default values."""
        assert (a.temperature == 293.15)
        assert (a.pressure == 101.325)
        assert (a.relative_humidity == 0.0)
        """Calculated values belonging to default values."""
        assert (abs(a.soundspeed - 343.2) < 1.0e-9)
        assert (abs(a.saturation_pressure - 2.33663045) < 1.0e-8)
        assert (abs(a.molar_concentration_water_vapour - 0.0) < 1.0e-9)
        assert (abs(a.relaxation_frequency_nitrogen - 9.0) < 1.0e-9)
        assert (abs(a.relaxation_frequency_oxygen - 24.0) < 1.0e-9)

    def test_attenuation_coefficient(self):
        """Test data is all at an air pressure of one standard atmosphere, 101.325 Pa."""
        data = np.loadtxt(data_path() + 'absorption_coefficient.csv', skiprows=1, delimiter=',')

        f = np.array([
            50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
            1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0
        ])

        for row in data:
            temperature = 273.15 + row[0]  # Degrees Celsius to Kelvin
            relative_humidity = row[1]
            alpha = row[2:] / 1000.0  # Given in dB/km while we calculate in dB/m.

            assert (f.shape == alpha.shape)

            a = Atmosphere(temperature=temperature, relative_humidity=relative_humidity)
            calculated_alpha = a.attenuation_coefficient(f)

            np.testing.assert_array_almost_equal(alpha, calculated_alpha, decimal=2)
