import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest

from acoustics.room import (t60_sabine, t60_eyring, t60_millington,
                            t60_fitzroy, t60_arau, t60_impulse)
from acoustics.room import mean_alpha, nrc
from acoustics.core.bands import octave, third


def setup_module(reverberation):
    reverberation.surfaces = np.array([240, 600, 500])
    reverberation.alpha = np.array([0.1, 0.25, 0.45])
    reverberation.alpha_bands = np.array([[0.1,   0.1, 0.1, 0.1],
                                          [0.25, 0.25, 0.25, 0.25],
                                          [0.45, 0.45, 0.45, 0.45]])
    reverberation.volume = 3000


def test_t60_sabine():
    calculated = t60_sabine(surfaces, alpha, volume)
    real = 1.211382149
    assert_almost_equal(calculated, real)


def test_t60_sabine_bands():
    calculated = t60_sabine(surfaces, alpha_bands, volume)
    real = np.array([1.211382149, 1.211382149, 1.211382149, 1.211382149])
    assert_array_almost_equal(calculated, real)


def test_t60_eyring():
    calculated = t60_eyring(surfaces, alpha, volume)
    real = 1.020427763
    assert_almost_equal(calculated, real)


def test_t60_eyring_bands():
    calculated = t60_eyring(surfaces, alpha_bands, volume)
    real = np.array([1.020427763, 1.020427763, 1.020427763, 1.020427763])
    assert_array_almost_equal(calculated, real)


def test_t60_millington():
    calculated = t60_millington(surfaces, alpha, volume)
    real = 1.020427763
    assert_almost_equal(calculated, real)


def test_t60_millington_bands():
    calculated = t60_millington(surfaces, alpha_bands, volume)
    real = np.array([1.020427763, 1.020427763, 1.020427763,
                     1.020427763])
    assert_array_almost_equal(calculated, real)


def test_t60_fitzroy():
    surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
    alpha_fitzroy = np.array([0.1, 0.1, 0.25, 0.25, 0.45, 0.45])
    calculated = t60_fitzroy(surfaces_fitzroy, alpha_fitzroy, volume)
    real = 0.699854185
    assert_almost_equal(calculated, real)


def test_t60_fitzroy_bands():
    surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
    alpha_bands_f = np.array([[0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                              [0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                              [0.1, 0.1, 0.25, 0.25, 0.45, 0.45]])
    calculated = t60_fitzroy(surfaces_fitzroy, alpha_bands_f, volume)
    real = np.array([0.699854185, 0.699854185, 0.699854185])
    assert_array_almost_equal(calculated, real)


def test_t60_arau():
    Sx = surfaces[0]
    Sy = surfaces[1]
    Sz = surfaces[2]
    calculated = t60_arau(Sx, Sy, Sz, alpha, volume)
    real = 1.142442931
    assert_almost_equal(calculated, real)


def test_t60_arau_bands():
    Sx = surfaces[0]
    Sy = surfaces[1]
    Sz = surfaces[2]
    calculated = t60_arau(Sx, Sy, Sz, alpha_bands, volume)
    real = np.array([1.142442931, 1.142442931, 1.142442931, 1.142442931])
    assert_array_almost_equal(calculated, real)


def test_mean_alpha_float():
    alpha = 0.1
    surface = 10
    calculated = mean_alpha(alpha, surface)
    real = 0.1
    assert_almost_equal(calculated, real)


def test_mean_alpha_1d():
    alpha = np.array([0.1, 0.2, 0.3])
    surfaces = np.array([20, 30, 40])
    calculated = mean_alpha(alpha, surfaces)
    real = 0.222222222
    assert_almost_equal(calculated, real)


def test_mean_alpha_bands():
    alpha = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    surfaces = np.array([20, 30, 40])
    calculated = mean_alpha(alpha, surfaces)
    real = np.array([0.222222222, 0.222222222, 0.222222222])
    assert_array_almost_equal(calculated, real)


def test_nrc_1d():
    alpha = np.array([0.1, 0.25, 0.5, 0.9])
    calculated = nrc(alpha)
    real = 0.4375
    assert_almost_equal(calculated, real)


def test_nrc_2d():
    alphas = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]])
    calculated = nrc(alphas)
    real = np.array([0.25, 0.55])
    assert_array_almost_equal(calculated, real)


@pytest.mark.parametrize("file_name, bands, rt, expected", [
    ('data/ir_sportscentre_omni.wav', octave(125, 4000), 't30',
     np.array([7.24027654, 8.47019681, 6.79466752,
               6.51780663, 4.79692643, 4.08912686])),
    ('data/ir_sportscentre_omni.wav', octave(125, 4000), 'edt',
     np.array([4.71644743, 5.94075422, 6.00702329,
               5.94062563, 5.03778274, 3.73465316])),
    ('data/living_room_1.wav', octave(63, 8000), 't30',
     np.array([0.27658574, 0.36466480, 0.30282462, 0.25946725, 0.22710926,
               0.21056449, 0.20445301, 0.18080435])),
    ('data/living_room_1.wav', octave(63, 8000), 't20',
     np.array([0.30418539, 0.36486166, 0.15138373, 0.15594470, 0.10192937,
               0.07587109, 0.14564938, 0.15231023])),
    ('data/living_room_1.wav', octave(63, 8000), 't10',
     np.array([0.18067203, 0.06121885, 0.10898306, 0.02377203, 0.03865264,
               0.02303814, 0.10484486, 0.07141563])),
    ('data/living_room_1.wav', octave(63, 8000), 'edt',
     np.array([0.27998887, 0.15885362, 0.07971810, 0.03710582, 0.02143263,
               0.00962214, 0.02179504, 0.01961945])),
    ('data/living_room_1.wav', third(100, 5000), 't30',
     np.array([0.28442960, 0.34634621, 0.25757467,
               0.31086982, 0.26658673, 0.37620645,
               0.34203975, 0.26774014, 0.21206741,
               0.24635442, 0.21050635, 0.23151149,
               0.19184106, 0.23050360, 0.25227970,
               0.20164536, 0.18413574, 0.21605655])),
    ('data/living_room_1.wav', third(100, 5000), 't20',
     np.array([0.20768170, 0.39155525, 0.18814558,
               0.17202362, 0.14049935, 0.20770758,
               0.32284992, 0.22108431, 0.10189647,
               0.10997154, 0.08072728, 0.12806066,
               0.07219205, 0.07380865, 0.08706586,
               0.12925068, 0.13712873, 0.17112382])),
    ('data/living_room_1.wav', third(100, 5000), 't10',
     np.array([0.21147617, 0.10531143, 0.13209715,
               0.16484339, 0.13578831, 0.04089752,
               0.11978640, 0.22370033, 0.02502991,
               0.02363114, 0.04751902, 0.05045455,
               0.01003664, 0.01754925, 0.03906140,
               0.08439052, 0.15418884, 0.09357333])),
    ('data/living_room_1.wav', third(100, 5000), 'edt',
     np.array([0.34939884, 0.31141886, 0.28158889,
               0.21030131, 0.13179537, 0.11555684,
               0.08503427, 0.11378130, 0.06413555,
               0.04531878, 0.04672024, 0.04721689,
               0.02432437, 0.01723772, 0.01637013,
               0.02170071, 0.02044526, 0.03593397])),
])
def test_t60_impulse(file_name, bands, rt, expected):
    calculated = t60_impulse(file_name, bands, rt)
    assert_array_almost_equal(calculated, expected, decimal=3)


def teardown_module(reverberation):
    pass
