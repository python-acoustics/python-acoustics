from acoustics.standards.iso_tr_25417_2007 import *


def test_sound_exposure_level():

    duration = 20.0
    fs = 44100.
    samples = int(fs * duration)

    signal = np.random.randn(samples)

    leq = equivalent_sound_pressure_level(signal)
    sel = sound_exposure_level(signal, fs)

    assert ((sel - leq) - 10.0 * np.log10(duration)) < 0.1
