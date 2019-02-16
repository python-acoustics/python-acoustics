import numpy as np
from acoustics import Signal
import pytest
import tempfile

import itertools as it
from acoustics.signal import EqualBand
import matplotlib.pyplot as plt

#def test_operator():

    #n = 10000
    #fs = 5000


class test_wav():
    """Test writing to and reading from wav file."""

    duration = 5.0
    fs = 10025
    samples = int(fs * duration)
    channels = 3

    signal = Signal(np.random.randn(channels, samples), fs)

    with tempfile.TemporaryFile() as file:
        signal.to_wav(file)
        signal = Signal.from_wav(file)
        assert signal.samples == samples
        assert signal.fs == fs
        assert signal.channels == channels


class TestSignal():

    # (channels, samples, sample rate)
    @pytest.fixture(params=[(1, 88200, 22050), (3, 88200, 22050), (3, 88200, 44100)])
    def signal(self, request):
        return Signal(np.random.randn(request.param[0], request.param[1]), request.param[2])

    def test_samples(self, signal):
        x = signal.samples

    def test_calibrate_to_scalar(self, signal):
        # Scalar decibel
        signal.calibrate_to(100.0)
        signal.copy().calibrate_to(100.0, inplace=True)

    def test_calibrate_to_channels(self, signal):
        # Value per channel. Note that [...,None] is required!
        signal.calibrate_to((np.ones(signal.channels) * 100.0)[..., None])
        signal.copy().calibrate_to((np.ones(signal.channels) * 100.0)[..., None], inplace=True)

    def test_calibrate_to_samples(self, signal):
        # Value per samples
        signal.calibrate_to(np.ones(signal.samples))
        signal.copy().calibrate_to(np.ones(signal.samples), inplace=True)

    def test_calibrate_to_samples_channels(self, signal):
        # Value per sample per channel
        signal.calibrate_to(np.ones(signal.shape))
        signal.copy().calibrate_to(np.ones(signal.shape), inplace=True)

    def test_calibrate_with(self, signal):
        calibration_signal_level = 50.0
        decibel = 94.0
        calibration_signal = Signal(np.random.randn(signal.samples), signal.fs).calibrate_to(calibration_signal_level)

        out = signal.calibrate_with(calibration_signal, decibel)
        assert ((out.leq() - signal.leq()).mean() - (decibel - calibration_signal_level)) < 0.01

    def test_channels(self, signal):
        x = signal.channels

    def test_duration(self, signal):
        x = signal.duration

    def test_decimate(self, signal):
        factor = 4
        decimated = signal.decimate(factor)
        assert (signal.fs / factor == decimated.fs)

    def test_upsample(self, signal):

        factor = 2
        assert (signal.upsample(factor).fs / signal.fs) == factor

    def test_gain_scalar(self, signal):
        gain = +20.0
        # `.all()` because of multichannel signals
        assert (np.abs(signal.gain(gain).leq() - (signal.leq() + gain)) < 0.01).all()
        assert (np.abs(signal.copy().gain(gain, inplace=True).leq() - (signal.leq() + gain)) < 0.01).all()

    def test_pick(self, signal):
        x = signal.pick(signal.duration * 0.1, signal.duration * 0.6)

    def test_times(self, signal):
        times = signal.times()

    def test_energy(self, signal):
        energy = signal.energy()

    def test_power(self, signal):
        power = signal.power()

    def test_ms(self, signal):
        ms = signal.ms()

    def test_rms(self, signal):
        rms = signal.rms()

    def test_correlate(self, signal):
        signal = signal[..., 0:100]
        if signal.channels > 1:  # Multichannel is not supported
            with pytest.raises(ValueError):
                assert ((signal.correlate() == signal.correlate(signal)).all())
        else:
            assert ((signal.correlate() == signal.correlate(signal)).all())

    def test_amplitude_envelope(self, signal):
        x = signal.amplitude_envelope()

    def test_instantaneous_frequency(self, signal):
        x = signal.instantaneous_frequency()

    def test_instantaneous_phase(self, signal):
        x = signal.instantaneous_phase()

    def test_detrend(self, signal):
        x = signal.detrend()

    def test_unwrap(self, signal):
        x = signal.unwrap()

    def test_complex_cepstrum(self, signal):
        t, c, d = signal.complex_cepstrum()

    def test_real_cepstrum(self, signal):
        t, c = signal.real_cepstrum()

    def test_power_spectrum(self, signal):
        freq, power = signal.power_spectrum()

    def test_phase_spectrum(self, signal):
        freq, phase = signal.phase_spectrum()

    def test_peak(self, signal):
        value = signal.peak()
        assert len(value) == signal.channels

    def test_peak_level(self, signal):
        value = signal.peak_level()
        assert len(value) == signal.channels

    def test_sound_exposure(self, signal):
        value = signal.sound_exposure()
        assert len(value) == signal.channels

    def test_sound_exposure_level(self, signal):
        value = signal.sound_exposure_level()
        assert len(value) == signal.channels

    def test_octaves(self, signal):

        freq, octaves = signal.octaves()

    def test_levels(self, signal):

        times, levels = signal.levels()

    def test_leq(self, signal):

        #s = Signal(np.random.randn(10000), 22050)

        leq = signal.leq()

        assert (type(leq) is np.ndarray)

    def test_bandpass(self, signal):
        x = signal.bandpass(1000.0, 2000.0)

    def test_bandstop(self, signal):
        x = signal.bandstop(1000.0, 2000.0)

    def test_highpass(self, signal):
        x = signal.highpass(1000.0)

    def test_lowpass(self, signal):
        x = signal.lowpass(1000.0)

    def test_octavepass(self, signal):
        x = signal.octavepass(1000.0, fraction=6)

    def test_bandpass_frequencies(self, signal):
        f = EqualBand(center=[100., 200., 300.], bandwidth=20.)
        f, x = signal.bandpass_frequencies(f)

    def test_bandpass_octaves(self, signal):
        f, x = signal.octaves()

    def test_bandpass_third_octaves(self, signal):
        f, x = signal.third_octaves()

    def test_bandpass_fractional_octaves(self, signal):
        f, x = signal.fractional_octaves()

    def test_weigh(self, signal):
        s = signal.weigh()
        s = signal.weigh('C')
        s = signal.weigh('A', zero_phase=True)

    ## Plot methods with arguments to test.
    #plot_methods = {'plot'                      : None,
                    #'plot_levels'               :   {
                        #'time'                  : [None, 0.125, 1.0],
                        #'method'                : ['average', 'weighting'],
                        #},
                    #'plot_octaves'              : None,
                    #'plot_third_octaves'        : None,
                    #'plot_fractional_octaves'   : {
                        #'fraction'              : [3, 6]
                        #},
                    #'plot_spectrum'             : {
                        #'N'                     : [None, 8000]
                        #},
                    #}

    #@pytest.yield_fixture
    #def plot_function_with_argument(self):
        ## This won't work with pytest. Apparently they do teardown after the yield
        ## statement and therefore don't support multiple yield statements.
        ## Using a closure doesn't help either.
        #for func, arguments in self.plot_methods.items():
            #if arguments is not None:
                #for prod in it.product(*arguments.values()):
                    #yield (func, dict(zip(arguments.keys(), prod)))
            #else:
                #yield (func, None)

    #def test_plot_functions(self, signal, plot_function_with_argument):
        #func, arguments = plot_function_with_argument
        #if arguments is None:
            #getattr(signal, func)()
        #else:
            #getattr(signal, func)(**arguments)


    def test_plot(self, signal):
        signal.plot()
        plt.close("all")

    def test_plot_levels(self, signal):
        signal.plot_levels()
        signal.plot_levels(method='average', time=1.0)
        signal.plot_levels(method='weighting', time=1.0)
        plt.close("all")

    def test_plot_octaves(self, signal):
        signal.plot_octaves()
        plt.close("all")

    def test_plot_third_octaves(self, signal):
        signal.plot_third_octaves()
        plt.close("all")

    def test_plot_fractional_octaves(self, signal):
        signal.plot_fractional_octaves(3)
        signal.plot_fractional_octaves(6)
        signal.plot_fractional_octaves(9)
        plt.close("all")

    def test_plot_power_spectrum(self, signal):
        signal.plot_power_spectrum()
        plt.close("all")

    def test_plot_phase_spectrum(self, signal):
        signal.plot_phase_spectrum()
        plt.close("all")

    def test_plot_spectrogram(self, signal):
        if signal.channels > 1:
            with pytest.raises(ValueError):
                signal.plot_spectrogram()
        else:
            try:
                signal.plot_spectrogram()
            except NotImplementedError:  # easy way to skip mpl 1.3.1 specgram mode issue
                pass
        plt.close("all")

    def spectrogram(self, signal):
        signal.spectrogram()

    def test_pickling(self, signal):
        import pickle

        p = pickle.dumps(signal)
        obj = pickle.loads(p)

        assert ((obj == signal).all())
        assert (obj.fs == signal.fs)
        assert (type(obj) is type(signal))
