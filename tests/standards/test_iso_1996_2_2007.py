import pytest
import numpy as np
from acoustics.standards.iso_1996_2_2007 import Tonality
from acoustics import Signal
import collections


def test_tonality():

    duration = 60.0
    fs = 10025.0
    samples = int(fs * duration)
    times = np.arange(samples) / fs

    signal = Signal(np.sin(2.0 * np.pi * 1000.0 * times), fs)

    tonality = Tonality(signal, signal.fs)

    # Test methods before analysis
    tonality.spectrum
    tonality.plot_spectrum()

    tonality.frequency_resolution
    tonality.effective_analysis_bandwidth

    # No values yet, cannot print overview.
    with pytest.raises(ValueError):
        print(tonality.overview())
    tonality.results_as_dataframe()

    assert len(list(tonality.noise_pauses)) == 0
    assert len(list(tonality.tones)) == 0
    assert len(list(tonality.critical_bands)) == 0

    # Perform analysis
    tonality.determine_noise_pauses().analyse()

    # Needs to be checked
    #assert len(list(tonality.noise_pauses)) == 1
    #assert len(list(tonality.tones)) == 1
    #assert len(list(tonality.critical_bands)) == 1

    tonality.critical_band_at(900.0)

    tonality.dominant_tone
    print(tonality.overview())
    tonality.results_as_dataframe()
    tonality.plot_results()



#Target = collections.namedtuple('Target', ['tonal_level',
                                           #'masking_noise_level',
                                           #'tonal_audibility',
                                           #'adjustment'])
#TARGET_DATA = {
    #'example1': Target(46.7, 37.3, 13.7, 6.0),
    #'example2': Target(54.1, 45.2, 11.1, 6.0),
    #'example3': Target(54.6, 45.5, 10.6, 6.0),
    ##'example4': Target(53.6, 45.5, 10.7, 6.0), # Time-varying test.
    #}

#@pytest.fixture(params=TARGET_DATA.keys())
#def example(request):
    #return request.param

#def test_verify_standard_examples(example):
    #"""Verify against the examples shown in the standard.


    #.. note:: The FTP server is slow. Be patient.

    #For the examples in the standard:
    #- Number of spectra: 350
    #- Signal duration: 2 minutes
    #- Window: Hanning
    #- Averaging: Linear
    #- Effective analysis bandwidth: 4.39 Hz

    #The sample frequency is 44.1 kHz. Dividing this sample frequency by
    #results in 360 spectra and an effective analysis bandwidth of
    #4.5 Hz, which is very close to the example.

    #"""
    #from ftplib import FTP
    #import tempfile
    #import os

    #FOLDER = 'THP wave files/'
    #FILES = {'calibration' : 'cal. 93,8 dB.wav',
             #'example1': 'sam 03 eks 1.wav',
             #'example2': 'sam 03 eks 2.wav',
             #'example3': 'sam 03 eks 3.wav',
             #'example4': 'sam 03 eks 4.wav',
             #}
    #CHANNEL = 0

    #fs = 44100.0
    #duration = 120.0
    #samples = duration*fs
    #spectra = 350
    #bins = samples // spectra

    ## Obtain data from Delta
    ## https://noiselabdk.wordpress.com/demo-download/
    ##hostname = 'ftp.delta.dk'
    ##remote_data = 'Demo Files/Tone wave files.zip'
    ##username = 'nlpublic'
    ##password = 'noiselab'
    ##local_data = 'data.zip'

    ##with tempfile.TemporaryDirectory() as directory:
        ##ftp = FTP(hostname, username, password)
        ##with open(local_data, 'wb') as target_file:
            ##ftp.retrbinary('RETR %s' % remote_data, target_file.write)



    #calibration_at = 93.8 # decibel
    #signal = Signal.from_wav(os.path.join(FOLDER, FILES['calibration']))[CHANNEL]
    #calibration_factor = 10.0**((calibration_at - signal.leq())/20.0)

    #target = TARGET_DATA[example]
    #filename = os.path.join(FOLDER, FILES[example])

    #signal = Signal.from_wav(filename)[CHANNEL].pick(0.0, 60.0).weigh('A') * calibration_factor
    #tonality = Tonality(signal, signal.fs, bins=bins)
    #tonality.determine_noise_pauses().analyse()

    #cb = tonality.dominant_tone.critical_band
    #print(tonality.results())
    #assert abs(tonality.dominant_tone.tone_level - target.tonal_level) < 1.0
    #assert abs(cb.masking_noise_level - target.masking_noise_level) < 1.0
    #assert abs(cb.tonal_audibility - target.tonal_audibility) < 1.0
    #assert abs(cb.adjustment - target.adjustment) < 1.0
