import pytest
import numpy as np
from acoustics.standards.iso_1996_2_2007 import Tonality
from acoustics import Signal

def test_tonality():

    duration = 60.0
    fs = 10025.0
    samples = int(fs*duration)
    times = np.arange(samples)/fs
    
    signal = Signal(np.sin(2.0*np.pi*1000.0*times), fs)  

    tonality = Tonality(signal, signal.fs)
    
    # Test methods before analysis
    tonality.spectrum
    tonality.plot_spectrum()
    
    tonality.frequency_resolution
    tonality.effective_analysis_bandwidth

    
    assert len(list(tonality.noise_pauses)) == 0
    assert len(list(tonality.tones)) == 0
    assert len(list(tonality.critical_bands)) == 0
    
    # Perform analysis
    tonality.determine_noise_pauses().analyse()
    
    assert len(list(tonality.noise_pauses)) == 1
    assert len(list(tonality.tones)) == 1
    assert len(list(tonality.critical_bands)) == 1
    
    tonality.critical_band_at(900.0)
    
    tonality.dominant_tone
    print(tonality.results())
    tonality.plot_results()


#def test_verify_standard_examples():
    #"""Verify against the examples shown in the standard.
    
    #.. note:: The FTP server is slow. Be patient.
    #"""
    #from ftplib import FTP
    
    ## Obtain data from Delta
    ## https://noiselabdk.wordpress.com/demo-download/
    #hostname = 'ftp.delta.dk'
    #remote_data = 'Demo Files/Tone wave files.zip'
    #username = 'nlpublic'
    #password = 'noiselab'
    #local_data = 'data.zip'

    #ftp = FTP(hostname, username, password)
    #with open(local_data, 'wb') as target_file:
        #ftp.retrbinary('RETR %s' % remote_data, target_file.write)

    #data = [('sam 03 eks 1.wav'),
            #('sam 03 eks 2.wav'),
            #('sam 03 eks 3.wav'),
            #('sam 03 eks 4.wav'),
            ##('sam 03 eks 1.wav'),
        #]