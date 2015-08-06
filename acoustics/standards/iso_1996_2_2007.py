"""
ISO 1996-2:2007

ISO 1996-2:2007 describes how sound pressure levels can be determined by direct measurement, 
by extrapolation of measurement results by means of calculation, or exclusively by calculation, 
intended as a basis for assessing environmental noise. 

"""
import itertools
import numpy as np
import pandas as pd
from scipy.signal import welch, hanning
from scipy.stats import linregress
import matplotlib.pyplot as plt
from acoustics.decibel import dbsum
from intervaltree import IntervalTree, Interval
from acoustics.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
import weakref
from tabulate import tabulate


TONE_WITHIN_PAUSE_CRITERION_DB = 6.0
"""A tone may exist when the level of any line in the noise pause is 6 dB or more about...."""

TONE_BANDWIDTH_CRITERION_DB = 3.0
"""Bandwidth of the detected peak."""

TONE_LINES_CRITERION_DB = 6.0
"""All lines with levels within 6 dB of the maximum level are classified as tones."""

TONE_SEEK_CRITERION = 1.0
"""Tone seek criterion."""

REGRESSION_RANGE_FACTOR = 0.75
"""Range of regression is usually +/- 0.75 critical bandwidth."""

_WINDOW_CORRECTION = {'hanning' : -1.8,
                      }

def window_correction(window):
    """Correction to be applied to :math:`L_{pt}` due to use of window."""
    try:
        return _WINDOW_CORRECTION[window]
    except KeyError:
        raise ValueError("Window correction is not available for specified window.")


def critical_band(frequency):
    """Bandwidth of critical band of frequency.
    
    :param frequency: Center frequency of tone.
    :returns: (bandwidth, center, lower, upper) of band.
    
    """
    if isinstance(frequency, np.ndarray):
        center[frequency<50.0] = 50.0
    else:
        center = 50.0 if frequency < 50 else frequency

    bandwidth = (center>500.0)*(center*0.20) + (center<=500.0)*100.0
    
    upper = center + bandwidth/2.0
    lower = center - bandwidth/2.0
    
    return center, lower, upper, bandwidth


def tones_level(tone_levels):
    """Total sound pressure level of the tones in a critical band given the level of each of the tones.
    
    .. math L_{pt} = 10 \log_{10}{\sum 10^{L_{pti}/10}}
    
    See equation C.1 in section C.2.3.1.
    """
    return dbsum(tone_levels)


def masking_noise_level(noise_lines, frequency_resolution, effective_analysis_bandwidth):
    """Masking noise level :math:`L_{pn}`
    
    :param noise_lines: Masking noise lines. See :func:`masking_noise_lines`.
    :param frequency_resolution: Frequency resolution :math:`\Delta f`.
    :param effective_analysis_bandwidth: Effective analysis bandwidth :math:`B`.
    
    .. math:: L_{pn} = 10 \log_{10}{\sum 10^{L_n/10}} + 10 \log_{10}{\frac{\Delta f}{B}}
    
    See equation C.11 in section C.4.4.
    
    """
    return dbsum(noise_lines) + 10.0*np.log10(frequency_resolution/effective_analysis_bandwidth)


def masking_noise_lines(levels, line_classifier, center, bandwidth, regression_range_factor):
    """Determine masking noise level lines using regression line. Returns array of :math:`L_n`.
    
    :param levels: Levels as function of frequency.
    :type levels: :class:`pd.Series`.
    :param lines_classifier: Categorical indicating what each line is.
    :param center: Center frequency. 
    :param bandwidth: bandwidth of critical band.
    :param regression_range_factor: Range factor.
    :returns: (Array with masking noise lines, slope, intercept).
    """
    slicer = slice(center-bandwidth*regression_range_factor, center+bandwidth*regression_range_factor)
    levels = levels[slicer]
    frequencies = levels.index
    regression_levels = levels[line_classifier=='noise']
    slope, intercept = linregress(x=regression_levels.index, y=regression_levels)[0:2]
    levels_from_regression = slope * frequencies + intercept
    return levels_from_regression, slope, intercept


def tonal_audibility(tones_level, masking_noise_level, center):
    """Tonal audibility.
    
    :param tones_level: Total sound pressure level of the tones in the critical band :math:`L_{pt}.
    :param masking_noise_level: Total sound pressure level of the masking noise in the critical band :math:`L_{pn}.
    :param center: Center frequency of the critical band :math:`f_c`.
    :returns: Tonal audibility :math:`\Delta L_{ta}`
    
    .. math:: \Delta L_{ta} = L_{pt} - L_{pn} + 2 + \log_{10}{1 + \left(\frac{f_c}{502}\right)^{2.5}}
    
    See equation C.3. in section C.2.4.
    """
    return tones_level - masking_noise_level + 2.0 + np.log10(1.0+(center/502.0)**(2.5))


def tonal_adjustment(tonal_audibility):
    """Adjustment :math:`K`. 
    
    :param tonal_audibility: Tonal audibility :math:`L_{ta}`.
    
    See equations C.4, C.5 and C.6 in section C.2.4.
    """
    if tonal_audibility > 10.0:
        return 6.0
    elif tonal_audibility < 4.0:
        return 0.0
    else:
        return tonal_audibility - 4.0
    

class Tonality(object):
    """Perform assessment of audibility of tones in noise.
    """
    
    def __init__(self, signal, sample_frequency, window='hanning', 
                 reference_pressure=REFERENCE_PRESSURE,
                 tsc=TONE_SEEK_CRITERION,
                 regression_range_factor=REGRESSION_RANGE_FACTOR,
                 bins=None,
                ):
        
        self.signal = signal
        """Samples in time-domain."""
        self.sample_frequency = sample_frequency
        """Sample frequency."""
        self.window = window
        """Window to be used."""
        self.reference_pressure = reference_pressure
        """Reference sound pressure."""
        self.tsc = tsc
        """Tone seeking criterium."""
        self.regression_range_factor = regression_range_factor
        """Regression range factor."""
        self.bins = bins
        """Frequency bins to use."""
        
        self._noise_pauses = list()
        """Private list of noise pauses that were determined or assigned."""
        self._spectrum = None
        """Power spectrum as function of frequency."""


    @property
    def noise_pauses(self):
        """Noise pauses that were determined."""
        yield from self._noise_pauses
    
    
    @property
    def tones(self):
        """Tones that were determined."""
        yield from (noise_pause.tone for noise_pause in self.noise_pauses if noise_pause.tone is not None)
    
    
    @property
    def critical_bands(self):
        """Critical bands that were determined. A critical band is determined for each tone."""
        yield from (tone.critical_band for tone in self.tones)


    @property
    def spectrum(self):
        """Power spectrum of the input signal.
        """
        if self._spectrum is None:
            bins = self.bins
            if bins is None:
                bins = self.sample_frequency
            bins //= 1 # Fix because of bug in welch with uneven bins
            f, p = welch(self.signal, fs=self.sample_frequency, nperseg=bins, window=self.window)
            self._spectrum = pd.Series(10.0*np.log10(p / (2.0e-5)**2.0), index=f)
        return self._spectrum
    
        
    @property
    def frequency_resolution(self):
        """Frequency resolution.
        """
        df = np.diff(np.array(self.spectrum.index)).mean()
        return df
        #return 1.0 / self.sample_frequency
    
    
    @property
    def effective_analysis_bandwidth(self):
        """Effective analysis bandwidth.
        
        In the case of the Hanning window
        
        .. math:: B_{eff} = 1.5 \Delta f
        
        with \Delta f the :attr:`frequency_resolution`.
        
        C.2.2: Note 1.
        """
        if self.window == 'hanning':
            return 1.5 *  self.frequency_resolution
        else:
            raise ValueError()
    
    
    def _set_noise_pauses(self, noise_pauses):
        """Manually set noise pauses. Expects iterable of tuples."""
        self._noise_pauses = [NoisePause(start, end) for start, end in noise_pauses]
        return self


    def determine_noise_pauses(self, end=None):
        """Determine noise pauses. The determined noise pauses are available in :attr:`noise_pause_ranges`.
        
        Noise pauses are search for using :func:`noise_pause_seeker`.
        """
        self._set_noise_pauses(noise_pause_seeker(np.array(self.spectrum[:end]), self.tsc))
        return self
        

    def _determine_tones(self):
        """Analyse the noise pauses for tones. The determined tones are available via :attr:`tones`.
        Per frequency line results are available via :attr:`line_classifier`.
        """
        levels = self.spectrum
        
        # First we need to check for the tones.
        for noise_pause in self.noise_pauses:
            # Mark noise pause start and end.
            self.line_classifier.iloc[noise_pause.start] = 'start'
            self.line_classifier.iloc[noise_pause.end] = 'end'
            # Mark all other lines within noise pause as neither tone nor noise.
            self.line_classifier.iloc[noise_pause.start+1] = 'neither'
            self.line_classifier.iloc[noise_pause.end-1] = 'neither'
            
            # Determine the indices of the tones in a noise pause
            tone_indices = determine_tone_lines(levels, self.frequency_resolution, 
                                                noise_pause.start, noise_pause.end)
            # If we have indices, ...
            if np.any(tone_indices):
                # Then we mark those as tone lines.
                self.line_classifier.iloc[tone_indices] = 'tone'
                # And create a tone object.
                noise_pause.tone = create_tone(levels, tone_indices, weakref.proxy(noise_pause))
        return self
        
        
    def _determine_critical_bands(self):
        """Put a critical band around each of the determined tones."""
        
        for tone in self.tones:
            critical_band = self.critical_band_at(tone.center)
            tone.critical_band = critical_band
            critical_band.tone = weakref.proxy(tone)
        return self
    
    def analyse(self):
        """Analyse the noise pauses for tones and put critical bands around each of these tones. 
        The tones are available via :attr:`tones` and the critical bands via :attr:`critical_bands`.
        Per frequency line results are available via :attr:`line_classifier`.
        """
        
        # Build classifier.
        levels = self.spectrum
        self.line_classifier = pd.Series(['noise']*len(levels), index=levels.index)
        # Determine tones. Puts noise pause starts/ends in classier as well as tone lines
        # and lines that are neither tone nor noise.
        self._determine_tones()
        # Determine critical bands.
        self._determine_critical_bands()
        return self
    
    def critical_band_at(self, frequency):
        """Put at a critical band at `frequency`.
        In order to use this function :attr:`line_classifier` needs to be available, 
        which means :meth:`analyse` needs to be used first.
        """
        return create_critical_band(self.spectrum, self.line_classifier, frequency, 
                                    self.frequency_resolution, 
                                    self.effective_analysis_bandwidth, 
                                    self.regression_range_factor, self.window)


    def plot_spectrum(self):
        """Plot power spectrum."""
        spectrum = self.spectrum
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(spectrum.index, spectrum)
        ax.set_xlabel('f in Hz')
        ax.set_ylabel('L in dB')
        return fig
    
    @property
    def dominant_tone(self):
        """Most dominant_tone tone.
        
        The most dominant_tone tone is the tone with the highest tonal audibility :math:`L_{ta}`.
        """
        if self.critical_bands:
            return sorted(self.tones, key=lambda x: x.critical_band.tonal_audibility, reverse=True)[0]
        elif self.tones:
            raise RuntimeError("Tones were determined but critical bands were not...?")
        elif self.noise_pauses:
            raise ValueError("Need to run analysis first.")
        else:
            raise ValueError("Need to define/search noise pauses first.")
    
    
    def plot_results(self, noise_pauses=False, tones=True, critical_bands=True):
        """Plot overview of results."""

        df = self.frequency_resolution
        #fig = self.plot_spectrum()
        #ax = fig.axes[0]

        levels = self.spectrum
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(levels.index, levels)
        ax.set_xlabel("$f$ in Hz")
        ax.set_ylabel("$L$ in dB")
        
        if noise_pauses:
            for pause in self.noise_pauses:
                ax.axvspan(pause.start*df, pause.end*df, color='green', alpha=0.05)
        
        if tones:
            for tone in self.tones:
                ax.axvline(tone.center)
        
        if critical_bands:
            for band in self.critical_bands:
                ax.axvspan(band.start, band.end, color='yellow', alpha=0.05)

        band = self.dominant_tone.critical_band
            #ax.avxspan(band.start, band.end, color='blue')
        ax.axvline(band.start, color='red', linewidth=0.1)
        ax.axvline(band.end, color='red', linewidth=0.1)

        # Limit xrange
        if noise_pauses: 
            _items = self.noise_pauses
        elif critical_bands: 
            _items = self.critical_bands
        span = IntervalTree(Interval(pause.start, pause.end) for pause in _items).range()
        ax.set_xlim(span.begin, span.end)
        
        return fig
    
    
    def results(self):
        """Print overview of results."""
        cb = self.dominant_tone.critical_band
        
        tones = [("Tone", "{:4.1f} Hz: {:4.1f} dB".format(tone.center, tone.tone_level)) for tone in self.tones]
        
        table = [("Critical band", "{:4.1f} to {:4.1f} Hz".format(cb.start, cb.end)),
                 ("Masking noise level $L_{pn}$", "{:4.1f} dB".format(cb.masking_noise_level)),
                 ("Tonal level $L_{pt}$", "{:4.1f} dB".format(cb.total_tone_level)),
                 ("Dominant tone", "{:4.1f} Hz".format(cb.tone.center)),
                 ("Tonal audibility $L_{ta}$", "{:4.1f} dB".format(cb.tonal_audibility)),
                 ("Adjustment $K_{t}$", "{:4.1f} dB".format(cb.adjustment)),            
            ]
        table += tones
        return tabulate(table)
              
 
class NoisePause(object):
    
    def __init__(self, start, end, tone=None):
        self.start = start
        self.end = end
        self.tone = tone
        
    def __str__(self):
        return "(start={},end={})".format(self.start, self.end)
    
    def __repr__(self):
        return "NoisePause{}".format(str(self))

    def _repr_html_(self):
        table = [("Start", self.start),
                 ("End", self.end)]
        return tabulate(table, tablefmt="html")


def create_tone(levels, tone_lines, noise_pause):
    """Create an instance of Tone."""
    
    center = levels.iloc[tone_lines].argmax()
    tone_level = tones_level(levels.iloc[tone_lines])
    return Tone(center, tone_lines, tone_level, noise_pause)


class Tone(object):
    """Tone."""
    
    def __init__(self, center, tone_lines, tone_level, noise_pause, critical_band=None):
        self.center = center
        self._tone_lines = tone_lines
        self.tone_level = tone_level
        self.noise_pause = noise_pause
        self.critical_band = critical_band

    def __str__(self):
        return "(center={:4.1f}, levels={:4.1f})".format(self.center, self.tone_level)

    def __repr__(self):
        return "Tone{}".format(str(self))

    def _repr_html_(self):
        table = [("Center frequency", "{:4.1f} Hz".format(self.center)),
                 ("Tone level", "{:4.1f} dB".format(self.tone_level))]
        return tabulate(table, tablefmt='html')


def create_critical_band(levels, line_classifier, frequency, frequency_resolution,
                         effective_analysis_bandwidth, regression_range_factor, window, tone=None):
    """Create an instance of CriticalBand."""
    
    center, start, end, bandwidth = critical_band(frequency)
    
    # Masking noise lines
    noise_lines, regression_slope, regression_intercept = masking_noise_lines(levels, line_classifier, center, bandwidth, regression_range_factor)
    # Masking noise level   
    noise_level = masking_noise_level(noise_lines, frequency_resolution, effective_analysis_bandwidth)
    # Total tone level
    tone_lines = levels[line_classifier=='tone'][start:end]
    tone_level = tones_level(tone_lines) - window_correction(window)
    # Tonal audibility
    audibility = tonal_audibility(tone_level, noise_level, center)
    # Adjustment Kt
    adjustment = tonal_adjustment(audibility)
    
    return CriticalBand(center, start, end, bandwidth, 
                        regression_range_factor, regression_slope, regression_intercept,
                        noise_level, tone_level, audibility, adjustment, tone)


class CriticalBand(object):
    
    def __init__(self, center, start, end, bandwidth, 
                 regression_range_factor, regression_slope, regression_intercept, 
                 noise_level, tone_level, audibility,
                 adjustment, tone=None):
                 
        self.center = center
        """Center frequency of the critical band."""
        self.start = start
        """Lower band-edge frequency of the critical band."""
        self.end = end
        """Upper band-edge frequency of the critical band."""
        self.bandwidth = bandwidth
        """Bandwidth of the critical band."""
        self.regression_range_factor = regression_range_factor
        """Range of regression factor. See also :attr:`REGRESSION_RANGE_FACTOR`."""        
        self.regression_slope = regression_slope
        """Linear regression slope."""
        self.regression_intercept = regression_intercept
        """Linear regression intercept."""
        self.masking_noise_level = noise_level
        """Masking noise level :math:`L_{pn}`."""
        self.total_tone_level = tone_level
        """Total tone level :math:`L_{pt}`."""
        self.tonal_audibility = audibility
        """Tonal audibility :math:`L_{ta}`."""
        self.adjustment = adjustment
        """Adjustment :math:`K_{t}`."""
        self.tone = tone

    def __str__(self):
        return "(center={:4.1f}, bandwidth={:4.1f}, tonal_audibility={:4.1f}, adjustment={:4.1f}".format(self.center,
                                                                                                         self.bandwidth,
                                                                                                         self.tonal_audibility,
                                                                                                         self.adjustment)
    

    def __repr__(self):
        return "CriticalBand{}".format(str(self))

    def _repr_html_(self):
        
        table = [
            ("Center frequency", "{:4.1f} Hz".format(self.center)),
            ("Start frequency", "{:4.1f} Hz".format(self.start)),
            ("End frequency", "{:4.1f} Hz".format(self.end)),
            ("Bandwidth", "{:4.1f} Hz".format(self.bandwidth)),
            ("Regression factor", "{:4.1f}".format(self.regression_range_factor)),
            ("Regression slope", "{:4.1f}".format(self.regression_slope)),
            ("Regression intercept", "{:4.1f}".format(self.regression_intercept)),
            ("Masking noise level", "{:4.1f} dB".format(self.masking_noise_level)),
            ("Total tone level", "{:4.1f} dB".format(self.total_tone_level)),
            ("Tonal audibility $L_{ta}$", "{:4.1f} dB".format(self.tonal_audibility)),
            ("Adjustment $K_{t}$", "{:4.1f} dB".format(self.adjustment)),
            ]
        
        return tabulate(table, tablefmt='html')

    
#----------Noise pauses----------------------------  

#def _search_noise_pauses(levels, tsc):
    #"""Search for start and end indices of noise pauses in a single direction.
    
    #:param levels: Spectral levels.    
    #:param tsc: Tone seeking criterium in decibels.
    #:returns: Tuple with start and end indices.
    #"""
    #starts = list()
    #ends = list()
    
    ## Do not search in two outer samples on each side because of possible out of index.
    #for i in range(2, len(levels)-2):
            #if (levels[i] - levels[i-1]) >= tsc and (levels[i-1] - levels[i-2]) < tsc:
                #starts.append(i)
            #if (levels[i] - levels[i+1]) >= tsc and (levels[i+1] - levels[i+2]) < tsc:
                #ends.append(i)
    #return starts, ends        
 
def _search_noise_pauses(levels, tsc):
    #starts = list()
    #ends = list()
    pauses = list()
    possible_start = None
    for i in range(2, len(levels)-2):
            if (levels[i] - levels[i-1]) >= tsc and (levels[i-1] - levels[i-2]) < tsc:
                possible_start = i
                #starts.append(i)
            if (levels[i] - levels[i+1]) >= tsc and (levels[i+1] - levels[i+2]) < tsc:
                if possible_start:
                    #starts.append(possible_start)
                    pauses.append((possible_start, i))
                    possible_start = None
                    #ends.append(i)
    #return starts, ends
    return pauses
    
 
 
def possible_noise_pauses(levels, tsc):
    """Determine start and end indices of possible noise pauses.
    
    :param levels: Spectral levels.    
    :param tsc: Tone seeking criterium in decibels.
    :returns: Tuple with two vectors representing start and end indices.
    
    A forward and backward sweep are performed. 
    Only the start and end indices that were found in both directions are returned.
    
    .. seealso:: :func:`noise_pause_seeker`
    """    
    # Forward and backward sweep
    forward_starts, forward_ends = _search_noise_pauses(levels, tsc)
    backward_ends, backward_starts = _search_noise_pauses(levels[::-1], tsc)
    # Only keep those values that were found in both directions.
    starts = sorted(list(set(forward_starts + backward_starts)))
    ends = sorted(list(set(forward_ends + backward_ends)))
    
    return starts, ends


def smallest_non_overlapping_ranges(intervals):
    """Return the smallest intervals that don't overlap any of the other overlaps.
    Note that combinations where the end point <= start point are filtered out.
    """
    # Throw out stop<=start values
    intervals = ((start, stop) for start, stop in intervals if stop > start)
    intervals = [Interval(*interval) for interval in intervals]
    # Keep filtering as long as we have overlapped intervals.
    while IntervalTree.from_tuples(intervals).find_nested():
        for iv in intervals:
            it = IntervalTree(intervals)
            overlapped = it.search(iv, strict=True)
            # The interval obviously overlaps itself, so let's throw it out of the results.
            overlapped.remove(iv)
            # If we still have overlapping ranges, then this interval 
            # is not an inner interval and should be dropped
            if overlapped:
                intervals.remove(iv) 
    # Return an iterable of list of tuples with each tuple representing an interval.
    return ((interval[0], interval[1]) for interval in intervals)
  

def noise_pause_seeker(levels, tsc):
    """Given the levels of a spectrum and a tone seeking criterium this top level function seeks possible noise pauses.
    
    :param levels: Spectral levels.
    :param df: Frequency resolution.
    :param tsc: Tone seeking criterium.
    
    Possible start and end indices of noise pauses are determined using :func:`possible_noise_pauses.
    Then, only those that correspond to the smallest intervals that do not overlap other intervals are kept.
    """    
    # Determine all possible noise starts and stops
    #starts, ends = possible_noise_pauses(levels, tsc)
    # Keep only the noise pauses that don't overlap any other possible noise pauses.
    #possible_pauses = itertools.product(starts, ends)
    #possible_pauses = intervals = ((start, end) for start, end in possible_pauses if end > start)
    #possible_pauses = smallest_non_overlapping_ranges(possible_pauses)
    #possible_pauses = zip(starts, ends)
    #possible_pauses = _search_noise_pauses(levels, tsc)
    
    #forward_starts, forward_ends = _search_noise_pauses(levels, tsc)
    #backward_ends, backward_starts = _search_noise_pauses(levels[::-1], tsc)
    #forward_pauses = ((start, end) for start, end in itertools.product(forward_starts, forward_ends) if end > start)
    #backward_pauses = ((start, end) for start, end in itertools.product(backward_starts, backward_ends) if end > start)
    n = len(levels)
    forward_pauses = _search_noise_pauses(levels, tsc)
    backward_pauses = _search_noise_pauses(levels[::-1], tsc)
    backward_pauses = [(n-1-start, n-1-end) for end, start in reversed(backward_pauses)]
    possible_pauses = sorted(list( set(forward_pauses) & set(backward_pauses) ) )
    
    yield from possible_pauses


#------------------- Tone seeking--------------------

    
def determine_tone_lines(levels, df, start, end):
    """Determine tone lines in noise pause.
      
    :param levels: Series with levels as function of frequency.
    :param df: Frequency resolution.
    :param start: Index of noise pause start.
    :param end: Index of noise pause end.
    
    :returns: Array with indices of tone lines in noise pause.
    
    """
    # Noise pause range object
    npr = slice(start, end+1)

    tone_indices = np.array([])
    
    # Levels but with integeres as indices instead of frequencies.
    # Benefit over np.array is that the index is maintained when the object is sliced.
    levels_int = levels.reset_index(drop=True)
    
    
    #levels_pause = levels.iloc[npr]

    # If any of the lines is six 6 dB above. See section C.4.3.
    if np.any((levels.iloc[npr] >= TONE_WITHIN_PAUSE_CRITERION_DB + levels.iloc[start-1]) & 
              (levels.iloc[npr] >= TONE_WITHIN_PAUSE_CRITERION_DB + levels.iloc[end+1])):

        # Indices of values that are within -3 dB point.
        indices_3db = np.nonzero(levels.iloc[npr] >= levels.iloc[npr].max() - TONE_BANDWIDTH_CRITERION_DB)[0]
        # -3 dB bandwidth
        bandwidth_for_tone_criterion = (indices_3db.max()-indices_3db.min()) * df
        # Frequency of tone.
        tone_center_frequency = levels.iloc[npr].argmax()
        tone_center_index = levels.reset_index(drop=True).iloc[npr].argmax()
        # Critical band
        _, _, _, critical_band_bandwidth = critical_band(tone_center_frequency)
        
        # Fullfill bandwidth criterion? See section C.4.3
        if bandwidth_for_tone_criterion < 0.10 * critical_band_bandwidth:
            # All values within 6 decibel are designated as tones.
            tone_indices = (levels_int.iloc[npr][ levels_int.iloc[npr] >= levels_int.iloc[npr].max() - TONE_LINES_CRITERION_DB ]).index.get_values()
        #else:
            #raise ValueError("10% bandwidth criterion is not fullfilled.") # Maybe warning instead...

    return tone_indices


#class CriticalBand(object):
    #"""Critical band that is centered around :class:`Tone`."""
    
    #def __init__(self, tone, regression_range_factor):
        #self._tone = tone
        #"""Tone which this critical band is around."""
        
        #center, start, stop, bandwidth = critical_band(tone.center)
        
        #self.center = center
        #"""Center frequency of critical band."""
        #self.start = start
        #"""Lower-edge frequency."""
        #self.stop = stop
        #"""Upper edge frequency."""
        #self.bandwidth = stop - start
        #"""Bandwidth."""
        #self.regression_range_factor = regression_range_factor
        #"""Range factor for regression. By default +/- 0.75"""
    
    #@property
    #def masking_noise_level_regression(self):
        #"""Regression"""
        #levels = self._tone._noise_pause._model.spectrum
        #noise_lines, slope, intercept = masking_noise_lines(self._tone._model.spectrum,
                                                            #self._lines_classifier,
                                                            #self.center, 
                                                            #self.bandwidth, 
                                                            #self.regression_range_factor)
        #return slope, intercept

    #@property
    #def masking_noise_level(self):
        #"""Masking noise level."""
        #model = self._model._tone._model
        #levels = model.spectrum

        #noise_lines, slope, intercept = masking_noise_lines(model.spectrum,
                                                            #self._lines_classifier,
                                                            #self.center, 
                                                            #self.bandwidth, 
                                                            #self.regression_range_factor)
        #return masking_noise_level(noise_lines, 
                                   #model.frequency_resolution, 
                                   #model.effective_analysis_bandwidth)
    
    #@property
    #def total_tone_level(self):
        #"""Total level of the tones in this band."""
        #levels = self._tone._noise_pause._model.spectrum
        ##level_tone_lines = np.ones_like(levels) * -np.inf
        
        ## We create one Series with all values 
        ##for tone in self._tone._model.tones():
        ##    level_tone_lines += levels[tone.marking=='tone']
        
        ## We select all lines that are marked as tones.
        #level_tone_lines = levels[self._tone._model._masking=='tone']
        ## Then we sum over the lines that are within the critical band.
        #total_level = tones_level(level_tone_lines[self.start:self.stop])
        ## And correct for the window
        #total_level -= window_correction(self._tone._noise_pause._model.window)
        #return total_level
    
    #@property
    #def tonal_audibility(self):
        #"""Tonal audibility of the tone in this critical band."""
        #return tonal_audibility(self.total_tone_level, self.masking_noise_level, self.center)

    #@property
    #def adjustment(self):
        #"""Adjustment :math:`K_t`."""
        #return adjustment(self.tonal_audibility)