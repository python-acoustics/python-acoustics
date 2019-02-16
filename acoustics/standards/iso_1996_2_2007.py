"""
ISO 1996-2:2007

ISO 1996-2:2007 describes how sound pressure levels can be determined by direct measurement,
by extrapolation of measurement results by means of calculation, or exclusively by calculation,
intended as a basis for assessing environmental noise.

"""
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import linregress
import matplotlib.pyplot as plt
from acoustics.decibel import dbsum
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

_WINDOW_CORRECTION = {
    'hanning': -1.8,
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
        center = frequency.copy()
        center[frequency < 50.0] = 50.0
    else:
        center = 50.0 if frequency < 50 else frequency

    bandwidth = (center > 500.0) * (center * 0.20) + (center <= 500.0) * 100.0

    upper = center + bandwidth / 2.0
    lower = center - bandwidth / 2.0

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
    return dbsum(noise_lines) + 10.0 * np.log10(frequency_resolution / effective_analysis_bandwidth)


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
    slicer = slice(center - bandwidth * regression_range_factor, center + bandwidth * regression_range_factor)
    levels = levels[slicer]
    frequencies = levels.index
    regression_levels = levels[line_classifier == 'noise']
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
    return tones_level - masking_noise_level + 2.0 + np.log10(1.0 + (center / 502.0)**(2.5))


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


class Tonality:
    """Perform assessment of audibility of tones in noise.

    Objective method for assessing the audibility of tones in noise.
    """

    def __init__(  # pylint: disable=too-many-instance-attributes
            self,
            signal,
            sample_frequency,
            window='hanning',
            reference_pressure=REFERENCE_PRESSURE,
            tsc=TONE_SEEK_CRITERION,
            regression_range_factor=REGRESSION_RANGE_FACTOR,
            nbins=None,
            force_tone_without_pause=False,
            force_bandwidth_criterion=False,
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
        self.nbins = nbins
        """Amount of frequency nbins to use. See attribute `nperseg` of :func:`scipy.signal.welch`."""

        self._noise_pauses = list()
        """Private list of noise pauses that were determined or assigned."""
        self._spectrum = None
        """Power spectrum as function of frequency."""

        self.force_tone_without_pause = force_tone_without_pause
        self.force_bandwidth_criterion = force_bandwidth_criterion

    @property
    def noise_pauses(self):
        """Noise pauses that were determined."""
        for noise_pause in self._noise_pauses:
            yield noise_pause

    @property
    def tones(self):
        """Tones that were determined."""
        for noise_pause in self.noise_pauses:
            if noise_pause.tone is not None:
                yield noise_pause.tone

    @property
    def critical_bands(self):
        """Critical bands that were determined. A critical band is determined for each tone."""
        for tone in self.tones:
            yield tone.critical_band

    @property
    def spectrum(self):
        """Power spectrum of the input signal.
        """
        if self._spectrum is None:
            nbins = self.nbins
            if nbins is None:
                nbins = self.sample_frequency
            nbins //= 1  # Fix because of bug in welch with uneven nbins
            f, p = welch(self.signal, fs=self.sample_frequency, nperseg=nbins, window=self.window, detrend=False,
                         scaling='spectrum')
            self._spectrum = pd.Series(10.0 * np.log10(p / self.reference_pressure**2.0), index=f)
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
            return 1.5 * self.frequency_resolution
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

    def _construct_line_classifier(self):
        """Set values of line classifier."""

        # Build classifier.
        levels = self.spectrum

        categories = ['noise', 'start', 'end', 'neither', 'tone']
        self.line_classifier = pd.Series(
            pd.Categorical(['noise'] * len(levels), categories=categories), index=levels.index)

        # Add noise pauses
        for noise_pause in self.noise_pauses:
            # Mark noise pause start and end.
            self.line_classifier.iloc[noise_pause.start] = 'start'
            self.line_classifier.iloc[noise_pause.end] = 'end'
            # Mark all other lines within noise pause as neither tone nor noise.
            self.line_classifier.iloc[noise_pause.start + 1:noise_pause.end] = 'neither'  # Half-open interval

        # Add tone lines
        for tone in self.tones:
            self.line_classifier.iloc[tone._tone_lines] = 'tone'

        return self

    def _determine_tones(self):
        """Analyse the noise pauses for tones. The determined tones are available via :attr:`tones`.
        Per frequency line results are available via :attr:`line_classifier`.
        """
        levels = self.spectrum

        # First we need to check for the tones.
        for noise_pause in self.noise_pauses:
            # Determine the indices of the tones in a noise pause
            tone_indices, bandwidth_for_tone_criterion = determine_tone_lines(
                levels,
                self.frequency_resolution,
                noise_pause.start,
                noise_pause.end,
                self.force_tone_without_pause,
                self.force_bandwidth_criterion,
            )
            # If we have indices, ...
            if np.any(tone_indices):
                # ...then we create a tone object.
                noise_pause.tone = create_tone(levels, tone_indices, bandwidth_for_tone_criterion,
                                               weakref.proxy(noise_pause))
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

        # Determine tones. Puts noise pause starts/ends in classier as well as tone lines
        # and lines that are neither tone nor noise.
        self._determine_tones()
        # Construct line classifier
        self._construct_line_classifier()
        # Determine critical bands.
        self._determine_critical_bands()
        return self

    def critical_band_at(self, frequency):
        """Put at a critical band at `frequency`.
        In order to use this function :attr:`line_classifier` needs to be available,
        which means :meth:`analyse` needs to be used first.
        """
        return create_critical_band(self.spectrum, self.line_classifier, frequency, self.frequency_resolution,
                                    self.effective_analysis_bandwidth, self.regression_range_factor, self.window)

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
        try:
            return sorted(self.tones, key=lambda x: x.critical_band.tonal_audibility, reverse=True)[0]
        except IndexError:
            return None

    def plot_results(self, noise_pauses=False, tones=True, critical_bands=True):
        """Plot overview of results."""

        df = self.frequency_resolution
        levels = self.spectrum

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(levels.index, levels)
        ax.set_xlabel("$f$ in Hz")
        ax.set_ylabel("$L$ in dB")

        if noise_pauses:
            for pause in self.noise_pauses:
                ax.axvspan(pause.start * df, pause.end * df, color='green', alpha=0.05)

        if tones:
            for tone in self.tones:
                ax.axvline(tone.center, color='black', alpha=0.05)

        if critical_bands:
            for band in self.critical_bands:
                ax.axvspan(band.start, band.end, color='yellow', alpha=0.05)

        band = self.dominant_tone.critical_band
        ax.axvline(band.start, color='red', linewidth=0.1)
        ax.axvline(band.end, color='red', linewidth=0.1)

        # Limit xrange
        if noise_pauses:
            _items = list(self.noise_pauses)
        elif critical_bands:
            _items = list(self.critical_bands)
        ax.set_xlim(min(item.start for item in _items), max(item.end for item in _items))
        return fig

    def overview(self):
        """Print overview of results."""
        try:
            cb = self.dominant_tone.critical_band
        except AttributeError:
            raise ValueError("Cannot show overview (yet). No tones have been determined.")

        tones = [("Tone", "{:4.1f} Hz: {:4.1f} dB".format(tone.center, tone.tone_level)) for tone in self.tones]

        table = [
            ("Critical band", "{:4.1f} to {:4.1f} Hz".format(cb.start, cb.end)),
            ("Masking noise level $L_{pn}$", "{:4.1f} dB".format(cb.masking_noise_level)),
            ("Tonal level $L_{pt}$", "{:4.1f} dB".format(cb.total_tone_level)),
            ("Dominant tone", "{:4.1f} Hz".format(cb.tone.center)),
            ("3 dB bandwidth of tone", "{:2.1f}% of {:4.1f}".format(cb.tone.bandwidth_3db / cb.bandwidth * 100.0,
                                                                    cb.bandwidth)),
            ("Tonal audibility $L_{ta}$", "{:4.1f} dB".format(cb.tonal_audibility)),
            ("Adjustment $K_{t}$", "{:4.1f} dB".format(cb.adjustment)),
            ("Frequency resolution", "{:4.1f} Hz".format(self.frequency_resolution)),
            ("Effective analysis bandwidth", "{:4.1f} Hz".format(self.effective_analysis_bandwidth)),
        ]
        table += tones
        return tabulate(table)

    def results_as_dataframe(self):
        """Return results in dataframe."""
        data = ((tone.center, tone.tone_level, tone.bandwidth_3db, tone.critical_band.start, tone.critical_band.end,
                 tone.critical_band.bandwidth, tone.critical_band.regression_slope,
                 tone.critical_band.regression_intercept, tone.critical_band.masking_noise_level,
                 tone.critical_band.total_tone_level, tone.critical_band.tonal_audibility,
                 tone.critical_band.adjustment) for tone in self.tones)
        columns = [
            'center', 'tone_level', 'bandwidth_3db', 'critical_band_start', 'critical_band_end',
            'critical_band_bandwidth', 'regression_slope', 'regression_intercept', 'masking_noise_level',
            'total_tone_level', 'tonal_audibility', 'adjustment'
        ]
        return pd.DataFrame(list(data), columns=columns)


class NoisePause:
    def __init__(self, start, end, tone=None):
        self.start = start
        self.end = end
        self.tone = tone

    def __str__(self):
        return "(start={},end={})".format(self.start, self.end)

    def __repr__(self):
        return "NoisePause{}".format(str(self))

    def __iter__(self):
        yield self.start
        yield self.stop

    def _repr_html_(self):
        table = [("Start", self.start), ("End", self.end)]
        return tabulate(table, tablefmt="html")


def create_tone(levels, tone_lines, bandwidth_for_tone_criterion, noise_pause):
    """Create an instance of Tone."""

    center = levels.iloc[tone_lines].idxmax()
    tone_level = tones_level(levels.iloc[tone_lines])
    return Tone(center, tone_lines, tone_level, noise_pause, bandwidth_for_tone_criterion)


class Tone:
    """Tone."""

    def __init__(self, center, tone_lines, tone_level, noise_pause, bandwidth_3db, critical_band=None):
        self.center = center
        self._tone_lines = tone_lines
        self.tone_level = tone_level
        self.noise_pause = noise_pause
        self.bandwidth_3db = bandwidth_3db
        self.critical_band = critical_band

    def __str__(self):
        return "(center={:4.1f}, levels={:4.1f})".format(self.center, self.tone_level)

    def __repr__(self):
        return "Tone{}".format(str(self))

    def _repr_html_(self):
        table = [("Center frequency", "{:4.1f} Hz".format(self.center)),
                 ("Tone level", "{:4.1f} dB".format(self.tone_level))]
        return tabulate(table, tablefmt='html')


def create_critical_band(
        levels,
        line_classifier,
        frequency,
        frequency_resolution,
        effective_analysis_bandwidth,
        regression_range_factor,
        window,
        tone=None,
):
    """Create an instance of CriticalBand."""

    center, start, end, bandwidth = critical_band(frequency)

    # Masking noise lines
    noise_lines, regression_slope, regression_intercept = masking_noise_lines(levels, line_classifier, center,
                                                                              bandwidth, regression_range_factor)
    # Masking noise level
    noise_level = masking_noise_level(noise_lines, frequency_resolution, effective_analysis_bandwidth)
    # Total tone level
    tone_lines = levels[line_classifier == 'tone'][start:end]
    tone_level = tones_level(tone_lines) - window_correction(window)
    # Tonal audibility
    audibility = tonal_audibility(tone_level, noise_level, center)
    # Adjustment Kt
    adjustment = tonal_adjustment(audibility)

    return CriticalBand(
        center,
        start,
        end,
        bandwidth,
        regression_range_factor,
        regression_slope,
        regression_intercept,
        noise_level,
        tone_level,
        audibility,
        adjustment,
        tone,
    )


class CriticalBand:
    def __init__(  # pylint: disable=too-many-instance-attributes
            self,
            center,
            start,
            end,
            bandwidth,
            regression_range_factor,
            regression_slope,
            regression_intercept,
            noise_level,
            tone_level,
            audibility,
            adjustment,
            tone=None,
    ):

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
        return "(center={:4.1f}, bandwidth={:4.1f}, tonal_audibility={:4.1f}, adjustment={:4.1f}".format(
            self.center, self.bandwidth, self.tonal_audibility, self.adjustment)

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


def _search_noise_pauses(levels, tsc):
    pauses = list()
    possible_start = None
    for i in range(2, len(levels) - 2):
        if (levels[i] - levels[i - 1]) >= tsc and (levels[i - 1] - levels[i - 2]) < tsc:
            possible_start = i
        if (levels[i] - levels[i + 1]) >= tsc and (levels[i + 1] - levels[i + 2]) < tsc:
            if possible_start:
                pauses.append((possible_start, i))
                possible_start = None
    return pauses


def noise_pause_seeker(levels, tsc):
    """Given the levels of a spectrum and a tone seeking criterium this top level function seeks possible noise pauses.

    :param levels: Spectral levels.
    :param df: Frequency resolution.
    :param tsc: Tone seeking criterium.

    Possible start and end indices of noise pauses are determined using :func:`possible_noise_pauses.
    Then, only those that correspond to the smallest intervals that do not overlap other intervals are kept.
    """
    n = len(levels)
    forward_pauses = _search_noise_pauses(levels, tsc)
    backward_pauses = _search_noise_pauses(levels[::-1], tsc)
    backward_pauses = [(n - 1 - start, n - 1 - end) for end, start in reversed(backward_pauses)]
    possible_pauses = sorted(list(set(forward_pauses) & set(backward_pauses)))
    return possible_pauses


#------------------- Tone seeking--------------------


def determine_tone_lines(levels, df, start, end, force_tone_without_pause=False, force_bandwidth_criterion=False):
    """Determine tone lines in noise pause.

    :param levels: Series with levels as function of frequency.
    :param df: Frequency resolution.
    :param start: Index of noise pause start.
    :param end: Index of noise pause end.

    :returns: Array with indices of tone lines in noise pause.

    """
    # Noise pause range object
    npr = slice(start, end + 1)

    # Return values
    tone_indices = np.array([])
    bandwidth_for_tone_criterion = None

    # Levels but with integeres as indices instead of frequencies.
    # Benefit over np.array is that the index is maintained when the object is sliced.
    levels_int = levels.reset_index(drop=True)

    # If any of the lines is six 6 dB above. See section C.4.3.
    if np.any((levels.iloc[npr] >= TONE_WITHIN_PAUSE_CRITERION_DB + levels.iloc[start - 1]) &
              (levels.iloc[npr] >= TONE_WITHIN_PAUSE_CRITERION_DB + levels.iloc[end + 1])) or force_tone_without_pause:

        # Indices of values that are within -3 dB point.
        indices_3db = np.nonzero(levels.iloc[npr] >= levels.iloc[npr].max() - TONE_BANDWIDTH_CRITERION_DB)[0]
        # -3 dB bandwidth
        bandwidth_for_tone_criterion = (indices_3db.max() - indices_3db.min()) * df
        # Frequency of tone.
        tone_center_frequency = levels.iloc[npr].idxmax()
        #tone_center_index = levels.reset_index(drop=True).iloc[npr].idxmax()
        # Critical band
        _, _, _, critical_band_bandwidth = critical_band(tone_center_frequency)

        # Fullfill bandwidth criterion? See section C.4.3
        if (bandwidth_for_tone_criterion < 0.10 * critical_band_bandwidth) or force_bandwidth_criterion:
            # All values within 6 decibel are designated as tones.
            tone_indices = (levels_int.iloc[npr][
                levels_int.iloc[npr] >= levels_int.iloc[npr].max() - TONE_LINES_CRITERION_DB]).index.get_values()

    return tone_indices, bandwidth_for_tone_criterion
