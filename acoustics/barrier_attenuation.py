"""
Barrier Attenuation
=======

The `barrier_attenuation` module contains functions for determining attenuation due to a
noise barrier. These calculations assume that the source is a point source, and the wall is infinitely long.

Calculation are based on 'Section 5.2 Barriers' of the 'Applications of Modern Acoustics' by Marshall Long, Moises Levy,
 and Richard Stern (2006). That section references the following as its source:
 Maekawa (1965). Z. Maekawa, “Noise Reduction by Screens.” Memoirs of Faculty of Eng., Kobe, Japan: Kobe Univ., 1965.

"""

from sympy import Point, Segment, Ray, Line
from acoustics.decibel import dbsum
from math import sqrt, pi, log10, tanh

# Failure for debugging
HORIZONTAL_ERR = "h_section.intersect is None"
GRAZING_ERR = "Grazing"
POINT_3D_ERR = "bar_cross_point_3D is None"
VERTICAL_ERR = "v_section.intersect is None"


def _get_fresnel_numbers(bands: list[float], pld: float, c: float) -> list[float]:
    r""" Get Fresnel Numbers at each band using the path-length difference and speed of sound c.

    :param bands: list of frequencies.
    :param pld: path-length difference (ft)
    :param c: speed of sound (ft/sec)
    :return: Fresnel Numbers at each band as a list.

    .. math:: N = \pm \frac{2}{ \lambda }  (pld)
    """
    return [(2 / (c / b)) * pld for b in bands]

def _get_bar_attenuation(N: float) -> float:
    r""" Get the barrier attenuation (dB) from the Fresnel number.

    :param N: Fresnel number
    :return: barrier attenuation (dB)

    .. math:: \Delta L_b = 20log \frac{ \sqrt{2 \pi N} }{tanh\sqrt{2 \pi N} } + K_b
    where
        Delta L_b = barrier attenuation for point source (dB)
        N = Fresnel number
        Kb = Barrier constant (5 dB for wall)

    A practical limit of 20 dB is applied after calculation.
    """
    k_b = 5  # assume k_b (barrier const.) for wall=5, berm=8
    limit = 20  # wall limit = 20
    n_d = sqrt(2 * pi * N)
    try:
        bar_att = (20 * log10(n_d / tanh(n_d))) + k_b
    except ZeroDivisionError:
        return 0
    return min(bar_att, limit)


def get_bar_attenuation_by_band(bands: list[float], pld: float, c: float) -> list[float]:
    """
    Insertion loss by octave band due to the barrier.
    Assumptions: Source is a point source, and the wall is infinitely long.

    :param bands: list of frequencies.
    :param pld: path-length difference (ft)
    :param c: speed of sound (ft/sec)
    :return: Insertion loss at each band as a list.

    """
    if pld <= 0:
        return [0] * len(bands)

    fresnel_nums = _get_fresnel_numbers(bands, pld, c)
    ob_bar_attenuation = [_get_bar_attenuation(N) for N in fresnel_nums]
    return ob_bar_attenuation


def get_weighted_bar_attenuation(levels: list[float], weights: list[float], band_bar_att: list[float]) -> float:
    """ Overall effective barrier attenuation based on weighting. Typically in dBA.

    :param levels: Un-weighted levels by band (can be OB, TOB, anything)
    :param weights: Band-based weighting to be applied.
    :param band_bar_att: Barrier attenuation by band.
    :return: Difference between the weighted sums before and after the barrier attenuation is applied.

    """
    weighted_levels = [x + y for x, y in zip(levels, weights)]
    level_0 = dbsum(weighted_levels)

    attenuated_weighted_levels = [x - y for x, y in zip(weighted_levels, band_bar_att)]
    level_1 = dbsum(attenuated_weighted_levels)

    return level_0 - level_1


class _HorizontalSection:
    """ A section of the geometry from the side. """
    def __init__(self, s: Point, r: Point, b: Segment):
        self.s = Point(s.x, s.y, 0)
        self.r = Point(r.x, r.y, 0)
        self.s_r = Segment(self.s, self.r)
        self.bar_p1 = Point(b.p1.x, b.p1.y, 0)
        self.bar_p2 = Point(b.p2.x, b.p2.y, 0)
        self.bar = Segment(self.bar_p1, self.bar_p2)
        try:
            self.intersect = self.s_r.intersection(self.bar)[0]
        except IndexError:
            self.intersect = None


class _VerticalSection:
    """ A section of the geometry from the top. """
    def __init__(
        self,
        s: Point,
        r: Point,
        bar_cross_point_3d: Point,
        h_sect: _HorizontalSection,
    ):
        self.s = Point(0, s.z, 0)
        self.r = Point(h_sect.s.distance(h_sect.r), r.z, 0)
        self.s_r = Segment(self.s, self.r)
        self.bar_cross_point = Point(
            h_sect.s.distance(h_sect.intersect),
            bar_cross_point_3d.z,
            0,
        )
        self.bar = Ray(self.bar_cross_point, self.bar_cross_point + Point(0, -1, 0))

        try:
            self.intersect = self.s_r.intersection(self.bar)[0]
        except IndexError:
            self.intersect = None


class BarrierSceneGeometry:
    """ Source, receiver, and barrier geometry for determining path-length difference"""
    def __init__(self, s: Point, r: Point, b: Segment):
        self.s = s
        self.r = r
        self.b = b
        self.s_r = Segment(self.s, self.r)

        self._h_section = None
        self._bar_cross_point_3d = None
        self._v_section = None
        self._failure = None

    def get_pld(self) -> float:
        """ Path-length difference

        :return: The path length difference.

        .. math:: pld = A + B - r
        where
            A is the shortest distance from the source to the top of the barrier.
            B is the shortest distance from the receiver to the top of the barrier.
            r is the distance between the source and the receiver.

        """
        self._h_section = None
        self._bar_cross_point_3d = None
        self._v_section = None
        self._failure = None

        # update if we can
        self._h_section = _HorizontalSection(self.s, self.r, self.b)
        if self._h_section.intersect is None:
            self._failure = HORIZONTAL_ERR
            return 0
        elif self._bar_s_r_is_grazing():
            self._failure = GRAZING_ERR
            return 0

        self._bar_cross_point_3d = self._get_barrier_cross_point_3d_attr()
        if self._bar_cross_point_3d is None:
            self._failure = POINT_3D_ERR
            return 0

        self._v_section = _VerticalSection(
            self.s, self.r, self._bar_cross_point_3d, self._h_section
        )
        # I don't think this is possible... vert check happens above.
        if self._v_section.intersect is None:
            self._failure = VERTICAL_ERR
            return 0

        return self.s.distance(self._bar_cross_point_3d) \
            + self.r.distance(self._bar_cross_point_3d) \
            - self.s.distance(self.r)

    def _bar_s_r_is_grazing(self) -> bool:
        """returns True if either the start or end point of the source-receiver line lie on the barrier line, or vice versa"""
        l1 = self._h_section.s_r
        l2 = self._h_section.bar

        return (
            l1.contains(l2.p1)
            or l1.contains(l2.p2)
            or l2.contains(l1.p1)
            or l2.contains(l1.p2)
        )

    def _get_barrier_cross_point_3d_attr(self) -> Point | None:
        """
        Determine the point at the top of the barrier for use in the PLD calculation.
        This will be where A and B are smallest (A = source to top of barrier, B = receiver to top of barrier).
        """
        vert_3Dline_at_intersect = Line(
            self._h_section.intersect,
            self._h_section.intersect + Point(0, 0, 1),
        )
        bar_cross_3Dpoint = vert_3Dline_at_intersect.intersection(self.b)[0]
        s_r_cross_3Dpoint = vert_3Dline_at_intersect.intersection(self.s_r)[0]

        # testing if line of sight is broken vertically
        if s_r_cross_3Dpoint.z > bar_cross_3Dpoint.z:
            return None

        return Point(
            self._h_section.intersect.x,
            self._h_section.intersect.y,
            bar_cross_3Dpoint.z,
        )


__all__ = [
    'get_bar_attenuation_by_band',
    'get_weighted_bar_attenuation',
    'HORIZONTAL_ERR',
    'GRAZING_ERR',
    'POINT_3D_ERR',
    'VERTICAL_ERR',
    'BarrierSceneGeometry',
    ]
