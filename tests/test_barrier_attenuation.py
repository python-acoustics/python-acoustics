import acoustics.barrier_attenuation as ba
from sympy import Point, Segment
from random import randint


OB = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
WEIGHTS = [-26.2, -16.1, -8.6, -3.2, -0, 1.2, 1, -1.1]


class TestBarrierAttenuation:
    def test_1(self):
        """Test s_r grazes barrier start"""
        s = Point(0, 0, 0)
        r = Point(10, 10, 0)
        b = Segment(Point(5, 5, 0), Point(5, 0, 0))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.GRAZING_ERR)

    def test_3(self):
        """Test s_r grazes barrier end"""
        s = Point(0, 0, 0)
        r = Point(10, 10, 0)
        b = Segment(Point(5, 0, 0), Point(5, 5, 0))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.GRAZING_ERR)

    def test_4(self):
        """Test barrier grazes s"""
        s = Point(5, 5, 0)
        r = Point(5, 0, 0)
        b = Segment(Point(0, 0, 0), Point(10, 10, 0))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.GRAZING_ERR)

    def test_5(self):
        """Test barrier grazes r"""
        s = Point(5, 0, 0)
        r = Point(5, 5, 0)
        b = Segment(Point(0, 0, 0), Point(10, 10, 0))

        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        WEIGHTS = [-26.2, -16.1, -8.6, -3.2, -0, 1.2, 1, -1.1]
        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.GRAZING_ERR)

    def test_6(self):
        """Test miss in just horizontal section"""
        s = Point(5, 0, 5)
        r = Point(20, 10, 5)
        b = Segment(Point(0, 5, 6), Point(10, 5, 6))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        WEIGHTS = [-26.2, -16.1, -8.6, -3.2, -0, 1.2, 1, -1.1]
        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.HORIZONTAL_ERR)

    def test_8(self):
        """Test miss in just vertical section"""
        s = Point(5, 0, 5)
        r = Point(5, 10, 15)
        b = Segment(Point(0, 5, 5), Point(10, 5, 5))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.POINT_3D_ERR)

    def test_9(self):
        """Test miss in vertical & horizontal section"""
        s = Point(5, 0, 5)
        r = Point(20, 10, 15)
        b = Segment(Point(0, 5, 6), Point(10, 5, 6))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.HORIZONTAL_ERR)

    def test_12(self):
        """Test pld == 0 due to barrier being too low."""
        s = Point(5, 0, 0)
        r = Point(5, 10, 10)
        b = Segment(Point(0, 5, 5), Point(10, 5, 5))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure is None)

    def test_15(self):
        """Test parallel s_r and bar give no attenuation"""
        s = Point(5, 0, 0)
        r = Point(5, 10, 10)
        b = Segment(Point(6, 10, 10), Point(6, 0, 0))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [randint(0, 100) for _ in range(8)]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert (pld == 0)
        assert (band_ba == [0] * len(OB))
        assert (bar_att == 0)
        assert (bar_scene_geo._failure == ba.HORIZONTAL_ERR)

    def test_16(self):
        """ Real-world example (Project: Mukilteo, sheet For Pres-grnd) """
        s = Point(0, 0, 9)
        r = Point(0, 68, 20)
        b = Segment(Point(-10, 13, 19), Point(10, 13, 19))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [69, 67, 68, 70, 65, 62, 57, 54]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert abs(pld - 2.5263) <= 1e-4
        assert abs(bar_att - 17.26378) <= 1e-4

    def test_17(self):
        """ Real-world example (Project: Mukilteo, sheet For Pres-6ft) """
        s = Point(0, 0, 22)
        r = Point(0, 184.61, 22)
        b = Segment(Point(-10, 6, 25.25), Point(10, 6, 25.25))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [69, 67, 68, 70, 65, 62, 57, 54]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert abs(pld - 0.8532) <= 1e-4
        assert abs(bar_att - 13.1774) <= 1e-4

    def test_17_meters(self):
        """test_17 but in meters"""
        s = Point(0, 0, 22/3.28)
        r = Point(0, 184.61/3.28, 22/3.28)
        b = Segment(Point(-10/3.28, 6/3.28, 25.25/3.28), Point(10/3.28, 6/3.28, 25.25/3.28))
        bar_scene_geo = ba.BarrierSceneGeometry(s, r, b)
        pld = bar_scene_geo.get_pld()

        levels = [69, 67, 68, 70, 65, 62, 57, 54]
        band_ba = ba.get_bar_attenuation_by_band(OB, pld, 1128/3.28)
        bar_att = ba.get_weighted_bar_attenuation(levels, WEIGHTS, band_ba)
        assert abs(pld - 0.8532/3.28) <= 1e-4
        assert abs(bar_att - 13.1774) <= 1e-4

