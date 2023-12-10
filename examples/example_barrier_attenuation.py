"""
An example of how to use :class:`acoustics.barrier_attenuation.BarrierSceneGeometry`.
"""

import acoustics.barrier_attenuation as ba
from sympy import Point, Segment

def main():
    """ Set up the location of the source, receiver, and barrier with xyz coordinates. """
    source = Point(0, 0, 22)
    receiver = Point(0, 184.61, 22)
    barrier = Segment(Point(-10, 6, 25.25), Point(10, 6, 25.25))

    """ Initialize the scene geometry class. """
    bar_scene_geo = ba.BarrierSceneGeometry(source, receiver, barrier)

    """ Get the path-length difference. """
    pld = bar_scene_geo.get_pld()
    print(pld)

    """ provide the octave bands, weights, and noise levels to determine the barrier attenuation by band. """
    ob = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    weights = [-26.2, -16.1, -8.6, -3.2, -0, 1.2, 1, -1.1]
    levels = [69, 67, 68, 70, 65, 62, 57, 54]
    bar_att_by_band = ba.get_bar_attenuation_by_band(ob, pld, 1128)
    print(bar_att_by_band)

    """ Determine overall weighted (A weighted in the case) barrier attenuation. """
    bar_att_overall = ba.get_weighted_bar_attenuation(levels, weights, bar_att_by_band)
    print(bar_att_overall)

if __name__ == '__main__':
    main()
