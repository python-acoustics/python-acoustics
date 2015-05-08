from acoustics.decibel import *

def test_dbsum():
    assert(abs(dbsum([10.0, 10.0]) - 13.0103) < 1e-5)

def test_dbmean():
    assert(dbmean([10.0, 10.0]) == 10.0)
    
def test_dbadd():
    assert(abs(dbadd(10.0, 10.0) - 13.0103) < 1e-5)

def test_dbsub():
    assert(abs(dbsub(13.0103, 10.0) < 10.0 + 1e-5))
