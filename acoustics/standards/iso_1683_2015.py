"""
ISO 1683:2015
=============
"""
REFERENCE_VALUES = {
    "gas": {
        "pressure": 2e-5,  # Pa
        "exposure": (2e-5)**2,  # Pa^2 s
        "power": 1e-12,  # W
        "energy": 1e-12,  # J
        "intensity": 1e-12,  # W/m^2
    },
    "liquid": {
        "pressure": 1e-5,  # Pa
        "exposure": 1e-5,  # Pa^2 s
        "power": 1e-12,  # W
        "energy": 1e-12,  # J
        "intensity": 1e-12,  # W/m^2
        "particle_displacement": 1e-12,  # m
        "particle_velocity": 1e-9,  # m/s
        "particle_acceleration": 1e-6,  # m/s^2
        "distance": 1,  # m
    },
    "vibration": {
        "displacement": 1e-12,  # m
        "velocity": 1e-9,  # m/s
        "acceleration": 1e-6,  # m/s^2
        "force": 1e-6,  # N}
    }
}
