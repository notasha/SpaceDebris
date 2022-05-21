import numpy as np
from numpy import pi
from numpy.linalg import norm

import tracemalloc

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate, cowell
from poliastro.core.elements import coe2rv, rv2coe

from astropy import units as u
from astropy.time import Time, TimeDelta

import quaternion

from debris import *
from satellites import *
from sensors import *

# Initialization ---------------------------------------------------------------

epoch = Time('2022-01-01 00:00:00.000', format='iso', scale='utc')
times = np.linspace(1 * u.h, 10 * u.h, num=1)

tracemalloc.start()

# debris = Debris(datafile='spatial dens 600-1000 km 2-10 cm.csv', timeline=times, epoch=epoch)
# debris = Debris(datafile='very small test data.csv', timeline=times, epoch=epoch)
debris = Debris(datafile='larger test data.csv', timeline=times, epoch=epoch)

sat1 = Satellite(sma = 700*u.km + Earth.R.to(u.km), inc = 98.18*u.deg, raan = 11.2*u.deg, timeline=times, epoch=epoch)

debris_sat_vectors = debris.to_orbital_frame(sat1)
debris_sat_dist = debris.get_abs(debris_sat_vectors)

magnitudes = debris.get_magnitudes(debris_sat_vectors, debris_sat_dist, epoch, times)
np.savetxt("magnitudes.csv", magnitudes)

print(tracemalloc.get_traced_memory())
tracemalloc.stop()

# sat1 = Satellite(sma = 700*u.km + Earth.R.to(u.km), inc = 98.18*u.deg, raan = 11.2*u.deg, timeline=times, epoch=epoch)
# sat2 = Satellite(sma = 1030*u.km + Earth.R.to(u.km), inc = 99.5*u.deg, raan = 11.2*u.deg, timeline=times, epoch=epoch)
#
# sensor1 = Sensor(direction_vector=[0, 0, -1], fov=6.5*u.deg, max_dist=300*u.km, limiting_magnitude=12)
# sensor2 = Sensor(direction_vector=[0, 0, -1], fov=6.5*u.deg, max_dist=300*u.km, limiting_magnitude=12)

# Visibility conditions verification -------------------------------------------

# debris_left1, seen_num1, seen_percent1 = sensor1.check_what_is_visible(debris, sat1, epoch, times)
# print(seen_num1, seen_percent1)
#
# debris_left2, seen_num2, seen_percent2 = sensor2.check_what_is_visible(debris, sat2, epoch, times,
#                                                                        first_iteration=False, debris_left=debris_left1)
# print(seen_num2, seen_percent2)

