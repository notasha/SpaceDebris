import numpy as np
from numpy import pi
from numpy.linalg import norm

import tracemalloc

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
# from poliastro.twobody.propagation import propagate, cowell
from poliastro.core.elements import coe2rv, rv2coe

from astropy import units as u
from astropy.time import Time, TimeDelta

import quaternion

from debris import *
from satellites import *
from sensors import *
from propagator import *
from detection import *

# Initialization ---------------------------------------------------------------

# tracemalloc.start()

debris = Debris(datafile='spatial dens 600-1000 km 2-10 cm.csv')
# debris = Debris(datafile='larger test data.csv')
# debris = Debris(datafile='very small test data.csv')
sat1 = Satellite(sma = 730*u.km + Earth.R.to(u.km), inc = 98.3*u.deg, raan = 11.2*u.deg)

# Checking the sensor vector ---------------------------------------------------

# # With quaternions
# vec = np.array([0, -1, 0])
# sens = sat1.orb2eci(vec)
# print(sens/norm(sens))
#
# # With angular momentum
# sat_vec = np.array([sat1.initial.x.value, sat1.initial.y.value, sat1.initial.z.value,
#                     sat1.initial.vx.value, sat1.initial.vy.value, sat1.initial.vz.value])
#
# angular_momentum = np.cross(sat_vec[:3], sat_vec[3:])
# sens_vec = angular_momentum / norm(angular_momentum)
# print(sens_vec)
# print(sat1.eci2orb(sens))

# Timeline breakdown ------------------------------------------------------------

epoch = Time('2022-01-01 00:00:00.000', format='iso', scale='utc')

simulation_time = 1 * u.d
points_in_batch = 500
batch_size = 500 # seconds
batches_num = int(simulation_time.to_value(u.s) / batch_size)
full_timeline = np.linspace(0, simulation_time.to_value(u.s), batches_num * points_in_batch)
print('total number of time steps =', full_timeline.size)

# Propagation of satellites -----------------------------------------------------

sat1_r, sat1_v = propagate(sat1.initial.x, sat1.initial.y, sat1.initial.z, sat1.initial.vx, sat1.initial.vy, sat1.initial.vz,
                           full_timeline, J2=True)
# print('sat1 r =', sat1_r.shape, ' |  sat1 v =', sat1_v.shape)

# Initializing sensors ----------------------------------------------------------

sensor1 = Sensor(direction_vector=[0,-1,0], fov=360*u.deg, max_dist=300*u.km, limiting_magnitude=12,
                 sat_r=sat1_r, sat_v=sat1_v)
# print('sensor vector', sensor1.direction_vector_eci.shape)

# Propagation and detection -----------------------------------------------------

print('debris N =', debris.total_num)
seen_debris = np.array([])

start1 = time.time()

x, y, z = debris.initial.x, debris.initial.y, debris.initial.z
vx, vy, vz = debris.initial.vx, debris.initial.vy, debris.initial.vz,
sma, sizes = debris.all_sma, debris.sizes

for t in range(batches_num):
    start_time, end_time = t * batch_size, (t + 1) * batch_size
    start_point, end_point = t* points_in_batch, points_in_batch *(t+1)
    print('---- iteration', t, '-----------', 'from', start_time, 'to', end_time, 'sec', '----',
          'from', start_point, 'to', end_point, 'elements')

    tofs = np.linspace(start_time, end_time, points_in_batch)

    x, y, z, vx, vy, vz, sma, sizes = updated_debris_list(x, y, z, vx, vy, vz, sma, sizes, seen_debris)

    # Propagating debris
    debris_r, debris_v = propagate(x, y, z, vx, vy, vz, tofs, J2=True)
    print('>> propagator:', 'debris r =', debris_r.shape, ' |  debris v =', debris_v.shape)

    # Detecting
    earth_sun_vectors = get_earth_sun_direction(epoch, tofs)
    debris_sat_vectors = get_debris_sat_vectors(sat1_r[:,:,start_point:end_point], debris_r)
    debris_sun_vectors = get_debris_sun_direction(debris_r, earth_sun_vectors)
    seen_debris = check_what_is_visible([sensor1], sma, sizes, debris_r, debris_sat_vectors, debris_sun_vectors,
                                        earth_sun_vectors, tofs, start_point, end_point)
    print('debris detected', seen_debris.size, ' |  debris left =', x.shape[0]-seen_debris.size)

    if x.shape[0] == 0:
        print('No debris left to detect')
        break

end1 = time.time()
print('-----------------------------')
print('time spent on simulation =', end1 - start1)