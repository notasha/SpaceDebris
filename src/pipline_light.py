import numpy as np
from numpy import pi
from numpy.linalg import norm

from poliastro.bodies import Earth
from poliastro.core.elements import rv2coe

from astropy import units as u

from debris import *
from satellites import *
from sensors import *
from propagator import *
from propagator_light import *
from detection import *

# Initialization ---------------------------------------------------------------

debris = Debris(datafile='spatial dens 600-1000 km 2-10 cm.csv')
# debris = Debris(datafile='spatial dens 600-650 km 2-10 cm.csv')
# debris = Debris(datafile='larger test data.csv')
# debris = Debris(datafile='very small test data.csv')
sat1 = Satellite(sma = 670*u.km + Earth.R.to(u.km), inc = 98.06*u.deg, raan = 11.2*u.deg)
sat2 = Satellite(sma = 730*u.km + Earth.R.to(u.km), inc = 98.3*u.deg, raan = 11.2*u.deg)

# Timeline breakdown ------------------------------------------------------------

epoch = Time('2022-01-01 00:00:00.000', format='iso', scale='utc')

obj_batches_num = 16
obj_batch_size = int(debris.total_num/obj_batches_num) # num of elements in one butch

simulation_time = 1 * u.d
points_in_batch = 10800
time_batch_size = 10800 # num of elements in one butch
time_batches_num = int(simulation_time.to_value(u.s) / time_batch_size)
full_timeline = np.linspace(0, simulation_time.to_value(u.s), time_batches_num * points_in_batch)

print('total number of time steps =', full_timeline.size)

# Propagation of satellites -----------------------------------------------------

sat1_r, sat1_v = propagate(sat1.initial.x, sat1.initial.y, sat1.initial.z,
                           sat1.initial.vx, sat1.initial.vy, sat1.initial.vz, full_timeline, J2=True)
print('sat1 r =', sat1_r.shape, ' |  sat1 v =', sat1_v.shape)

sat2_r, sat2_v = propagate(sat2.initial.x, sat2.initial.y, sat2.initial.z,
                           sat2.initial.vx, sat2.initial.vy, sat2.initial.vz, full_timeline, J2=True)

# Initializing sensors ----------------------------------------------------------

sensor1 = Sensor(direction_vector=[0,-1,0], fov=6.5*u.deg, max_dist=300*u.km, limiting_magnitude=12,
                 sat_r=sat1_r, sat_v=sat1_v)
print('sensor vector 1', sensor1.direction_vector_eci.shape)

sensor2 = Sensor(direction_vector=[0,-1,0], fov=6.5*u.deg, max_dist=300*u.km, limiting_magnitude=12,
                 sat_r=sat2_r, sat_v=sat2_v)

# Propagation and detection | With loops -----------------------------------------

print('debris N =', debris.total_num)

seen_debris = np.array([])
seen_debris_counter = []
total_seen_by_sensor = []

start1 = time.time()

for i in range(obj_batches_num):
    start, end = i * obj_batch_size, (i+1) * obj_batch_size

    x, y, z = debris.initial.x[start:end], debris.initial.y[start:end], debris.initial.z[start:end]
    vx, vy, vz = debris.initial.vx[start:end], debris.initial.vy[start:end], debris.initial.vz[start:end]
    sma, sizes = debris.all_sma[start:end], debris.sizes[start:end]
    r = np.column_stack([x, y, z])
    v = np.column_stack([vx, vy, vz])

    print('----')
    print('Region: from', sma[0], 'to', sma[-1])
    print('Debris numbers: from', start, 'to', end)
    print('----')

    for t in range(time_batches_num):
        start_time, end_time = t * time_batch_size, (t+1) * time_batch_size
        start_point, end_point = t * points_in_batch, points_in_batch * (t+1)
        print('---- iteration', t, '-----------', 'from', start_time, 'to', end_time, 'sec', '----',
              'from', start_point, 'to', end_point, 'elements')

        tofs = np.linspace(start_time, end_time, points_in_batch)

        x, y, z, vx, vy, vz, sma, sizes = updated_debris_list(x, y, z, vx, vy, vz, sma, sizes, seen_debris)

        # Propagating debris
        debris_inc, debris_raan, debris_nu = get_some_orbital_elements(r.astype(np.float64), v.astype(np.float64))
        debris_r = propagate_with_rotation_matrix(sma, debris_inc, debris_raan, debris_nu, tofs)
        print('>> propagator:', 'debris r =', debris_r.shape)

        # Detecting
        earth_sun_vectors = get_earth_sun_direction(epoch, tofs)
        debris_sat_vectors = np.array([get_debris_sat_vectors(sat1_r[:,:,start_point:end_point], debris_r),
                                       get_debris_sat_vectors(sat2_r[:, :, start_point:end_point], debris_r)])
        # print('debris-sat vectors', debris_sat_vectors.shape)
        debris_sun_vectors = get_debris_sun_direction(debris_r, earth_sun_vectors)

        seen_debris, seen_by_sensor = check_what_is_visible([sensor1, sensor2], sma, sizes, debris_r, debris_sat_vectors,
                                                            debris_sun_vectors, earth_sun_vectors, tofs,
                                                            start_point, end_point)

        print('debris detected', seen_debris.size, ' |  debris left =', x.shape[0]-seen_debris.size)
        seen_debris_counter.append(seen_debris.size)
        total_seen_by_sensor.append(seen_by_sensor)
        # print(total_seen_by_sensor)
        if x.shape[0] == 0:
            print('No debris left to detect')
            break

end1 = time.time()
print('----------------------------')
print('time spent on simulation =', end1 - start1)

# Detection | Without loops -------------------------------------------------------

# print('debris N =', debris.total_num)
#
# start1 = time.time()
#
# # Pre-calculations
#
# earth_sun_vectors = get_earth_sun_direction(epoch, timeline)
# print('earth-sun vectors', earth_sun_vectors.shape)
#
# debris_sat_vectors = np.array([get_debris_sat_vectors(sat1_r, debris_r),
#                                get_debris_sat_vectors(sat2_r, debris_r)])
# print('debris-sat vectors', debris_sat_vectors.shape)
#
# debris_sun_vectors = get_debris_sun_direction(debris_r, earth_sun_vectors)
# print('debris-sun vectors', debris_sun_vectors.shape)
#
# seen_debris, seen_by_sensor = check_what_is_visible([sensor1, sensor2], debris.all_sma, debris.sizes, debris_r,
#                                                     debris_sat_vectors, debris_sun_vectors, earth_sun_vectors,
#                                                     timeline, 0, timeline.size)
#
# print('detection events:', 'sensor 1', seen_by_sensor[0][0], 'sensor 2', seen_by_sensor[0][1])
# print('debris detected', seen_debris.size, ' |  debris left =', x.shape[0]-seen_debris.size)
#
# end1 = time.time()
# print('-----------------------------')
# print('time spent on simulation =', end1 - start1)

# Save stats --------------------------------------------------------------------

# np.savetxt('seen counter.csv', np.array(seen_debris_counter_600_700, dtype=int))
# # print(total_seen_by_sensor)
# np.savetxt('total seen by sensor.csv', total_seen_by_sensor_600_700)
