import numpy as np
from astropy.coordinates import get_sun

from satellites import *
from sensors import *

Earth_radius = Earth.R.to_value(u.km)

def check_what_is_visible(sensors, debris_sma, debris_sizes, unseen_debris_vectors, debris_sat_vectors,
                          debris_sun_vectors, earth_sun_vectors, tofs, start_point, end_point):

    conditions = np.zeros([len(sensors), debris_sma.size, unseen_debris_vectors.shape[-1]])
    seen = []
    total_by_sensor = []
    # seen_counter1, seen_counter2 = 0, 0
    seen_count = {s: 0 for s in range(len(sensors))}

    for s in range(len(sensors)):
        debris_sat_dist = get_debris_sat_distance(debris_sat_vectors[s,:,:])
        # print('>> detection:', 'deb-sat dist', debris_sat_dist.shape)
        distances = check_distance(sensors[s].max_dist, debris_sat_dist)

        FOV = check_if_in_fov(sensors[s].fov, sensors[s].direction_vector_eci[:,start_point:end_point],
                              unseen_debris_vectors)

        LOS = check_sun_debris_los(debris_sma, debris_sun_vectors)

        debris_magnitudes = get_magnitudes(debris_sizes, earth_sun_vectors, debris_sat_vectors[s,:,:],
                                           debris_sat_dist, tofs)
        magnitudes = check_magnitudes(sensors[s].limiting_magnitude, debris_magnitudes)

        # print('>> detection:', 'dist', distances.shape, ' |  fov', FOV.shape, ' |  los', LOS.shape, ' |  m', magnitudes.shape)
        conditions[s,:,:] = np.sum(np.array([distances, FOV, LOS, magnitudes]), axis=0, dtype=np.int16)

        print('>> detection:', 'sensor', s, 'dist', 1 in distances, ' |  fov', 1 in FOV,
              ' |  los', 1 in LOS, ' |  m', 1 in magnitudes)

        for n in range(debris_sma.size):
            for t in range(tofs.size):
                if conditions[s,n,t] == 4:
                    seen.append(n)
                    seen_count[s] += 1

    total_by_sensor.append(seen_count)
    # print('>> detection:', total_by_sensor)
    seen_record = np.unique(seen)
    # print('seen', seen)
    # print('seen_record', seen_record)
    # print('conditions', conditions.shape)
    # np.savetxt('conditions.csv', conditions[0,:,:])

    return seen_record, total_by_sensor

def updated_debris_list(x, y, z, vx, vy, vz, sma, sizes, seen_list):
    if seen_list.size != 0:
        upd_x = np.delete(x, seen_list)
        upd_y = np.delete(y, seen_list)
        upd_z = np.delete(z, seen_list)
        upd_vx = np.delete(vx, seen_list)
        upd_vy = np.delete(vy, seen_list)
        upd_vz = np.delete(vz, seen_list)
        upd_sma = np.delete(sma, seen_list)
        upd_sizes = np.delete(sizes, seen_list)
        return upd_x, upd_y, upd_z, upd_vx, upd_vy, upd_vz, upd_sma, upd_sizes
    else:
        return x, y, z, vx, vy, vz, sma, sizes

def normalize(vector):
    """
    This function normalizes 3D vector along 3rd axis
    :param vector: ndarray, [N debris, 3 coordinates, t points of time]
    :return: normalized vector
    """
    abs_vector = norm(vector, axis=2).reshape([vector.shape[0], vector.shape[1], -1])
    normed_vector = vector / abs_vector
    return normed_vector

def get_earth_sun_direction(epoch, times):
    return np.array(get_sun(epoch + times).cartesian.xyz.to_value(u.km))

def get_debris_sun_direction(earth_obj_vectors, earth_sun_vectors):
    # print('>> detection:', 'getting sun direction')
    obj_to_sun = earth_sun_vectors - earth_obj_vectors
    return normalize(obj_to_sun)

def get_debris_sat_vectors(sat_vectors, debris_vectors):
    return sat_vectors - debris_vectors

def get_debris_sat_distance(debris_sat_vectors):
    return norm(debris_sat_vectors, axis=1)

def check_distance(max_distance, debris_sat_distance):
    return np.less(debris_sat_distance, max_distance * np.ones_like(debris_sat_distance))

def get_phase_angle(debris_sat_vectors, debris_sun_vectors):
    # print('>> detection:', 'getting phase angle')

    # reshaped_deb_sun = debris_sun_vectors.reshape([debris_sun_vectors.shape[0] * debris_sun_vectors.shape[1], -1])
    # reshaped_deb_sat = debris_sat_vectors.reshape([debris_sat_vectors.shape[0] * debris_sat_vectors.shape[1], -1])
    # print('reshaped_deb_sun', reshaped_deb_sun.shape, 'reshaped_deb_sat', reshaped_deb_sat.shape)

    # cross_product1 = np.cross(debris_sat_vectors, debris_sun_vectors, axisa=0)
    # cross_product = np.cross(reshaped_deb_sun, reshaped_deb_sat, axis=1)
    dot_product = np.einsum('ijk, jk -> ik', debris_sat_vectors, debris_sun_vectors)
    norms_product = norm(debris_sun_vectors, axis=0) * norm(debris_sat_vectors, axis=1)

    # norms_product = norm(sensor_vector, axis=0) * norm(obj_vector, axis=1)
    # return np.arccos(dot_product / norms_product)
    # np.einsum('ijk, jk -> ik', obj_vector, sensor_vector)
    # phase_angles = np.arctan2(norm(cross_product, axis=2), dot_product)
    phase_angles = np.arccos(dot_product / norms_product)
    # print('dot_product', dot_product.shape, 'norms_product', norms_product.shape)
    # print('phase_angles', phase_angles.shape)
    return phase_angles

def get_magnitudes(debris_sizes, earth_sun_vectors, debris_sat_vectors, debris_sat_dist, timeline):
    # print('>> detection:', 'getting magnitudes')
    phase_angles = get_phase_angle(debris_sat_vectors, earth_sun_vectors)

    rho = 0.175  # Bond albedo
    F_dif = 2 / (3 * pi**2) * ((pi - phase_angles) * np.cos(phase_angles) + np.sin(phase_angles))
    F_sp = 1 / (4 * pi)
    beta = 0.5  # mixing coefficient

    d = debris_sizes.to_value(u.km)
    A = (np.ones([timeline.size, d.size]) * (pi * d**2 / 4)).T

    m_obj = -26.74 - 2.5 * np.log10(A * rho * (beta * F_dif + (1 - beta) * F_sp)) + 5 * np.log10(debris_sat_dist)

    return m_obj

def check_magnitudes(limiting_magnitude, debris_magnitudes):
    return np.less(debris_magnitudes, limiting_magnitude * np.ones_like(debris_magnitudes))

def check_sun_debris_los(debris_sma, debris_sun_vectors):
    """
    Line of sight

    :param debris_sma:
    :param debris_to_sun_vectors:
    :return:
    """
    earth_sun_distance = 149.60 * 10 ** 6  # km

    max_los = np.sqrt(earth_sun_distance**2 - Earth_radius**2) + \
              np.sqrt((Earth_radius + debris_sma)**2 - Earth_radius**2)
    los = norm(debris_sun_vectors, axis=1)
    # print('los', los.shape, 'max_los', (np.repeat(max_los[:,np.newaxis], los.shape[1], axis=1)).shape)
    return np.less(los, np.repeat(max_los[:,np.newaxis], los.shape[1], axis=1))

def get_angle_eciframe(sensor_vector, obj_vector):
    # sensor_vector [3,t] | obj_vector [N,3,t]
    # print('>> detection:', 'obj vector', obj_vector.shape, ' |  sensor vector', sensor_vector.shape)
    dot_product = np.einsum('ijk, jk -> ik', obj_vector, sensor_vector) # [N,t]
    norms_product = norm(obj_vector, axis=1) * norm(sensor_vector, axis=0).T # [N,t]
    # print('>> detection:', 'dot_product', dot_product.shape, ' |  norms_product', norms_product.shape)
    return np.arccos(dot_product / norms_product)

def check_if_in_fov(max_fov, sensor_vector, obj_vector_eciframe):
    angle = get_angle_eciframe(sensor_vector, obj_vector_eciframe)
    return np.less(angle, max_fov/2 * np.ones_like(angle))