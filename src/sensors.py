import numpy as np
from numpy.linalg import norm

from astropy import units as u

from debris import *

class Sensor(Debris):
    def __init__(self, direction_vector, fov, max_dist, limiting_magnitude):
        self.direction_vector = np.array(direction_vector) / norm(direction_vector)
        self.fov = fov.to(u.rad).value
        self.max_dist = max_dist.value
        self.limiting_magnitude = limiting_magnitude

    def get_abs(self, vector):
        return np.sum(np.abs(vector)**2, axis=-1)**(1./2)

    def get_angle(self, obj_vector_in_orbframe):
        cross_product = np.cross(obj_vector_in_orbframe, self.direction_vector)
        dot_product = np.dot(obj_vector_in_orbframe, self.direction_vector)

        return np.arctan2(self.get_abs(cross_product), dot_product)

    def check_what_is_visible(self, Debris, satellite, epoch, timeline, first_iteration=True, debris_left=None):

        # detection_counter = np.zeros(debris_vectors_in_orbframe.shape[0])
        if first_iteration:
            sat_debris_vectors = Debris.to_orbital_frame(satellite)
            sat_debris_distance = self.get_abs(sat_debris_vectors)
            sensor_debris_angles = self.get_angle(sat_debris_vectors)
            magnitudes = Debris.get_magnitudes(-sat_debris_vectors, sat_debris_distance, epoch, timeline)

            unseen_debris = np.array([sat_debris_distance, sensor_debris_angles, magnitudes])
        else:
            unseen_debris = debris_left

        seen_num = 0

        for t in range(timeline.size):
            seen_counter = []
            for i in range(unseen_debris.shape[1]):
                if unseen_debris[0,i,t] < self.max_dist and unseen_debris[1,i,t] < self.fov and unseen_debris[2,i,t] < self.limiting_magnitude:
                    seen_counter.append(i)
                    seen_num += 1
            unseen_debris = np.delete(unseen_debris, seen_counter, axis=1)

        seen_percent = int(seen_num / Debris.total_num * 100)

        return unseen_debris, seen_num, seen_percent