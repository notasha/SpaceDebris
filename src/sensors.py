import numpy as np

from debris import *
from satellites import *
# from visibility_conditions import *

class Sensor():
    def __init__(self, direction_vector, fov, max_dist, limiting_magnitude, sat_r=None, sat_v=None):
        self.direction_vector_orb = np.array(direction_vector)
        self.fov = fov.to_value(u.rad)
        self.max_dist = max_dist.value
        self.limiting_magnitude = limiting_magnitude

        if isinstance(sat_r, np.ndarray) and isinstance(sat_v, np.ndarray):
            self.r = sat_r
            self.v = sat_v
            self.direction_vector_eci = self.direction_vector_to_eci(self.r, self.v)

    def direction_vector_to_eci(self, r, v):
        angular_momentum = np.cross(r, v, axis=1).reshape([3,r.shape[2]])
        norms = norm(angular_momentum, axis=0)
        normed_angular_momentum = np.zeros_like(angular_momentum)

        # print('>> sensor:', 'ang momentum', angular_momentum.shape, ' |  norms', norms.shape)

        for i in range(norms.size):
            normed_angular_momentum[:,i] = angular_momentum[:,i] / norms[i]
        return -normed_angular_momentum # [3, t]

    # def get_angle_orbframe(self, obj_vector_in_orbframe):
    #     cross_product = np.cross(obj_vector_in_orbframe, self.direction_vector_orb)
    #     dot_product = np.dot(obj_vector_in_orbframe, self.direction_vector_orb)
    #     return np.arctan2(norm(cross_product, axis=2), dot_product)