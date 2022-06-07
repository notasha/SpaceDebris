import numpy as np
from numpy import pi
from numpy.linalg import norm
import time

from poliastro.bodies import Earth
from poliastro.core.elements import rv2coe

from astropy import units as u
from astropy.time import TimeDelta, Time


Earth_mu = Earth.k.to(u.km ** 3 / u.s ** 2)

Earth_radius = Earth.R.to(u.km)

class Debris():
    """
    The module contains parameters of debris.

    In order to use your own orbit radiuses with the step you want, turn on redistribution
    while creating a debris object and specify min and max orbit radiuses you want to use
    and the step size. For example, you do this

        debris = Debris(datafile='your_data.csv', redistribution=True, min_radius=600,
                        max_radius=1000, radius_step=10)

    Then, you would use the following radiuses as orbits

        600, 610, 620, ... 990, 1000

    """

    def __init__(self, datafile, seen_debris=None, timeline=None, epoch=None):

        self.sma, self.densities = self.csv_to_arrays(datafile)
        self.sizes = self.randomize_sizes(2, 7) * u.cm
        self.initial = self.RandomCoordinates(self.all_sma)

        # if isinstance(seen_debris, list):
        #     self.delete_seen_debris(seen_debris)

        # if isinstance(timeline, np.ndarray) and isinstance(epoch, Time):
        #     start = time.time()
        #     self.coordinates = self.propagate(timeline)
        #     end = time.time()
        #     print(end - start)

    def csv_to_arrays(self, datafile):
        """
        This function reads spatial density data from csv file. The csv file is
        formed manually with the data got from MASTER.

        :param datafile: path to the csv file or just its name if it is in the the
        same directory as the current py file
        :return: two numpy arrays, the first one contains orbit radiuses and the other
        consists spatial densities

        Pandas version just in case ----------------------------------------------

        df = pd.read_csv(datafile, sep='\s+', header=None, usecols=[0, 18],
        engine="python")
        df.columns = ['alt', 'density']
        """

        data = np.genfromtxt(datafile, usecols=(0,18))
        sma = data[:,0] + Earth_radius.value
        densities = data[:,1]

        return sma, densities

    class RandomCoordinates():
        def __init__(self, all_sma):
            self.x, self.y, self.z, self.vx, self.vy, self.vz = self.get_random_state_vectors(all_sma)

        def get_random_state_vectors(self, all_sma):

            print('>> debris:', 'randomizing vectors')
            x, y, z = self.randomize_vector_components(all_sma) * u.km

            velocity_magnitude = np.sqrt(Earth_mu.value / all_sma)
            vx, vy, vz = self.randomize_vector_components(velocity_magnitude) * u.km / u.s

            return x, y, z, vx, vy, vz

        def randomize_vector_components(self, magnitude):

            phi = np.random.uniform(low=0, high=2*pi, size=magnitude.size)
            theta = np.random.uniform(low=0, high=2*pi, size=magnitude.size)

            x_component = magnitude * np.cos(phi) * np.sin(theta)
            y_component = magnitude * np.sin(phi) * np.sin(theta)
            z_component = magnitude * np.cos(theta)

            return x_component.astype(np.float16), y_component.astype(np.float16), z_component.astype(np.float16)

    # def get_some_orbital_elements(self):
    #     r = np.column_stack([self.initial.x.value,
    #                          self.initial.y.value,
    #                          self.initial.z.value]).astype(np.float64)
    #     v = np.column_stack([self.initial.vx.value,
    #                          self.initial.vy.value,
    #                          self.initial.vz.value]).astype(np.float64)
    #
    #     inc, raan, nu = np.zeros(r.shape[0]), np.zeros(r.shape[0]), np.zeros(r.shape[0])
    #
    #     for i in range(r.shape[0]):
    #         _, _, inc[i], raan[i], _, nu[i] = rv2coe(Earth_mu, r[i], v[i])
    #
    #     return inc, raan, nu

    # @classmethod
    # def delete_seen_debris(cls, list_of_seen):
    #     cls.__init__.initial.x = np.delete(cls.initial.x, list_of_seen)
    #     cls.initial.y = np.delete(cls.initial.y, list_of_seen)
    #     cls.initial.z = np.delete(cls.initial.z, list_of_seen)
    #
    #     cls.initial.vx = np.delete(cls.initial.vx, list_of_seen)
    #     cls.initial.vy = np.delete(cls.initial.vy, list_of_seen)
    #     cls.initial.vz = np.delete(cls.initial.vz, list_of_seen)

    # def propagate(self, timeline):
    #     """
    #     This function calculates coordinates of the satellite for the specified times
    #     """
    #     coordinates = []
    #     for d in range(self.initial.x.size):
    #         # print('propagating', d)
    #         r = np.stack([self.initial.x[d], self.initial.y[d], self.initial.z[d]])
    #         v = np.stack([self.initial.vx[d], self.initial.vy[d], self.initial.vz[d]])
    #         prop_r, _ = cowell(Earth_mu, r, v, TimeDelta(timeline), rtol=1e-11)
    #         coordinates.append(prop_r.value)
    #     return np.array(coordinates)

    # def to_orbital_frame(self, satellite, debris_coordinates):
    #     """
    #     Here, we transit from ECI to the satellite orbital frame. Iteration through times
    #     :param satellite:
    #     :return:
    #     """
    #     coordinates_in_orbframe = np.zeros_like(debris_coordinates)
    #
    #     for t in range(debris_coordinates.shape[1]):
    #         coordinates_in_orbframe[:,t,:] = satellite.eci2orb(debris_coordinates[:,t,:])
    #     return coordinates_in_orbframe

    def randomize_sizes(self, min, max):
        return np.random.randint(min, max, self.total_num)

    @property
    def num_per_orbit(self):

        orb_volumes = 4/3 * pi * (self.sma[1:] ** 3 - self.sma[:-1] ** 3)
        debris_num_per_orbit = np.round_(self.densities[:-1] * orb_volumes)

        return debris_num_per_orbit.astype('int64')

    @property
    def total_num(self):
        return int(sum(self.num_per_orbit))

    @property
    def all_sma(self):
        return np.repeat(self.sma[:-1], self.num_per_orbit)

# TESTS -------------------------------------------------------------

# times = np.linspace(0, 1000, 2) * u.s
# # print(times)
# debris = Debris(datafile='very small test data.csv', timeline=times)
# print(debris.state_vectors.shape)
# # print(debris.coordinates)
# print(debris.coordinates.shape)


# print(np.repeat(debris.spatial_density[:-1,0], debris.num_per_orbit).shape)
# print(debris.total_num)
# print(debris.spatial_density[:,0].shape)
# print(debris.num_per_orbit.shape)


# Parameters
# start = '2021-01-01 00:00'
# end = '2031-01-01 00:00'
# timestep = 1000

# BACKUP -------------------------------------------------------------

    # def redistribute_evenly(self, datafile, min_sma, max_sma, sma_step):
    #     """
    #     This method is developed just in case it will be needed to use other values of
    #     orbit radiuses with the same data. For example, in the csv file got from MASTER
    #     database altitudes change with the step of 3 km. If we need to operate with a
    #     10-km step, or 1-km step, this method will be of help.
    #
    #     :param datafile:
    #     :param min_sma:
    #     :param max_sma:
    #     :param sma_step:
    #     :return:
    #     """
    #
    #     data_sma, data_density = self.csv_to_arrays(datafile)
    #
    #     sma = np.arange(min_sma, max_sma, sma_step)
    #     density = np.zeros(data_density.size, dtype=float)
    #
    #     for line in range(data_sma.size-1):
    #
    #         # Create local variables
    #         current_orb, current_density = round(data_sma[line]), data_density[line]
    #         next_orb = data_sma[line+1]
    #
    #         for i in range(sma.size):
    #
    #             if sma[i] >= current_orb and sma[i] < next_orb:
    #                 density[i] = current_density
    #
    #     return sma, density