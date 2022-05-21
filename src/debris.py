import numpy as np
from numpy import pi
from numpy.linalg import norm
import time

from poliastro.bodies import Earth
from poliastro.twobody.propagation import cowell

from astropy import units as u
from astropy.time import TimeDelta, Time
from astropy.coordinates import get_sun

global Earth_mu
Earth_mu = Earth.k.to(u.km ** 3 / u.s ** 2)

global Earth_radius
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
    def __init__(self, datafile, timeline=None, epoch=None):

        self.sma, self.densities = self.csv_to_arrays(datafile)
        self.state_vectors = self.get_random_state_vectors()

        if isinstance(timeline, np.ndarray) and isinstance(epoch, Time):
            start = time.time()
            self.coordinates = self.propagate(timeline)
            end = time.time()
            print(end - start)

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

    def get_random_state_vectors(self):
        """

        :return:
        """
        print('randomizing vectors')
        r_x, r_y, r_z = self.randomize_vector_components(self.all_sma)

        velocity_magnitude = np.sqrt(Earth_mu.value / self.all_sma)
        v_x, v_y, v_z = self.randomize_vector_components(velocity_magnitude)

        return np.column_stack((r_x, r_y, r_z, v_x, v_y, v_z))

    def randomize_vector_components(self, magnitude):

        phi = np.random.uniform(low=0, high=2*pi, size=magnitude.size)
        theta = np.random.uniform(low=0, high=2*pi, size=magnitude.size)

        x_component = magnitude * np.cos(phi) * np.sin(theta)
        y_component = magnitude * np.sin(phi) * np.sin(theta)
        z_component = magnitude * np.cos(theta)

        return x_component.astype(np.float16), y_component.astype(np.float16), z_component.astype(np.float16)

    def propagate(self, timeline):
        """
        This function calculates coordinates of the satellite for the specified times
        """
        coordinates = []
        i = 0
        for deb in self.state_vectors:
            print('propagating', i+1)
            r, v = deb[:3] * u.km, deb[3:] * u.km/u.s
            prop_r, _ = cowell(Earth_mu, r, v, TimeDelta(timeline), rtol=1e-6)
            coordinates.append(prop_r.value)
            i += 1
        return np.array(coordinates, dtype=np.float16)

    def to_orbital_frame(self, satellite):
        """
        Here, we transit from ECI to the satellite orbital frame. Iteration through times
        :param satellite:
        :return:
        """
        coordinates_in_orbframe = np.zeros_like(self.coordinates)

        for t in range(self.coordinates.shape[1]):
            coordinates_in_orbframe[:,t,:] = satellite.eci2orb(self.coordinates[:,t,:])
        return coordinates_in_orbframe

    def get_magnitudes(self, debris_sat_vectors, debris_sat_dist, epoch, timeline):
        print('getting magnitudes')
        debris_sun_vectors = self.get_sun_direction(epoch, timeline)
        phase_angles = self.get_phase_angle(debris_sun_vectors, debris_sat_vectors)

        rho = 0.175  # Bond albedo
        F_dif = 2 / (3 * pi**2) * ((pi - phase_angles) * np.cos(phase_angles) + np.sin(phase_angles))
        F_sp = 1 / (4 * pi)
        beta = 0.5  # mixing coefficient

        d = self.randomize_sizes(2, 7) * 10**(-5)
        A = (np.ones([timeline.size, d.size]) * (pi * d**2 / 4)).T

        m_obj = -26.74 - 2.5 * np.log10(A * rho * (beta * F_dif + (1 - beta) * F_sp)) + 5 * np.log10(debris_sat_dist)

        return m_obj

    def get_sun_direction(self, epoch, times):
        print('getting sun direction')
        earth_to_sun = np.array(get_sun(epoch + times).cartesian.xyz.to(u.km).value).T
        earth_to_debris = self.coordinates
        debris_to_sun = earth_to_sun - earth_to_debris

        return self.normalize(debris_to_sun)

    def get_phase_angle(self, debris_sun_vectors, debris_sat_vectors):
        print('getting phase angle')
        reshaped_deb_sun = debris_sun_vectors.reshape([debris_sun_vectors.shape[0] * debris_sun_vectors.shape[1],-1])
        reshaped_deb_sat = debris_sat_vectors.reshape([debris_sat_vectors.shape[0] * debris_sat_vectors.shape[1],-1])

        cross_product = np.cross(reshaped_deb_sun, reshaped_deb_sat, axis=1)
        dot_product = np.dot(reshaped_deb_sun, reshaped_deb_sat.T)[0]
        phase_angles = np.arctan2(self.get_abs(cross_product), dot_product)

        return phase_angles.reshape([debris_sun_vectors.shape[0], debris_sun_vectors.shape[1]])

    def get_abs(self, vector):
        return np.sqrt(np.sum(np.abs(vector)**2, axis=-1))

    def normalize(self, vector):
        abs_vector = norm(vector, axis=2).reshape([vector.shape[0], vector.shape[1], -1])
        normed_vector = vector / abs_vector
        return normed_vector

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
# debris = Debris(datafile='test.csv', timeline=times)
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