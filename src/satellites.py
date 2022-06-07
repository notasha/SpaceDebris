import numpy as np
from numpy import pi

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
# from poliastro.twobody.propagation import propagate, cowell
from poliastro.core.elements import coe2rv

from astropy import units as u
from astropy.time import TimeDelta, Time

import quaternion

Earth_mu = Earth.k.to_value(u.km ** 3 / u.s ** 2)
Earth_radius = Earth.R.to_value(u.km)
Earth_J2 = Earth.J2.value

class Satellite():
    def __init__(self, sma, ecc=0*u.one, inc=0*u.deg, raan=0*u.deg, argp=0*u.deg, nu=0*u.deg, epoch=None):
        """
        The module keeps all the satellite parameters together

          alt: altitude over the Earth's surface
          sma: semi-major axis
          ecc: eccentricity, default to 0 (circular orbit)
          inc: inclination, default to 0 deg (equatorial orbit)
          raan: right ascension of the ascending node, default to 0 deg
          argp: argument of the perigee, default to 0 (circular orbit)
          nu: true anomaly, default to undefined (circular orbit)
          epoch: Epoch
          times: array of time moments for which positions of the sat needed to be
          calculated
        """

        self.sma, self.ecc, self.inc = sma.to(u.km), ecc.to(u.one), inc.to(u.rad)
        self.raan, self.argp, self.nu = raan.to(u.rad), argp.to(u.rad), nu.to(u.rad)

        self.initial = self.InitialState(self.sma, self.ecc, self.inc,
                                         self.raan, self.argp, self.nu)

        if isinstance(epoch, Time):
            self.epoch = epoch

        # # Orbital frame quaternions
        # self.q1 = self.quat(angle=self.raan.value, axis_vector=[0, 0, 1])
        # self.q2 = self.quat(angle=self.inc.value, axis_vector=[1, 0, 0])
        # self.q3 = self.quat(angle=self.nu.value, axis_vector=[0, 0, 1])

    # @property
    # def orbit(self):
    #     """
    #     This function creates a near-Earth orbit from 6 classical orbital elements
    #     """
    #     return Orbit.from_classical(Earth, self.sma, self.ecc, self.inc, self.raan,
    #                                 self.argp, self.nu, self.epoch)

    class InitialState():
        def __init__(self, sma, ecc, inc, raan, argp, nu):
            r, v = coe2rv(Earth_mu, sma, ecc, inc, raan, argp, nu)
            self.x, self.y, self.z = r * u.km
            self.vx, self.vy, self.vz = v * u.km / u.s

    # def propagate(self, timeline):
    #     """
    #     This function calculates coordinates of the satellite for the specified times
    #     """
    #     coordinates = self.orbit.propagate(TimeDelta(timeline), method=cowell)
    #     return coordinates

    # def quat(self, angle, axis_vector):
    #     w = np.cos(angle / 2)
    #     x, y, z = np.sin(angle / 2) * np.array(axis_vector)
    #     return np.quaternion(w, x, y, z)
    #
    # def eci2orb(self, vectors_in_eci):
    #     rotation_quat = (self.q3 * self.q2) * self.q1
    #     return quaternion.rotate_vectors(rotation_quat, vectors_in_eci)
    #
    # def orb2eci(self, orb_vectors):
    #     rotation_quat = ((self.q3 * self.q2) * self.q1).inverse()
    #     return quaternion.rotate_vectors(rotation_quat, orb_vectors)

# TESTS -------------------------------------------------------------

# epoch = Time('2022-01-01 00:00:00.000', format='iso', scale='utc')
# N = 50 # number of points

# sma=670 * u.km + Earth.R.to(u.km)
# inc=98.06 * u.deg
# raan = 11.2 * u.deg

# times = np.linspace(0, 10000, 2)
#
# sat1 = Satellite(sma = 670*u.km + Earth.R.to(u.km), inc = 98.06*u.deg, raan = 11.2*u.deg, timeline=times)
# print(np.asarray(sat1.coordinates.xyz).reshape([2,3]))
# print(sat1.coordinates[:3])



# coords = sat1.coordinates(2)
# # print(sat1.orbit.get_frame())
# print(coords[0])
#
# from poliastro.core.elements import coe2rv, rv2coe
#
# posit = coe2rv(k= Earth.k.to(u.km ** 3 / u.s ** 2),
#                p= sma, ecc= 0 * u.one, inc=inc, raan=raan, argp=0* u.deg, nu=0* u.deg)
# print(posit)
#
# coe = rv2coe(k = Earth.k.to(u.km ** 3 / u.s ** 2), r = posit[0], v = posit[1])
# print(coe)

# print(670 * u.km + Earth.R.to(u.km))
# print((670 * u.m + Earth.R.to(u.km)).to(u.km))