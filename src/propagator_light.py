import numpy as np
from numpy import pi

from poliastro.bodies import Earth
from astropy import units as u

from poliastro.core.elements import rv2coe

Re = Earth.R.to_value(u.km) # mean Earth equtorial radius
mu = Earth.k.to_value(u.km ** 3 / u.s ** 2) # Earth gravitation parameter
J2 = Earth.J2.value # first zonal harmonic coefficient in the expansion of the Earth's gravity field

def propagate_with_rotation_matrix(sma, inc, RAAN0, u0, t):

    dRAAN = -1.5 * J2 * np.sqrt(mu) * Re**2 / np.power(sma, 7/2) * np.cos(inc) # right ascension of the ascending node precession
    arglat = np.sqrt(mu / sma ** 3) * (1 - 1.5 * J2 * (Re / sma)**2 * (1 - 4 * np.cos(inc))) # argument of latitude precession
    # print('>> debris propagator:', 'arglat', arglat.shape)
    # print('>> debris propagator:', 't =', t.shape)
    # print('>> debris propagator:', (u0).shape)

    u = (u0 + np.outer(t, arglat)).T # argument of latitude
    # print('>> debris propagator:', 'u =', u.shape)

    RAAN = (RAAN0 + np.outer(t, dRAAN)).T
    # print('>> debris propagator:', 'RAAN =', RAAN.shape)
    # print('>> debris propagator:', 'sma =', sma.shape)
    # Rotation matrix
    print('>> debris propagator:', np.array(np.einsum('i, ij -> ij', sma,
                             (np.cos(u) * np.cos(RAAN) - np.sin(u) * np.einsum('i, ij -> ij', np.cos(inc), np.sin(RAAN))))).shape)

    x = np.array(np.einsum('i, ij -> ij', sma,
                           (np.cos(u) * np.cos(RAAN) - np.sin(u) * np.einsum('i, ij -> ij', np.cos(inc), np.sin(RAAN)))))
    y = np.array(np.einsum('i, ij -> ij', sma,
                           np.cos(u) * np.sin(RAAN) + np.sin(u) * np.einsum('i, ij -> ij', np.cos(inc), np.cos(RAAN))))
    z = np.array(np.einsum('i, ij -> ij', sma,
                           np.einsum('i, ij -> ij', np.sin(inc), np.sin(u))))

    r = np.stack([x, y, z], axis=1)

    return r

def get_some_orbital_elements(r, v):
    inc, raan, nu = np.zeros(r.shape[0]), np.zeros(r.shape[0]), np.zeros(r.shape[0])
    for i in range(r.shape[0]):
        _, _, inc[i], raan[i], _, nu[i] = rv2coe(mu, r[i], v[i])
    return inc, raan, nu
