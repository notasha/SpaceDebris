import numpy as np
from scipy.integrate import DOP853, solve_ivp

from astropy import units as u
from poliastro.bodies import Earth

Earth_mu = Earth.k.to_value(u.km ** 3 / u.s ** 2)
Earth_radius = Earth.R.to_value(u.km)
Earth_J2 = Earth.J2.value

def func_twobody(t, u_, k, J2_):
    """Differential equation for the initial value two body problem.
    This function follows Cowell's formulation.
    Parameters
    ----------
    t : float
        Time.
    u_ : numpy.ndarray
        Six component state vector [x, y, z, vx, vy, vz] (km, km/s).
    k : float
        Standard gravitational parameter.
    """
    x, y, z, vx, vy, vz = u_.reshape(-1, 6).T
    r3 = np.power((x**2 + y**2 + z**2), 1.5)
    if J2_:
        J2 = J2_perturbation(x, y, z)
        du = np.column_stack([vx, vy, vz, -k * x / r3 + J2[0], -k * y / r3 + J2[1], -k * z / r3 + J2[2]])
    else:
        du = np.column_stack([vx, vy, vz, -k * x / r3, -k * y / r3, -k * z / r3])

    return du.reshape([u_.size])

def J2_perturbation(x, y, z):

    r0 = np.linalg.norm([x, y, z], axis=0)
    r2 = np.power(r0, 2)
    r5 = np.power(r0, 5)
    z2r2 = np.divide(np.power(z, 2), r2)

    J2 = (1.5 * Earth_J2 * Earth_mu * Earth_radius ** 2 *
          np.divide(np.array([x * (5 * z2r2 - 1), y * (5 * z2r2 - 1), z * (5 * z2r2 - 3)]), r5))

    return J2

def propagate(x, y, z, vx, vy, vz, tofs, f=func_twobody, rtol=1e-11, J2=False):
    """

    :param grav_param: ~astropy.units.Quantity | Standard gravitational parameter of the attractor
    :params x, y, z : ~astropy.units.Quantity | Position components
    :param vx, vy, vz: ~astropy.units.Quantity | Velocity components
    :param tofs: ~astropy.units.Quantity | Array of times to propagate
    :param f: function(t0, u, k), optional | Objective function, default to Keplerian-only forces
    :param rtol: float, optional | Maximum relative error permitted, defaults to 1e-11.
    :param J2: boolean, optional | J2 Perturbations, turned off by default

    :return:
    """
    # Convert input parameters to the standart units and get only values

    # print('t2', tofs.size)
    u0 = np.column_stack([x.to_value(u.km),
                          y.to_value(u.km),
                          z.to_value(u.km),
                          vx.to_value(u.km / u.s),
                          vy.to_value(u.km / u.s),
                          vz.to_value(u.km / u.s)]).reshape([x.size * 6])

    # print('u0', u0.shape)
    # print('x', x.size)

    # # Divide timeline into batches
    # batch_size = 500 # seconds
    # N = int(propagation_time / batch_size)
    # print('N =', N)
    #
    # # Integrate
    # solution = []
    # for t in range(N):
    #     tofs = np.linspace(t * batch_size, (t+1) * batch_size, points_in_batch)
    #     print('t', t, '=', tofs)
    #     result = solve_ivp(
    #         fun=f,
    #         t_span=[min(tofs), max(tofs)],
    #         t_eval=tofs,
    #         y0=u0,
    #         method='DOP853',
    #         vectorized=True,
    #         args=(Earth_mu, J2),
    #         rtol=rtol,
    #         atol=1e-12
    #     )
    #     solution.append(result.y)
    #     print('result t', result.t)
    # prop_rv = np.array(solution).reshape([-1, 6, points_in_batch * N])

    result = solve_ivp(
        fun=f,
        t_span=[min(tofs), max(tofs)],
        t_eval=tofs,
        y0=u0,
        method='DOP853',
        vectorized=True,
        args=(Earth_mu, J2),
        rtol=rtol,
        atol=1e-12
    )
    # print('>> propagator:', 'result', result.y.shape)
    prop_rv = np.array(result.y).reshape([-1,6,tofs.size])
    # print(prop_rv.shape)
    return prop_rv[:,:3,:], prop_rv[:,3:,:]


# def J2(r):
#     r0 = np.linalg.norm(r)
#     r2 = r0 ** 2
#     r5 = r0 ** 5
#     z2r2 = r[2] ** 2 / r2
#
#     J2 = (1.5 * Earth_J2 * Earth_mu * Earth_radius ** 2 *
#           np.array([r[0] * (5 * z2r2 - 1), r[1] * (5 * z2r2 - 1), r[2] * (5 * z2r2 - 3)]) / r5)
#     return J2
#
#
# def rhs_p3(t, x):
#     v = np.array(x[3:])
#     r = np.array(x[0:3])
#
#     dxdt = np.zeros(6)
#     dxdt[0:3] = v
#     dxdt[3:] = -Earth_mu * r / np.linalg.norm(r) ** 3 + J2(r)
#
#     return dxdt