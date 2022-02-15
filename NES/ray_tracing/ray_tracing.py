import numpy as np
from scipy.integrate import solve_ivp

from .utils import nes_op_rts_right_part, nintegrate_ode_system, directions_dict, velocities_list

# Available solvers:
solvers_list = ["specified steps", "scipy"]


def nes_op_ray_tracing(x_0,
                       num_points,
                       nes_op,
                       target_front=0.01,
                       solver="specified steps",
                       direction="backward",
                       velocity="interpolation",
                       **kwargs):
    """
    Traces a ray given the receiver / starting point and a trained NES-OP network. The resulting ray is sorted in
    traveltime-increasing order.

    Arguments:
        x_0: numpy array (D,)
            Ray ending / starting point in D-dimensional space

        num_points: positive integer
            Desired number of ray points

        nes_op: NES-OP network
            A trained instance of the NES-OP network

        target_front: positive number
            Time of the target wavefront

        solver: string
            ODE solver. Two options are supported: "specified steps" (straightforward implementation of the Runge-Kutta
            method of 4th order without any accuracy checks) and "scipy" (scipy.integrate.solve_ivp)

        direction: "forward" or "backward"
            Tracing direction (forwards or backwards in time)

        velocity: string
            String defining how to evaluate wave velocity along the ray. Two options are supported: "interpolation"
            (use training velocity interpolation) and "learned velocity" (use inverse of the slowness, i.e. eikonal
            gradient)

        **kwargs: dictionary
            Keyword arguments for scipy.integrate.solve_ivp function

    Return:
        ray: numpy array (N, D)
            Array of ray points sorted in traveltime-increasing order

    """

    assert solver in solvers_list, ("Two solvers are supported: 'specified steps' and 'scipy'. " +
                                    "Instead {} is passed.".format(solver))
    assert direction in directions_dict, ("Two directions are supported: 'forward' and 'backward'. " +
                                          "Instead {} is passed.".format(direction))
    assert velocity in velocities_list, ("Two options are supported for velocity evaluation: 'interpolation' and " +
                                         "'learned velocity'.")

    # Time in the starting point:
    time_0 = np.squeeze(nes_op.Traveltime(np.atleast_2d(x_0)))

    if direction == "forward":

        assert target_front > time_0, "Target time must be greater than the initial one for froward tracing."

    else:

        assert target_front < time_0, "Target time must be less than the initial one for backward tracing."

    # Array of travel times:
    ray_travel_times = np.linspace(np.min([time_0, target_front]),
                                   np.max([time_0, target_front]),
                                   num_points)

    # Define the step sizes:
    steps = np.diff(ray_travel_times)

    # Set the first step and the max step for the integration:
    if not ("first_step" in kwargs):

        kwargs["first_step"] = np.min(steps) / 2

    if not ("max_step" in kwargs):

        kwargs["max_step"] = np.max(steps)

    # Solve the ray tracing system:
    if solver == "scipy":

        ray = np.transpose(solve_ivp(nes_op_rts_right_part,
                                     t_span=[ray_travel_times[0], ray_travel_times[- 1]],
                                     y0=x_0,
                                     args=[nes_op, direction, velocity],
                                     t_eval=ray_travel_times,
                                     **kwargs).y)

    else:

        ray = nintegrate_ode_system(nes_op_rts_right_part,
                                    x_0,
                                    ray_travel_times,
                                    args=[nes_op, direction, velocity])

    # Return the ray:
    return ray[:: directions_dict[direction]]


def nes_op_ray_amplitude(ray,
                         nes_op,
                         velocity="interpolation"):
    """
    Computes ray amplitude using a trained NES-OP network and assuming close source vicinity initial conditions.

    Arguments:
        ray: numpy array (N, D)
            Array of ray points in D-dimensional space sorted in time-increasing order. The first point must be at some
            distance from the source

        nes_op: NES-OP network
            A trained instance of the NES-OP network

        velocity: string
            String defining how to evaluate wave velocity along the ray. Two options are supported: "interpolation"
            (use training velocity interpolation) and "learned velocity" (use inverse of the slowness, i.e. eikonal
            gradient)

    Returns:
        amplitude: number
            Ray amplitude in the receiver

    """

    assert velocity in velocities_list, ("Two options are supported for velocity evaluation: 'interpolation' and " +
                                         "'learned velocity'.")

    # Problem dimensions:
    dims = np.shape(ray)[- 1]

    # Travel times along the ray:
    ray_times = np.squeeze(nes_op.Traveltime(ray))

    # Travel time Laplacians along the ray:
    laplacians = np.squeeze(nes_op.Laplacian(ray))

    # Wave velocities along the ray:
    if velocity == "interpolation":

        ray_vels = np.squeeze(nes_op.velocity(ray))

    else:

        ray_vels = np.squeeze(1 / np.sqrt(np.sum(nes_op.Gradient(ray) ** 2, axis=- 1)))

    # Wave velocity in the source:
    start_vel = np.squeeze(nes_op.velocity(nes_op.xs))

    # Initial-front amplitude:
    start_ampl = np.sqrt(start_vel) / np.sqrt(np.sum((nes_op.xs - ray[0]) ** 2)) ** (dims - 1)

    # Ray amplitude in the receiver:
    amplitude = start_ampl * np.exp(- 1 / 2 * np.trapz(ray_vels ** 2 * laplacians, ray_times))

    return amplitude
