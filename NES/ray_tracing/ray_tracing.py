import numpy as np
from scipy.integrate import solve_ivp

from .utils import nes_op_rts_right_part, nintegrate_ode_system, directions_dict

# Available solvers:
solvers = ["specified steps", "scipy"]


def nes_op_tracing(x_0,
                   num_points,
                   nes_op,
                   solver="specified steps",
                   direction="backward",
                   target_front=0.001,
                   vel_func=None,
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

        solver: string
            ODE solver. Two options are supported: "specified steps" (straightforward implementation of the Runge-Kutta
            method of 4th order without any accuracy checks) and "scipy" (scipy.integrate.solve_ivp)

        direction: "forward" or "backward"
            Tracing direction (forwards or backwards in time)

        target_front: positive number
            Time of the target wavefront

        vel_func: callable
            Function accepting point coordinates as argument and returning wave velocity in this point

        **kwargs: dictionary
            Keyword arguments for scipy.integrate.solve_ivp function

    Return:
        ray: numpy array (N, D)
            Array of ray points sorted in traveltime-increasing order

    """

    assert solver in solvers, ("Two solvers are supported: 'specified steps' and 'scipy'. " +
                               "Instead {} is passed.".format(solver))
    assert direction in directions_dict, ("Two directions are supported: 'forward' and 'backward'. " +
                                          "Instead {} is passed.".format(direction))

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
                                     args=[nes_op, direction, vel_func],
                                     t_eval=ray_travel_times,
                                     **kwargs).y)

    else:

        ray = nintegrate_ode_system(nes_op_rts_right_part,
                                    x_0,
                                    ray_travel_times,
                                    nes_op=nes_op,
                                    direction=direction,
                                    vel_func=vel_func)

    # Return the ray:
    return ray[:: directions_dict[direction]]