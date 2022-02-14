import numpy as np

# A dictionary for ray directions:
directions_dict = {"forward": 1, "backward": - 1}


def nes_op_rts_right_part(ray_time, x_vec, nes_op, direction, vel_func=None):
    """
    Returns the right part of the ray tracing system evaluated at the point specified by the x vector using ray travel
    time as the sampling parameter and solution of the eikonal equation for slowness computation.

    Arguments:
        ray_time: number
            Travel time from the source to the current point

        x_vec: numpy array (D,)
            A ray point in D-dimensional space

        nes_op: NES-OP network
            A trained instance of the NES-OP network

        direction: "forward" or "backward"
            Tracing direction (forwards or backwards in time)

        vel_func: callable
            Function accepting point coordinates as argument and returning wave velocity in this point

    Return:
        r_part: numpy array (D,)
            Vector of the right-hand-side of the ray tracing system for its physical space component

    """

    # Slowness vector:
    slow = np.squeeze(nes_op.Gradient(np.atleast_2d(x_vec)))

    # Wave velocity:
    if callable(vel_func):

        vel = np.squeeze(vel_func(x_vec))

    else:

        vel = 1 / np.sqrt(np.sum(slow ** 2))

    # Return the right part:
    r_part = slow * (vel ** 2) * directions_dict[direction]

    return r_part


def nintegrate_ode_system(rhs_func, y_0, x, args=(), **kwargs):
    """
    Integrates a system of ODEs dy / dx == rhs_func(x, y), y(x[0]) == y_0 using fourth order

    Arguments:
        rhs_func: callable
            Function accepting X-coordinate and current Y-value and returning right-hand-side of the ODE system

        y_0: numpy array (D,)
            Initial value

        x: numpy array (N,)
            Array of X-coordinates at which to evaluate the solutions

        args: tuple
            Additional positional arguments for the rhs_func

        kwargs: dict
            Additional keyword arguments for the rhs_func

    Returns:
        y: numpy array (N, D)
            Array of ODE system's solution points

    """

    # Number of ray points:
    num_points = len(x)

    # Integration steps:
    steps = x[1:] - x[: - 1]

    # Array for ODE solutions:
    y = np.zeros((num_points,) + np.shape(y_0), dtype=np.float64)
    y[0] = y_0

    for i in range(num_points - 1):

        # Current point:
        x_curr = x[i]

        # ODE solution at the current point:
        y_curr = y[i]

        # Current step along the ray:
        step_curr = steps[i]

        # Runge-Kutta terms:
        k_1 = rhs_func(x_curr, y_curr, *args, **kwargs)
        k_2 = rhs_func(x_curr + step_curr / 2, y_curr + step_curr / 2 * k_1, *args, **kwargs)
        k_3 = rhs_func(x_curr + step_curr / 2, y_curr + step_curr / 2 * k_2, *args, **kwargs)
        k_4 = rhs_func(x_curr + step_curr, y_curr + step_curr * k_3, *args, **kwargs)

        # Find the solution at the next step:
        y[i + 1] = y_curr + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * step_curr / 6

    # Return the solution:
    return y
