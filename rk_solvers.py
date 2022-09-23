# import statements
import numpy as np

def explicit_solver_fixed_step(func, y0, t0, t1, h, alpha, beta, gamma, *args):
    """
    Compute solution(s) to ODE(s) using any explicit RK method with fixed step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial condition(s) for dependent variable(s).
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        h (float): fixed step size along independent variable.
        alpha (ndarray): weights in the Butcher tableau.
        beta (ndarray): nodes in the Butcher tableau.
        gamma (ndarray): RK matrix in the Butcher tableau.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s) solved at t values.
    """
    steps = int(np.ceil((t1-t0) / h))
    n = len(beta)

    t = t0 + h * np.arange(steps + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    f = np.zeros((n, len(y0)))

    for k in range(steps):

        for i in range(n):
            f[i] = func(t[k] + beta[i] * h, y[k] + h * np.dot(gamma[i], f), *args)
        
        y[k+1] = y[k] + h * np.dot(alpha, f)

    return t, y


def dp_solver_adaptive_step(func, y0, t0, t1, atol, *args):
    """
    Compute solution to ODE using the Dormand-Prince embedded RK method with an adaptive step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial conditions for each solution variable.
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        atol (float): error tolerance for determining adaptive step size.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s).
    """
    alpha = np.array([
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    ])
    beta = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    gamma = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    phi = 0.9

    n = len(beta)
    f = np.zeros((n, len(y0)))
    h = 10.0e-3
    t = t0
    k = 0
    y = {k:[y0, t]}

    while t < t1:
        y[k+1] = [np.zeros(len(y0)), 0]

        for i in range(0,n):
            f[i] = func(t + beta[i] * h, y[k][0] + h * np.dot(gamma[i][:i], f[:i]), *args)
        
        fourth_order = y[k][0] + h * np.dot(alpha[0], f)
        y[k+1][0] = y[k][0] + h * np.dot(alpha[1], f)

        error = max(abs(y[k+1][0] - fourth_order))

        if error <= atol:
            k += 1
            t += h
            y[k][1] = t
            h *= phi * ((atol / error) ** (1/4))
            
            if t + h > t1:
                h = t1 - t
        else:
            h *= phi * ((atol / error) ** (1/5))


    t, y = zip(*[(y[k][1], y[k][0]) for k in y])
    return t, np.array(y)


def backward_euler_solver(func, y0, t0, t1, h, tol, max_iter, *args):
    """
    Compute solution to ODE using the Backward Euler's method.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial conditions for each solution variable.
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        h (float): fixed step size along independent variable.
        tol (float): error tolerance for determining adaptive step size.
        max_iter (int): maximum iteration to try until moving on to the next point.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s).
    """
    steps = int(np.ceil((t1-t0) / h))

    t = t0 + h * np.arange(steps + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for k in range(steps):
        error = 9999
        i = 0 
        y[k+1] = y[k]

        while (error >= tol and i < max_iter):
            previous = y[k+1]
            y[k+1] = y[k] + h * func(t[k+1], y[k+1], *args)
            error = abs(previous[0] - y[k+1][0]) / (1 + abs(y[k+1][0]))
            i += 1

    return t, y
