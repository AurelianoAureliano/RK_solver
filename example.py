# Ryo Kamata

# import statements
from rk_solvers import *
import matplotlib.pyplot as plt

plot = 1
save = 1

# Lorenz system derivative and parameters
def derivative_lorenz(t, y, sigma, rho, beta):
    """
    Compute the derivatives for the Lorenz system.

    Args:
        t (float): independent variable, time.
        y (ndarray): dependent variables x, y and z in Lorenz system.
        sigma (float): system parameter of Lorenz system.
        rho (float): system parameter of Lorenz system.
        beta (float): system parameter of Lorenz system.

    Returns:
        f (ndarray): derivatives of x, y and z in Lorenz system.
    """
    f = np.zeros(3)
    f[0] = sigma * (y[1] - y[0])
    f[1] = y[0] * (rho - y[2]) - y[1]
    f[2] = y[0] * y[1] - beta * y[2]
    
    return f

sigma = 10.
rho = 28.
beta = 8. / 3.

t0 = 0.
t1 = 40.
atol = 10.e-5
args = [sigma, rho, beta]

y0 = np.array([1.]*3)
t, y = dp_solver_adaptive_step(derivative_lorenz, y0, t0, t1, atol, *args)

plt.figure(figsize=(9,7))

plt.subplot(2,2,1)
plt.title("X-Z Phase Diagram")
plt.plot(y[:,0], y[:,2], label="xyz(0)=1.0")

for i in range(0,3):
    plt.subplot(2,2,2+i)
    plt.title(f"{['x','y','z'][i]}(t)")
    plt.plot(t, y[:,i], label="xyz(0)=1.0")

plt.tight_layout()

if save:
    plt.savefig("example_output.png")

if plot:
    plt.show()