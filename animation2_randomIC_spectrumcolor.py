import numpy as np
import matplotlib.pyplot as plt
import finite
import timesteppers
import equations
import matplotlib.colors

resolution = 200
spatial_order = 2
grid_x = finite.UniformPeriodicGrid(resolution, 20)
grid_y = finite.UniformPeriodicGrid(resolution, 20)
domain = finite.Domain((grid_x, grid_y))
x, y = domain.values()

epsilon = 0.5

# IC: Random values ±1
np.random.seed(1)
u = np.sign(np.random.rand(domain.shape[0], domain.shape[1])-0.5)

vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.01

# 2D plots
plt.ion()
u_plt = plt.imshow(u.T, cmap="Spectral", origin='lower')
plt.colorbar()
plt.draw()

plt.pause(1)

while vb_problem.t < 5-1e-5:
    vb_problem.step(dt)
    u_plt.set_data(u.T)
    plt.pause(.01)

plt.pause(1)


