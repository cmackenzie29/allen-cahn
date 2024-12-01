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

# IC: Torus
u = np.zeros(domain.shape)
for i,ii in enumerate(x[:,0]):
    for j,jj in enumerate(y[0]):
        u[i,j] = ((ii-10)**2+(jj-10)**2<=49) and ((ii-10)**2+(jj-10)**2>=16)
# Scale values to be ±1 rather than 0 & 1
u *= 2
u -= 1


vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.05

# 2D plots
plt.ion()
u_plt = plt.imshow(u.T, cmap="Spectral", origin='lower')
plt.colorbar()
plt.draw()

plt.pause(1)

while vb_problem.t < 50-1e-5:
    vb_problem.step(dt)
    u_plt.set_data(u.T)
    plt.pause(.0001)

plt.pause(1)

