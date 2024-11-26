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

# IC: Dumbbell
u = np.zeros(domain.shape)
for i,ii in enumerate(x[:,0]):
    for j,jj in enumerate(y[0]):
        u[i,j] = ((ii-4)**2+(jj-10)**2<=4) or ((ii-16)**2+(jj-10)**2<=4) or ((ii>5 and ii<15) and (jj<=11) and (jj>=9))
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

while vb_problem.t < 10-1e-5:
    vb_problem.step(dt)
    u_plt.set_data(u.T)
    plt.pause(.01)

plt.pause(1)


