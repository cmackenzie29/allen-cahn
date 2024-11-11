import numpy as np
import matplotlib.pyplot as plt
import finite
import timesteppers
import equations
import matplotlib.colors

resolution = 200
spatial_order = 8
grid_x = finite.UniformPeriodicGrid(resolution, 20)
grid_y = finite.UniformPeriodicGrid(resolution, 20)
domain = finite.Domain((grid_x, grid_y))
x, y = domain.values()

epsilon = 0.5

np.random.seed(1)
u = np.sign(np.random.rand(domain.shape[0], domain.shape[1])-0.5)
#u[:] = np.exp(-np.sqrt((x-10)**2 + (y-10)**2)**2/4)

#dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)
#dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)
vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)#timesteppers.RK22(equations.Advection_System(u,v,dx,dy))

alpha = 0.1
dt = alpha*grid_x.dx

# 2D plots
plt.ion()
u_plt = plt.imshow(u, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']))
plt.colorbar()
plt.draw()

while vb_problem.t < 10-1e-5:
    vb_problem.step(dt)
    u_plt.set_data(u)
    plt.pause(.01)

# plt.imshow(u, cmap="bwr")
# plt.colorbar()
# plt.show()



