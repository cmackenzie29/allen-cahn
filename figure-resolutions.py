import numpy as np
import matplotlib.pyplot as plt
import finite
import timesteppers
import equations
import matplotlib.colors

resolution = 50
spatial_order = 2
grid_x = finite.UniformPeriodicGrid(resolution, 20)
grid_y = finite.UniformPeriodicGrid(resolution, 20)
domain = finite.Domain((grid_x, grid_y))
x, y = domain.values()

epsilon = 0.5

# IC: Random
np.random.seed(10)
IC = np.sign(np.random.rand(domain.shape[0], domain.shape[1])-0.5)
u = IC.copy()

vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.1

plt.subplot(2,3,1)
plt.title("Resolution: 50")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem.t < 5-1e-5:
    vb_problem.step(dt)

plt.subplot(2,3,4)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])




resolution = 100
grid_x = finite.UniformPeriodicGrid(resolution, 20)
grid_y = finite.UniformPeriodicGrid(resolution, 20)
domain = finite.Domain((grid_x, grid_y))
x, y = domain.values()

# IC: Random
u = np.zeros(domain.shape)
for i in range(len(IC)):
    for j in range(len(IC)):
        u[2*i:2*i+2, 2*j:2*j+2] = IC[i,j]


vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.1

plt.subplot(2,3,2)
plt.title("Resolution: 100")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem.t < 5-1e-5:
    vb_problem.step(dt)

plt.subplot(2,3,5)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])




resolution = 200
grid_x = finite.UniformPeriodicGrid(resolution, 20)
grid_y = finite.UniformPeriodicGrid(resolution, 20)
domain = finite.Domain((grid_x, grid_y))
x, y = domain.values()

# IC: Random
u = np.zeros(domain.shape)
for i in range(len(IC)):
    for j in range(len(IC)):
        u[4*i:4*i+4, 4*j:4*j+4] = IC[i,j]


vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.1

plt.subplot(2,3,3)
plt.title("Resolution: 200")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


while vb_problem.t < 5-1e-5:
    vb_problem.step(dt)

plt.subplot(2,3,6)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])



plt.show()





