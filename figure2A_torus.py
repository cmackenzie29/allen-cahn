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

# t = 0
plt.subplot(1,4,1)
plt.title("t = 0")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 5
while vb_problem.t < 5-1e-5:
    vb_problem.step(dt)

plt.subplot(1,4,2)
plt.title("t = 5")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 25
while vb_problem.t < 25-10e-5:
    vb_problem.step(dt)

plt.subplot(1,4,3)
plt.title("t = 25")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 50
while vb_problem.t < 50-1e-5:
    vb_problem.step(dt)

plt.subplot(1,4,4)
plt.title("t = 50")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()





