import numpy as np
import matplotlib.pyplot as plt
import finite
import timesteppers
import equations
import matplotlib.colors

resolution = 100
spatial_order = 2
grid_x = finite.UniformPeriodicGrid(resolution, 20)
grid_y = finite.UniformPeriodicGrid(resolution, 20)
domain = finite.Domain((grid_x, grid_y))
x, y = domain.values()

epsilon = 0.5

# IC: Random values of ±1
np.random.seed(1)
IC = np.sign(np.random.rand(domain.shape[0], domain.shape[1])-0.5)
u = IC.copy()


vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.05


# t = 0
plt.subplot(1,5,1)
plt.title("t = 0")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 0.1
while vb_problem.t < 0.1-1e-5:
    vb_problem.step(dt)

plt.subplot(1,5,2)
plt.title("t = 0.1")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 1
while vb_problem.t < 1-10e-5:
    vb_problem.step(dt)

plt.subplot(1,5,3)
plt.title("t = 1")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 5
while vb_problem.t < 5-1e-5:
    vb_problem.step(dt)

plt.subplot(1,5,4)
plt.title("t = 5")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 10
while vb_problem.t < 10-1e-5:
    vb_problem.step(dt)

plt.subplot(1,5,5)
plt.title("t = 10")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()





