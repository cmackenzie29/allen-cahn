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

# IC: Random
np.random.seed(20)
IC = np.sign(np.random.rand(domain.shape[0], domain.shape[1])-0.5)

dt = 0.1
u = IC.copy()

plt.subplot(3,4,5)
plt.title("I.C.")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

epsilon = 0.1
u = IC.copy()
vb_problem_1 = equations.AllenCahn(u,epsilon,spatial_order,domain)

while vb_problem_1.t < 0.1-1e-5:
    vb_problem_1.step(dt)

plt.subplot(3,4,2)
plt.title("ε = 0.1")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem_1.t < 1-1e-5:
    vb_problem_1.step(dt)

plt.subplot(3,4,6)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem_1.t < 5-1e-5:
    vb_problem_1.step(dt)

plt.subplot(3,4,10)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


epsilon = 0.2
u = IC.copy()
vb_problem_2 = equations.AllenCahn(u,epsilon,spatial_order,domain)

while vb_problem_2.t < 0.1-1e-5:
    vb_problem_2.step(dt)

plt.subplot(3,4,3)
plt.title("ε = 0.2")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem_2.t < 1-1e-5:
    vb_problem_2.step(dt)

plt.subplot(3,4,7)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem_2.t < 5-1e-5:
    vb_problem_2.step(dt)

plt.subplot(3,4,11)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


epsilon = 0.5
u = IC.copy()
vb_problem_3 = equations.AllenCahn(u,epsilon,spatial_order,domain)

while vb_problem_3.t < 0.1-1e-5:
    vb_problem_3.step(dt)

plt.subplot(3,4,4)
plt.title("ε = 0.5")
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem_3.t < 1-1e-5:
    vb_problem_3.step(dt)

plt.subplot(3,4,8)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

while vb_problem_3.t < 5-1e-5:
    vb_problem_3.step(dt)

plt.subplot(3,4,12)
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

plt.show()



