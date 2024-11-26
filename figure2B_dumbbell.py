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
u *= 2
u -= 1


vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)

dt = 0.05

# t = 0
plt.subplot(1,4,1)
plt.title("t = 0")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t =  1
while vb_problem.t < 1-1e-5:
    vb_problem.step(dt)

plt.subplot(1,4,2)
plt.title("t = 1")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 5
while vb_problem.t < 5-10e-5:
    vb_problem.step(dt)

plt.subplot(1,4,3)
plt.title("t = 5")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])


# t = 10
while vb_problem.t < 10-1e-5:
    vb_problem.step(dt)

plt.subplot(1,4,4)
plt.title("t = 10")
#plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.imshow(u.T, cmap="Spectral", vmin=-1, vmax=1, origin='lower')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()





