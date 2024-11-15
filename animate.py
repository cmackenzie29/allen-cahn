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

# IC: Random
np.random.seed(1)
u = np.sign(np.random.rand(domain.shape[0], domain.shape[1])-0.5)

# IC: Dumbbell
# u = np.zeros(domain.shape)
# for i,ii in enumerate(x[:,0]):
#     for j,jj in enumerate(y[0]):
#         u[i,j] = ((ii-4)**2+(jj-10)**2<=3) or ((ii-16)**2+(jj-10)**2<=3) or ((ii>5 and ii<15) and (jj<=11) and (jj>=9))

# IC: Torus
# u = np.zeros(domain.shape)
# for i,ii in enumerate(x[:,0]):
#     for j,jj in enumerate(y[0]):
#         u[i,j] = ((ii-10)**2+(jj-10)**2<=50) and ((ii-10)**2+(jj-10)**2>=5)

# IC: Maze
# u = np.zeros(domain.shape)
# for i,ii in enumerate(x[:,0]):
#     for j,jj in enumerate(y[0]):
#         u[i,j] = ((ii>1 and ii<3) and (jj>1 and jj<19)) or ((ii>3 and ii<17) and (jj>1 and jj<3)) or ((ii>15 and ii<17) and (jj>3 and jj<17)) or ((ii>5 and ii<15) and (jj>15 and jj<17)) or ((ii>5 and ii<7) and (jj>5 and jj<15)) or ((ii>7 and ii<13) and (jj>5 and jj<7)) or ((ii>11 and ii<13) and (jj>7 and jj<11))

#dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)
#dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)
vb_problem = equations.AllenCahn(u,epsilon,spatial_order,domain)#timesteppers.RK22(equations.Advection_System(u,v,dx,dy))

alpha = 0.1
dt = alpha*grid_x.dx

# 2D plots
plt.ion()
u_plt = plt.imshow(u.T, cmap=matplotlib.colors.ListedColormap(['#F87A53', '#36BA98']), origin='lower')#, norm=matplotlib.colors.LogNorm(vmin=0.0001, vmax=1))
plt.colorbar()
plt.draw()

while vb_problem.t < 10-1e-5:
    vb_problem.step(dt)
    u_plt.set_data(u.T)
    plt.pause(.01)

# plt.imshow(u, cmap="bwr")
# plt.colorbar()
# plt.show()



