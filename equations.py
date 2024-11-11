from timesteppers import StateVector, CrankNicolson, AdamsBashforth, RK22
from scipy import sparse
import numpy as np
import finite

class AllenCahn:
    
    def __init__(self, u, epsilon, spatial_order, domain):
        self.u = u
        self.epsilon2 = epsilon**2
        self.dx2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        self.dy2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)
        
        self.t = 0
        self.iter = 0
        self.ts_A = Diffusion2D(u, self.epsilon2, self.dx2, self.dy2)
        self.ts_B = AdamsBashforth(ReactionAC(u), 2)

    def step(self, dt):
        self.t += dt
        self.iter += 1
        self.ts_B.step(dt/2)
        self.ts_A.step(dt)
        self.ts_B.step(dt/2)

class ReactionAC:

    def __init__(self, c):
        self.X = StateVector([c])
        N = c.shape[0]
        self.M = sparse.eye(N, N)
        self.L = sparse.csr_matrix((N, N))
        self.F = lambda X: -X.data + X.data**3


class Diffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.c = c
        diffx = Diffusion(c, D, dx2, 0)
        diffy = Diffusion(c, D, dy2, 1)
        self.ts_x = CrankNicolson(diffx, 0)
        self.ts_y = CrankNicolson(diffy, 1)

    def step(self, dt):
        self.ts_y.step(dt/2)
        self.ts_x.step(dt)
        self.ts_y.step(dt/2)      

class Diffusion:
    
    def __init__(self, c, D, d2, axis):
        self.X = StateVector([c], axis=axis)
        N = c.shape[axis]
        self.M = sparse.eye(N, N)
        self.L = -D*d2.matrix


