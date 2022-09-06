import numpy as np
from numpy.random import default_rng, SeedSequence
#rng = default_rng(SeedSequence(12345))

"""  Global setup: constants, initial condition, communication weight, seed for random generators """

def const_setup(N_val, k_val):
    """ const_setup(22, 3) <- example of use
        Arg: N_val - number of points
             k_val - how many closest points each point follow
    """
    global N,k,rng
    N, k = N_val, k_val
    rng = default_rng(SeedSequence(12345))
    return N,k,rng

""" initial: masses, positions and velocities """

def masses(l, h, n, rng):
    """ masses(0.9,1.1,N,rng) <- example of use
        Arg: l - lowest possible masses of a point
             h - highest possible masses of a point
             n - number of points 
             rng - random generator used
        """
    global m
    m = rng.uniform(low=l, high=h, size=n)
    return m

def init_cond(N, rng, speed=1, newQ=True, seed=1234):
    """ init_cond(1,True) <- example of use
        Returns CS_matrix of initioal conditions
        Arg: speed  - standard deviation of coordinates of initial speeds
             newQ - True if you want to create new 
        """
    if not newQ:
        rng = default_rng(SeedSequence(seed)) # add IF to have the same seed
    
    x_s = rng.uniform(low=0.45, high=0.55, size=N)# x - coordinates of intial positions
    y_s = rng.uniform(low=0.45, high=0.55, size=N)# y - coordinates of intial positions

    vx_s = speed*rng.standard_normal(N) # x - coordinates of intial speeds
    vy_s = speed*rng.standard_normal(N) # y - coordinates of intial speeds

    y_0 = np.c_[x_s,y_s,vx_s,vy_s] #returns a CS_matrix of initioal conditions
    return y_0