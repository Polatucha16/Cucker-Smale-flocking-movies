import numpy as np
from numpy.random import default_rng, SeedSequence


def init_cond(setup, t0=0, duration=1, newQ=False):
    """ Creates initial conditions from user defined functions
        Parameters
        ----------
            setup - Setup object
        Returns
        ------
            y0 : CS_matric - initial conditions
            t0 : initial time
            duration : how long simulate
            m0 : vector of initial masses """
    y0 = set_random_y0(setup, newQ=newQ)
    m0 = set_random_masses(setup, newQ=newQ)
    return y0, t0, duration, m0


### initial condition functions ###
def set_random_y0(setup, newQ=False, low=0, high=1, velo_st_dev=4):
    """From data in setup and **kwargs produce initial CS_matrix"""
    rng = new_random_or_seed(setup, newQ=newQ)
    
    x_s = rng.uniform(low=low, high=high, size=setup.N) # x coordinates of intial position
    y_s = rng.uniform(low=low, high=high, size=setup.N) # y coordinates of intial position

    vx_s = velo_st_dev*rng.standard_normal(setup.N) # x coordinates of intial velocity
    vy_s = velo_st_dev*rng.standard_normal(setup.N) # y coordinates of intial velocity

    y_0 = np.c_[x_s, y_s, vx_s, vy_s] #returns a CS_matrix - initial condition
    return y_0

# def some_other_user_defined_y0(setup, newQ, **kwargs):
#     """From data in setup and **kwargs produce initial CS_matrix"""
#     return y_0


### initial masses ###
def set_random_masses(setup, newQ=False, low=1, high=1):
    """ set_random_masses(setup, True, low=0.5, high=1.5) <- example of use
        Arg: low - lowest possible masses of a point
             high - highest possible masses of a point
        """
    rng = new_random_or_seed(setup, newQ=newQ)
    m = rng.uniform(low=low, high=high, size=setup.N)
    return m

# def some_user_defined_masses(setup, newQ):
#     return m


### UTILITIES ###
def new_random_or_seed(setup, newQ=False):
    """ Function which tell which rng to use"""
    if not newQ:
        return default_rng(SeedSequence(setup.seed))
    return setup.rng