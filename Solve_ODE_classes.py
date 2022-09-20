import numpy as np
from numpy.random import default_rng, SeedSequence
from scipy.integrate import RK45, solve_ivp

import initial_conditions as ic
import rhs_functions as rhsfun
import draw_save as draw_save

# draw_and_save needs:
import os
import subprocess
from datetime import datetime
import pathlib 
from tqdm import tqdm


class Const:
    """ Class that holds constants
        Parameters
        ----------
        N: int -------- number of birds
        k: int -------- number of neighbours that influence every bird
        fps: int ------ frames per second in output video
        speed: float -- video speed, smaller - slower moving birds (no change in behaviour)
        """
    N: int       
    k: int       
    fps: int     
    speed: float 
    
    def __init__(self, N, k, fps=30, speed=1):
        self.N, self.k = N, k
        self.fps, self.speed  = fps, speed
    
class Setup(Const):
    """ Class that sets up initial condition for the solver 
        Parameters
        ----------
        const : Const ----- source of constant parameters
        seed : int -------- seed for random generator innitiated in object 
        t0 : float -------- starting time of solution
        duration : float -- time of the simulation
        newQ : bool ------- answer to question: "New initial condition?"
                            If False then rng is reset to see
        
        Example
        -------
            After creating instance you can change initial condition with *make_ic method.
                const = scls.Const(N=5, k=4, fps=30, speed=1)
                setup = scls.Setup(const, seed=164534, t0=0, duration=1, newQ=True)
            
                print(setup.__dict__)
                solve.make_ic(t0=2, duration=7, newQ=False)
                print(setup.__dict__)
        """
    seed : int
    rng: np.random._generator.Generator
    
    y0: np.array
    t0: float
    duration: float
    m: np.array
    
    def __init__(self, const, seed=1234, t0=0, duration=1, newQ=False):
        super().__init__(const.N, const.k, const.fps, const.speed)
        self.seed, self.newQ, self.rng = seed, newQ, default_rng(SeedSequence(seed))
        self.t0, self.duration = t0, duration
        # self.make_ic(t0=t0, duration=duration, newQ=newQ)
        
        self.const_parent = const
    
    def make_ic(self, t0=None, duration=None, newQ=None):
        """ Method that produce initial conditions and set them to 
            Setup object properties"""
        # **kwargs better ?
        if t0 is None:
            t0 = self.t0
        if duration is None:
            duration = self.duration
        if newQ is None:
            newQ = self.newQ  
        self.y0, self.t0, self.duration, self.m = ic.init_cond(self, t0=t0, duration=duration, newQ=newQ)
        return self
    

class Solve(Setup):
    """ Class that with CS method calls solve_ivp with parameters from Const and Setup object.
        Check data with: solve.__dict__
        Example
        -------
        Solving equations:
            const = scls.Const(N=5, k=4, fps=30, speed=1)
            setup = scls.Setup(const, seed=164534, t0=0, duration=1, newQ=True)
            solve = scls.Solve(const, setup)
            ode_bunch, pts = solve.CS(comm_weight=comm_weight, prediction_parameter=0.15)
        """
    
    def __init__(self, setup):
        super().__init__(setup.const_parent, setup.seed, setup.t0, setup.duration, setup.newQ)
        self.y0, self.m = setup.y0, setup.m
        self.setup_parent = setup
    
    def CS(self, comm_weight=rhsfun.comm_weight, prediction_parameter=0):
        print('Invoke the solver_ivp.')
        sol = solve_ivp(
            lambda t, y: rhsfun.ft_fun_CS(t, y, self.m, rhsfun.comm_weight, self.k, prediction_parameter = prediction_parameter),
            (self.t0, self.t0+self.duration),
            self.y0.ravel(),
            method='RK45',
            # method='LSODA',
            t_eval = np.linspace(self.t0, self.t0+self.duration, int(self.duration*self.fps/self.speed)),
            max_step = 1, 
            dense_output=False, 
            vectorized=False)
        print(sol.message)
        # with sol.y you obtain (len(y),time) data
        return sol, sol.y.T.reshape(-1,self.N,4)
    
    # def YOCDE(self,param):
    #     """your other collective dynamis equation you want to solve"""

class Draw(Solve):
    """ Class that draw pictures of solutions from solve"""
    
    def __init__(self, solve):
        super().__init__(solve.setup_parent)
        self.solve_parent = solve
    
    def draw_and_save(self, pts, pt):
        """Save picures and video of the solution stored in pts
            Parameters
            ----------
            pts : numpy.ndarray - Array of the .shape=(frames,self.N,4)
                    that stores solution points to draw. Every row (axis=0)
                    is an array (N,4) which keep positions and velocities of
                    points solved by solve_ivp.
            """
        
        # Create a folder and save images in folder named after current date nad time
        folder_name = datetime.now().strftime('%Y%m%d%H%M%S')
        p = pathlib.Path.cwd()/'data'/folder_name/'images'
        p.mkdir(parents=True, exist_ok=True)
        
        for i, data_point in tqdm(enumerate(pts), total=len(pts)):
            draw_save.just_save(data_point, pt, nbds=self.k, img_size=5, path=p, nu_name=i)
            # draw_save.just_save(data_point,self.neighbours[i], pt, path=p, nu_name=i)
        
        # movie making part:
        bwd = os.getcwd() # remember cwd to come back after using command
        os.chdir(p.as_posix()) # change shell position to newly created folder

        """ In order to create a video we can execute a command:

                ffmpeg -framerate 60 -i %06d.png -c:v libx264 -pix_fmt yuv420p out.mp4

            To run this you have to be able to run ffmpeg from shell. Install ffmpeg from:
            https://ffmpeg.org/download.html
            #note: One can build docker image with ffmpeg """

        #glue_comannd = 'ffmpeg -framerate 60 -i %04d.png -c:v libx264 -pix_fmt yuv420p out.mp4'
        command = 'ffmpeg -framerate 30 -i %06d.png -c:v libx264 -pix_fmt yuv420p '
        destiny = ''.join([p.parents[0].as_posix(),r'/out.mp4'])
        glue_comannd = command+destiny
        subprocess.run(glue_comannd.split(" "), shell=True)

        os.chdir(bwd) # return to be ready for the next one 
        return p