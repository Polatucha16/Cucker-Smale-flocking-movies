# math
import numpy as np
from numpy.random import default_rng, SeedSequence
from scipy.integrate import RK45, solve_ivp

# classes are more or less wrappers of functions from those modules:
import rhs_functions as rhsfun
import setup_functions as sf
import solution_points as solpts
import draw_nearest_neighbours as dnn

#miscellaneous
import os
import subprocess
from datetime import datetime
import pathlib 
from tqdm import tqdm



class InitialCondition:
    """ Class that holds basic parameters, state random generator 
        and creates initial condition for Cucker-Smale model"""
    
    number_of_birds: int
    k_closest: int
    seed: int
    rng: np.random._generator.Generator
    
    def __init__(self, number_of_birds, k_closest=3, seed=1234):
        self.N = number_of_birds
        self.k = k_closest
        
        self.seed = seed
        self.rng = default_rng(SeedSequence(seed))
    
    def mass(self, low=0.9, high=1.1):
        return sf.masses(low,high, self.N, self.rng)
    
    def y_init(self, rng, speed=1, newQ=True):
        return sf.init_cond(self.N , self.rng, speed, newQ, seed=self.seed)

class Solve:
    """ class that uses solve_ivp to solve Cucker-Smale equation
        on a given interval"""
    
    y0: np.array
    k_closest: int
    
    t_start: float
    t_stop: float
    
    def __init__(self, y0, k_closest=3, t_start=0, t_stop=1):
        self.y0 = y0
        self.k = k_closest
        self.t_start = t_start
        self.t_stop = t_stop
    
    def CS(self, mass, comm_weight=rhsfun.singular_weight, prediction_parameter=0):
        print('Invoke the solver_ivp.')
        sol = solve_ivp(
            lambda t, y: rhsfun.ft_fun_CS(t, y, mass, comm_weight, self.k, prediction_parameter),
            (self.t_start,self.t_stop),
            self.y0.ravel(),
            method='RK45',
            max_step = 1, 
            dense_output=True, 
            vectorized=True)
        print(sol.message)
        return sol

class DataPrepare:
    """ fps - frames per second
        speed - speed of the video:
            speed=s means there will be 1 sec of video for every s second of solution
            (and 1/s*fps frames per 1 second of simulation)
    """
    number_of_birds: int
    k_closest: int
    fps: int 
    speed: float 
    
    def __init__(self,number_of_birds, k_closest=3, fps=60, speed=1):
        self.k = k_closest
        self.N = number_of_birds
        self.fps = fps
        self.speed = speed
    
    def prepare(self,sol):
        """Arg: sol - result of solve_ivp """
        t_min, t_max = sol.t[0], sol.t[-1]
        frames = int(self.fps*(t_max- t_min)/self.speed)
        
        return solpts.data_prepare(sol, self.N, self.k, frames)

class Draw:
    
    birds: np.ndarray
    neighbours: np.ndarray
    
    def __init__(self, birds, neighbours):
        self.birds = birds
        self.neighbours = neighbours
    
    def draw(self, frame, pt, saveQ = False):
        
        if not saveQ: #(if saveQ = false print just one picture)
            return dnn.draw_with_neighbours(self.birds, self.neighbours, frame, pt)
        
        # we are here only inf saveQ = True
        # we create a folder and save images in folder named after current date nad time
        folder_name = datetime.now().strftime('%Y%m%d%H%M%S')
        p = pathlib.Path.cwd()/'data'/folder_name/'images'
        p.mkdir(parents=True, exist_ok=True)
        
        # here we use draw_with_neighbours with optional arguments
        for i, data_point in tqdm(enumerate(self.birds), total=len(self.birds)):
            dnn.just_save(data_point,self.neighbours[i], pt, path=p, nu_name=i)
        
        # movie making part:
        bwd = os.getcwd() # remember cwd to come back after using command
        os.chdir(p.as_posix()) # change shell position to newly created folder

        """ In order to create a video we can execute a command:

                ffmpeg -framerate 60 -i %06d.png -c:v libx264 -pix_fmt yuv420p out.mp4

            To run this you have to be able to run ffmpeg from shell. Install ffmpeg from:
            https://ffmpeg.org/download.html
            #note: One can build docker image with ffmpeg """

        #glue_comannd = 'ffmpeg -framerate 60 -i %04d.png -c:v libx264 -pix_fmt yuv420p out.mp4'
        command = 'ffmpeg -framerate 60 -i %06d.png -c:v libx264 -pix_fmt yuv420p '
        destiny = ''.join([p.parents[0].as_posix(),r'/out.mp4'])
        glue_comannd = command+destiny
        subprocess.run(glue_comannd.split(" "), shell=True)

        os.chdir(bwd) # return to be ready for the next one 
        
        return p
            
            
            
            
        