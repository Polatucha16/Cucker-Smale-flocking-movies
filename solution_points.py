import numpy as np
from tqdm import tqdm
from rhs_functions import ft_k_closest_to_ith

def data_prepare(ft_solution_CS, N, k, frames):
    """ Function that takes OdeSolution instance of CS_system of shape (N,k) and produce:
            CS_matrices of (position and velociy) and matricies of closest neighbours.
        Returns:
                x_s -> shape = (frames,N,4) solution data points
                neighbours -> shape = (frames,N,k). Adresses of closest neighbours:
                    In the i-th row of neighbours[t] we have indecies of k points that are closest to i-th
        Arg:    ft_solution_CS ->  Bunch object, the result of solve_ivp function with .sol method
                N -> number of points
                k -> number of neighbours to look for
                frames - how many data points you want to produce in the interval provided by the solve_ivp
        """
    # ft_solution_CS.t provides time of start and finish
    t_min, t_max = ft_solution_CS.t[0], ft_solution_CS.t[-1]
    t_s = np.linspace(t_min, t_max, num = frames)
    
    # Preallocating matrices since "np.append" copies the data therefore is slower.
    x_s = np.zeros((frames,N,4))
    neighbours = np.zeros((frames,N,k))
    
    # for every frame write data to a (N,4) CS_matrix and
    # at every frame we calculate k closest neighbourhs
    
    for i, time in tqdm(enumerate(t_s), total=len(t_s)): 
        x_s[i] = ft_solution_CS.sol(time).reshape((N,4))
        for point, vect in enumerate(x_s[i]):
            neighbours[i][point] = ft_k_closest_to_ith(x_s[i],point, k)

    """ Change positions of data x_s to (0,1)^n.
        We do not change velocities, thus [:,:2] """ 
    # redundant? should be in main loop above
    for count, value in enumerate(t_s):
        x_s[count][:,:2] = np.divmod(x_s[count][:,:2],1)[1] 

    # """ reshaping neighbours matrix """
    # #neighbours = np.delete(neighbours, 0, axis=0)
    # neighbours = neighbours.reshape((60*speed_multiplier*seconds, N, k))
    return x_s, neighbours
