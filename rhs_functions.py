import numpy as np

""" 1. Matrix shortcuts functions """

def zero_pos(y):
    """ Use: zero_pos(y.copy())
        Changes y to have zeros in place of positions
            arg: y is CS_matrix of shape (num_of_points,2*dimention) 
            returns: CS_matrix with all the positions set to zero"""
    num_of_points, dimention = y.shape
    dimention = int(dimention/2)    #From now on (num_of_points,dimention) is the shape of subarray of positions
    
    ind = list(range(dimention))
    y[:,ind] = np.zeros((num_of_points,dimention))
    return y

def zero_vel(y):
    """ Use: zero_vel(y.copy())
        Changes y to have zeros in place of velocities
            arg: y is CS_matrix of shape (num_of_points,2*dimention) 
            returns: CS_matrix with all the velocities set to zero"""
    num_of_points, dimention = y.shape
    dimention = int(dimention/2)     # From now on (num_of_points,dimention) is the shape of subarray of velocities

    ind = list(range(2*dimention))[-dimention:]
    y[:,ind] = np.zeros((num_of_points,dimention))
    return y
    

"""Cucker-Smale RHS on flat torus funtions """

def ft_dist( u_arg, v_arg ):
    """ Distance on Flat Torus (0,1)^n
        Arg: u,v are vectors of the lenght n 
        u,v can be anywhere
        """
    # first lets return to (0,1)^n cube
    u=np.divmod(u_arg,1)[1]
    v=np.divmod(v_arg,1)[1]
    h = []
    for u_i, v_i in zip(u, v):
        dist = np.abs(u_i - v_i)
        if dist > 0.5:
            h.append(1 - dist)
        else:
            h.append(dist)
    
    return np.linalg.norm(h)
    
    
def ft_dists_to_ith(y, ith):
    """ Arg: CS_matrix y   -  CS_matrix of our data
             int ith -  number from range(y.shape[0]) row to measure distance from
        returns: the vector of ft_dist from ith to all other points *keeping the order* """
    u = zero_vel(y.copy())   # positions of points 
    ith_vect = u[ith]        # position which we going to meausure distance to    
    dist_to_ith = []         # placeholder for distances
    
    for vect in u:           # remember values of ft_dist's in a list
        dist_to_ith.append(ft_dist(vect,ith_vect)) #<= slow change to preallocation of vector
    return np.array(dist_to_ith)
    
    
def ft_k_closest_to_ith(y, ith, k_closest):
    """ Arg: CS_matrix y   -  CS_matrix of our data
             int ith -  number from range(y.shape[0]) row to measure distance from
             int k_closest - number of closest points to find
        Returns: list of indecies of rows of CS_matrix of k_closest elements to ith """
    
    dist_to_ith = ft_dists_to_ith(y, ith)
        
    # below we find indices of rows of k closest points (those are k+1 smallest number since there is 0) 
    # remove index of ith point itself (don't know the order thats why np.setdiff1d )
    ind_k_closest = np.argpartition(dist_to_ith,k_closest+1)[:k_closest+1]
    ind_k_closest = np.setdiff1d(ind_k_closest,np.array([ith]))
    
    return ind_k_closest


def rhs_d_pos(y):
    """ return CS_matrix of derivaties of positions in CS model, that is: dx_i = v_i"""
    u = zero_pos(y.copy())
    u = u[:,[2,3,0,1]]
    return u


def rhs_d_vel(y, mass, weight, k_closest):
    """ return CS_matrix of derivaties of velocities in CS model, that is: dV_i = sum_j m_j*(v_j-v_i)*\phi(dist(x_j,x_i))"""
    
    vel = zero_pos(y.copy())   # velocities only
    result = np.zeros(y.shape)
    
    for i, row in enumerate(vel):
        # here we find i-th row of rhs, so first which velocities we include:
        index = ft_k_closest_to_ith(y, i, k_closest)
        
        temp_i = np.zeros(y.shape)
        temp_i[index] = np.array([row for _ in range(k_closest)]) # velocity of ith point (written in row) 
        
        temp = np.zeros(y.shape)
        temp[index] = vel[index] # velocities of points given by index
        mass_vel = np.matmul(np.diag(mass),temp-temp_i) # at index places we have  m_j*(v_j-v_i) the rest is 0.
        
        dist_to_ith = ft_dists_to_ith(y, i) 
        weighted_dists = np.diag(weight(dist_to_ith)) # diagonal matrix with \phi(dist(x_j-x_i)) at (j,j) place
        mass_vel_weight = np.matmul(np.diag(weight(dist_to_ith)),mass_vel)
        # at this point mass_vel_weight at *index places have m_j*(v_j-v_i) * \phi(dist(x_j-x_i))
        result[i] = np.sum(mass_vel_weight,axis=0)
    
    return result


def rhs_d_vel_with_forecast(y, mass, weight, k_closest, h):
    """ return CS_matrix of derivaties of velocities in CS_system with forecast h at the point y
        System is: dV_i = sum_j m_j*(v_j-v_i)*\phi(dist(x_j+h*v_j,x_i+h*v_i))
        Arg:    y - CS_data_point of shape (N,k);
                mass - shape (N,) of masses of points 
                weight - callable 'communication weight' funtion
                k_closest - int how many closest birds every bird looks at
                h - float how far every bird thinks it goes strait and forecasts accordingly 
        Return: result - the derivatives velocities RHS of CS_system with forecast h at the point y
    """
    vel = zero_pos(y.copy())   # velocities only
    result = np.zeros(y.shape)
    
    for i, row in enumerate(vel):
        # here we find *i-th row of rhs, so first which points we include:
        index = ft_k_closest_to_ith(y, i, k_closest)
        
        # *temp is zero matrix except closest to *i-th points where it is equal to *vel
        temp = np.zeros(y.shape)
        temp[index] = vel[index]
        
        # create a matrix such that at places pointed by *index, there are copies of *row
        temp_i = np.zeros(y.shape)
        temp_i[index] = np.array([row for _ in range(k_closest)]) # velocity of ith point (current variable: row) 
        
        # at *index places *mass_vel will have  m_j*(v_j-v_i) the rest is 0.
        mass_vel = np.matmul(np.diag(mass),temp-temp_i)
        
        # here we calculate the argument of *weight function, thus we create matrix where positions
        # are altered to x_j + h*v_j. The rest of rhs will be the same as rhs without forecast 
        y_forecast = y.copy()
        y_forecast[:,[0,1]] =  y_forecast[:,[0,1]] + h*y_forecast[:,[2,3]]
        
        # *dist_to_ith will store values of *weight function at *y_forecast positions (j-th place = ft_dist(x_j+v_j-(x_i + v_i)))
        dist_to_ith = ft_dists_to_ith(y_forecast, i)
        
        # *weighted_dists is a matrix with \phi(ft_dist(x_j+v_j-(x_i + v_i)))) at (j,j) place.
        
        # *mass_vel_weight is a np.array (N,4) which at *index places have:
        #  m_j*(v_j-v_i) * \phi(ft_dist(x_j+v_j-(x_i + v_i))))
        weighted_dists = np.diag(weight(dist_to_ith))
        mass_vel_weight = np.matmul(weighted_dists,mass_vel)
        
        # in CS model derivative of velocity is the sum of the above "infuences" from other birds thus:
        result[i] = np.sum(mass_vel_weight,axis=0)
    
    return result

""" Fianl functions that go to solve_ivp """
def singular_weight(s):
    """Communication weight in Cucker-Smale model"""
    return np.float_power(s,-0.3, out=np.zeros_like(s), where=s!=0)


def ft_fun_CS(t, y, m, comm_weight, k, prediction_parameter = 0):
    """ solve_ivp operate on a vector (N*4,) of the form:
        (*)    (a1,a2,va1,va2,b1,b2,vb1,vb2, ...)
        Function prepares data for matrix multiplication,
        multiply data by apropiate matricies to obtain RHS.
        Return RHS of CS eq. in the form accepted by solve_ivp (*) """
    N = len(m)
    y = y.reshape((N,4))
    x_derivatives = rhs_d_pos(y)
    if prediction_parameter == 0:
        y_derivatives = rhs_d_vel(y, m, comm_weight, k)
    else:
        y_derivatives = rhs_d_vel_with_forecast(y, m, comm_weight, k, prediction_parameter)

    rhs = x_derivatives + y_derivatives
    return rhs.ravel()
        