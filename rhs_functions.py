import numpy as np
import flat_torus_functions as ft_fun

def comm_weight(s):
    """Communication weight in Cucker-Smale model"""
    return np.float_power(s, -0.8, out=np.zeros_like(s), where=s!=0)

def ft_fun_CS(t, y, m, comm_weight, k, prediction_parameter):
    """ solve_ivp operate on a vector (.shape is (N*4,)) of the form:
        (*)    (bird1_x, bird1_y, bird1_dx, bird1_dy, 
                bird2_x, bird2_y, bird2_dx, bird2_dy,
                ...
                birdN_x, birdN_y, birdN_dx, birdN_dy)
        This function is a argument for solve_ivp named
            fun : callable
        Return RHS of CS eq. in the form accepted by solve_ivp (*)"""
    N = len(m)
    y = y.reshape((N,4))
    x_derivatives = rhs_d_pos(y)
    y_derivatives = rhs_d_vel_with_forecast(y, m, comm_weight, k, prediction_parameter)
    rhs = x_derivatives + y_derivatives
    return rhs.ravel()

def rhs_d_pos(y):
    """ return CS_matrix of derivaties of positions in CS model, that is: dx_i = v_i"""
    u = zero_pos(y.copy())
    return u[:,[2,3,0,1]]

def rhs_d_vel_with_forecast(y, mass, weight, k, h):
    """ Returns CS_matrix of derivaties of velocities in CS_system with forecast h at the point y.
        Symbolically with v_i = (birdi_dx, birdi_dy), i \in range(N) the system is:
            dv_i = \sum_{j\in neigh(i)}  { m_j*(v_j-v_i) * \phi(dist(x_j + h*v_j, x_i + h*v_i)) }
            
        Parameters
        ----------
                y :      CS_data_point of shape (N,k);
                mass :   shape (N,) of masses of points 
                weight : callable 'communication weight' funtion
                k :      int how many closest birds every bird looks at
                h :      float how far every bird thinks it goes strait and forecasts accordingly 
    """
    N = len(mass)              # number of points
    vel = zero_pos(y.copy())   # velocities only
    result = np.zeros(y.shape) # place for the result
    
    # fast KDTtree method for finding neighbours
    dists, neigh = ft_fun.find_neighbours(y, k)
    
    y_forecast = y.copy()
    # y_forecast[:,[0,1]] = np.divmod(y_forecast[:,:2],1)[1] 
    y_forecast[:,[0,1]] =  y_forecast[:,[0,1]] + h*y_forecast[:,[2,3]]
    # y_forecast[:,[0,1]] = np.divmod(y_forecast[:,:2],1)[1] 
    pos_forecast = zero_vel(y_forecast.copy())
    
    for i, row in enumerate(vel):
        # Below  we find *i-th row of RHS for velocities
        # First we read from *neigh which points are *k closest to *i-th.
        # Load from the second element [1:], beacuse search method finds the *i-th point itself.
        index = neigh[i]#[1:]
        
        # for j in *index array *vel_diff has v_j-v_i
        # for j in *index array *mass_vel has  m_j*(v_j-v_i)
        vel_diff = substract_ith_at_index(vel, i, index)
        mass_vel = np.matmul(np.diag(mass), vel_diff)
        
        
        # for j in *index array *dists_to_i_forecast has ft_dist( (x_j+h*v_j), (x_i+ h*v_i)) 
        # for j in *index array, (j,j) place in *weighted_dists_forecast has weight( dist((x_j+h*v_j), (x_i+ h*v_i)) )
        dists_to_i_forecast = np.zeros(N)
        for v in index:
            dists_to_i_forecast[v] = ft_fun.ft_dist( pos_forecast[v], pos_forecast[i] )
        weighted_dists_forecast = np.diag(weight(dists_to_i_forecast))
        
        #for j in *index array vector (j,:) of *mass_vel_weight_forecast is  m_j*(v_j-v_i)* weight( dist((x_j+h*v_j), (x_i+ h*v_i)) )
        mass_vel_weight_forecast = np.matmul(weighted_dists_forecast, mass_vel)
        
        # in CS model derivative of velocity is the sum of the above "infuences" from birds at index thus:
        result[i] = np.sum(mass_vel_weight_forecast, axis=0)
    return result


### auxiliary functions:

def zero_pos(y):
    """ Returns y with the first int(half of columns) set to zero.
        Parameters
        ----------
            arg: y is CS_matrix of shape (N, 2*dim) 
            returns: CS_matrix with all the *positions* set to zero"""
    num_of_points, dimention = y.shape
    dimention = int(dimention/2)    #From now on (num_of_points,dimention) is the shape of subarray of positions
    
    ind = list(range(dimention))
    y[:,ind] = np.zeros((num_of_points,dimention))
    return y

def zero_vel(y):
    """ Returns y with the second int(half of columns) set to zero.
        Parameters
        ----------
            arg: y is CS_matrix of shape (N, 2*dim) 
            returns: CS_matrix with all the *velocities* set to zero"""
    num_of_points, dimention = y.shape
    dimention = int(dimention/2)     # From now on (num_of_points,dimention) is the shape of subarray of velocities

    ind = list(range(2*dimention))[-dimention:]
    y[:,ind] = np.zeros((num_of_points,dimention))
    return y

def substract_ith_at_index(arr, i, index):
    """ Returns array_result which at index has difference
        between: (row at index) and (i-ith row).
        Rows not in index are zeros.
        E.g.: for i=5 and for index = array([i_1,i_2,...])
            array_result[i_j] = arr[i_j] - arr[5]
        Parameters
        ----------
              arr : np.array (n,m)
                i : int and element of range(n)
            index : np.array and as elements subset of range(n)
        """
    temp = np.zeros(arr.shape)
    temp[index] = arr[index]
    
    temp_i = np.zeros(arr.shape)
    temp_i[index] = np.array([arr[i] for no, el in enumerate(index)])
    return temp-temp_i
    