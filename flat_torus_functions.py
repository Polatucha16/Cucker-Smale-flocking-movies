import numpy as np
from scipy.spatial import KDTree


#### NEW functions for k closest neugbours: ###

def pt_clones(point):
    """ Returns: np.array.shape=(3,2) consisting of
        3 closest to domain (0,1)^2 points equal to 
        an argument *point after identification R^2 / Z^2.
        Parameters
            pt : np.array.shape = (2,) """
    pt = np.divmod(point,1)[1]
    frame_move = pt - ft_closest_corner(pt)
    
    # creation of 4 points square frame 1 x 1
    grid_x, grid_y = np.meshgrid(np.array([0,1]) ,np.array([0,1]))
    grid_pts = np.vstack([grid_x.ravel(),grid_y.ravel()]).T
    
    result = frame_move + grid_pts
    ind_of_pt = np.flatnonzero((result == pt).all(1))[0]
    return np.delete(result, ind_of_pt, axis=0)


def places_of_k_smallest_values(arr, k):
    """ Returns a tuple of raveled indecies pointing to *k smallest
        values of the matrix *arr.
        USE: 
            arr[k_smallest_vals(arr,k)] -
            - returns a vector of k smallest values from arr
        
        Parameters
        ----------
        arr : np.array
        k : int smaller than np.product(arr.shape)
        """
    flat_arr = arr.ravel()
    inds = np.argpartition(flat_arr, k)[:k] 
    return np.unravel_index(inds, arr.shape, order='C')

def find_neighbours(y, k):
    """ This function adapts KDTree for the flat torus.
        For matrix points stored in y[:,:2], *find_neighbours
        returns matricies of distances and indecies of closest neighbours
        Parameters
        ----------
            y: CS_matrix, positions and velocities of points, shape = (N,4) 
            k: int, number of neighbours to find 
            
        Comments:
        1. KDTree is queried for k+1 points because it is evaluated at points used
            to define KDTree this it find there is 0 distance to the nearest point.
            When looking for points from *copies_of_i it would be reasonable to find k
            but then it would not fit for concatenate-ion, thus for this technical reason
            we add one more point.
        2.  The original KDTree.query when applied on points used to define KDTree returns two arrays such that
            first columns are 0,0,0,... and 0,1,2,... respectivley (query find distances to itself) - call it property (*) 
            The the inner working uses function: places_of_k_smallest_values on joined
            distances. Appended distnces are from points outside the square (0,1)^2 
            thus do not have zeros. In result places_of_k_smallest_values always finds zero from the original vector
            of distnces.
            Function places_of_k_smallest_values uses numpy.argpartition which do not keep order therefore 
            we need to sort again for point that are close to edge to keep property (*) same as original KDTree.
        """
    
    pts = np.divmod(y[:,:2],1)[1] 
    dists, neigh = KDTree(pts).query(pts, k=k+1, eps=0, p=2, distance_upper_bound = np.inf, workers=4)
    
    for i, dist_vec in enumerate(dists.copy()):
        # In KDTree.query.d (distances) are sorted thus we only check if last distance is bigger that distances to edge
        corner = ft_closest_corner(pts[i])
        dist_to_edge = min(abs(corner - pts[i])) # lol... bylo max <- debil? XD 
        if dist_to_edge < dist_vec[-1]:
            # maybe on the other side there are closer points? Let ask KDTree from copies of the pts[i]
            # dists_TW, ind_TW are distances "Through Walls" and indecies "Through Walls"
            copies_of_i = pt_clones(pts[i])
            dists_TW, ind_TW = KDTree(pts).query(copies_of_i, k+1, eps=0, p=2, distance_upper_bound = np.inf, workers=1)
            dist_join, ind_join = np.concatenate((dist_vec.reshape(1,-1), dists_TW)), np.concatenate((neigh[i].reshape(1,-1), ind_TW))
            # From joined dists that do not cross wall and dists from copies we find k smallest
            places = places_of_k_smallest_values(dist_join, k+1)
            
            # argpartition (used in places_of_k_smallest_values) do not keep order, lets get back order in dists.
            # final data (not sorted) are:
            dist_closest = dist_join[places]
            neigh_closest = ind_join[places]
            
            ind_sorted = np.argsort(dist_closest) # find the permutaion that sort dists
            dists[i], neigh[i] = dist_closest[ind_sorted], neigh_closest[ind_sorted]
            
            # dist_sorted  = dist_join[places]
            # dists[i], neigh[i] = dist_join[places], ind_join[places]
            
    return dists, neigh

### Flat Torus metric functions ###
def ft_dist( u_arg, v_arg ):
    """ Distance on Flat Torus (0,1)^n
        Arg: u_arg, v_arg are np.arrays of the length n 
        
        Plot this function:
            resol = 200
            center = np.array([75,30])/resol
            a = np.zeros((resol,resol))
            for i,row in enumerate(a):
                for j,el in enumerate(row):
                    a[i,j] = ft_fun.ft_dist([i/resol,j/resol], center )
            plt.subplots(1,1, figsize=(5, 5))
            plt.imshow(a, cmap='coolwarm')#
            plt.show()
        """
    # first lets return to (0,1)^n cube
    u=np.divmod(u_arg,1)[1]
    v=np.divmod(v_arg,1)[1]
    h = np.zeros(u.shape)
    for i, (u_i, v_i) in enumerate(zip(u, v)):
        dist = np.abs(u_i - v_i)
        if dist > 0.5:
            h[i]=(1 - dist)
        else:
            h[i]=dist
    return np.linalg.norm(h)

def do_segment_cross_domain_boundary(x, y):
    """ Return bool 
        True if the shortest path crosses the boundary of domain square
        args: x,y are np,array shape(n,) or shape(n,1)
            have to be in the same square domain.
        """
    return ft_dist(x, y ) != np.linalg.norm(y-x)

def ft_closest_corner(x):
    """ Return the np.array which represents the 
        closest corner to x"""
    return np.divmod(x, 0.5)[0].astype(int)

def intersections_with_grid(x, y):
    """ (Use notation x = (x1,x2) and X,Y for variables on the plane.)
        Vector v =  np.rint(x-y) moves y to one of 8 surrounding
        (to same place after identyfication)
        squares such that the distance from x to y+v is the shortest.
        Find intersection of segment (x , y+v) and integer gridlines.
        Let denote w = (w1,w2) = y+v - x
        Line passing through x and x+v is given by
            EQL: (-w2,w1).(X,Y)=(-w2,w1).(x1,x2) 
        Function c = ft_closest_corner(x) tells us intersection lines
            EQ1: X = c1, EQ2: Y = c2
        Vector v tells us which lines intersect with EQL since
            if v_i !=0  then we are looking for
            intersection of EQ_i=ci and EQL so solve  {EQ_i, EQL}
        """
    # vectors c and v enable us to build equations EQ_i
    c = ft_closest_corner(x)
    v = np.rint(x-y) 
    
    # w_perp is perpendicular to interval (x, y)
    # and dot(w_perp,(X,Y)) = dot(w_perp,x) is equation in X,Y variables
    # for the line passing through the points x and y
    w = y + v - x
    w_perp = w[[1,0]]
    w_perp[0] = -w_perp[0]
    
    inters_points = []
    for i, v_i in enumerate(v):
        if v_i != 0:
            pt = np.zeros(2) # future point of intersection
            
            o_i = np.divmod(i+1,2)[1] # o_i is the other_index
            
            pt[i] = c[i] # here and in next line we solve { EQ_i, EQL }
            pt[o_i] = np.dot(w_perp,x)/w_perp[o_i] + (w[o_i]*c[i]) / w[i]
            
            inters_points.append(pt)
    inters_points = np.asarray(inters_points)
    
    if inters_points.shape == (2, 2):
            # the closer of the two points remains unchanged 
        ind_x_to_y_1st = np.apply_along_axis(np.linalg.norm, 1,inters_points - x ).argmin()
        ind_x_to_y_2nd = np.divmod(ind_x_to_y_1st+1,2)[1] 
            # the second point has to go back to (0,1)**2 domain, we use vector v:
        inters_points[ind_x_to_y_2nd] = inters_points[ind_x_to_y_2nd] - v
        return inters_points
    elif inters_points.shape == (1, 2):
            # one point of intersection, we have to make a copy using vector v
        inters_points = np.append(inters_points,np.array([inters_points[0]-v]),axis=0)
        return inters_points
    return