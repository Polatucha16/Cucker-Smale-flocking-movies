import numpy as np
from matplotlib import pyplot as plt

import rhs_functions as rhsfun


def do_segment_cross_domain_boundary(x, y):
    """ Return bool 
        True if the shortest path crosses the boundary of domain square
        args: x,y are np,array shape(n,) or shape(n,1)
            have to be in the same square domain.
        """
    return rhsfun.ft_dist(x, y ) != np.linalg.norm(y-x)

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

# def draw_with_neighbours(data, neighbours, frame, pt: int): # y: CS_matrix
#     """ needs update  """
    
#     fig, ax = plt.subplots(1,1,figsize=(10, 10))
#     ax.set_aspect(aspect = 1)
#     plt.xlim(0,1)
#     plt.ylim(0,1)
    
#     ind_to = neighbours[frame][pt].astype(int)
#     for vec in data[frame,ind_to]:
#         x = data[frame][pt][:2]
#         y = vec[:2]
#         bound_pts = intersections_with_grid(x, y)
#         if do_segment_cross_domain_boundary(x, y):

#                 # below is the index of point closer to x
#             ind_x_to_y_1st = np.apply_along_axis(np.linalg.norm, 1, bound_pts - x).argmin()
#             x_to_y_1st = np.array(bound_pts[ind_x_to_y_1st])
#             plt.plot(*np.array([x,x_to_y_1st]).T, marker = ('.'),c='r')

#                 # below is the index of point closer to y
#             ind_x_to_y_2nd = np.apply_along_axis(np.linalg.norm, 1, bound_pts - y).argmin()
#             x_to_y_2nd = np.array(bound_pts[ind_x_to_y_2nd])
#             plt.plot(*np.array([x_to_y_2nd,y]).T, marker = '.',c='r')

#         else:
#             plt.plot(*np.array([data[frame][pt][:2],vec[:2]]).T, marker = '.',c='r')

#     #reszta punktow
#     pos_t = []  
#     for i, bird in enumerate(data):#[:20]):
#         pos_t.append(bird[:2])
#     pos_t = np.array(pos_t).T
#     plt.scatter(*pos_t,s=5)
    
#     if  kwargs != {}:
#         p = kwargs['path']
#         plt.savefig(p/str(frame).zfill(4))
#         plt.close(fig)
#         return
#     else:
#         plt.show()
#     return

def just_save(data, neighbours, pt, **kwargs):
    """ Saves CS_picture form:
                data : CS_matrix
                neighbours : matrix of neighbours 
                pt : index of point to draw neighbours from
            **kwargs:
                path : str path to save the image
                nu_name : int number of the frame in the movie
        """
    
    fig, ax = plt.subplots(1,1,figsize=(10, 10))
    ax.set_aspect(aspect = 1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    # red neighbours
    ind_to = neighbours[pt].astype(int)
    for vec in data[ind_to]:
        x = data[pt][:2]
        y = vec[:2]
        bound_pts = intersections_with_grid(x, y)
        if do_segment_cross_domain_boundary(x, y):

                # below is the index of point closer to x -> draw interval from boundary to x
            ind_x_to_y_1st = np.apply_along_axis(np.linalg.norm, 1, bound_pts - x).argmin()
            x_to_y_1st = np.array(bound_pts[ind_x_to_y_1st])
            plt.plot(*np.array([x,x_to_y_1st]).T, marker = ('.'),c='r')

                # below is the index of point closer to y -> -> draw interval from boundary to y
            ind_x_to_y_2nd = np.apply_along_axis(np.linalg.norm, 1, bound_pts - y).argmin()
            x_to_y_2nd = np.array(bound_pts[ind_x_to_y_2nd])
            plt.plot(*np.array([x_to_y_2nd,y]).T, marker = '.',c='r')

        else:
            plt.plot(*np.array([data[pt][:2],vec[:2]]).T, marker = '.',c='r')
    
    pos_t = np.zeros((len(data),2))  
    for i, bird in enumerate(data):#[:20]):
        pos_t[i] = bird[:2]
    pos_t = np.array(pos_t).T
    plt.scatter(*pos_t, s=5)
    
    p = kwargs['path']
    frame = kwargs['nu_name']
    plt.savefig(p/str(frame).zfill(6))
    plt.close(fig)
    
    return