import numpy as np
from matplotlib import pyplot as plt

import flat_torus_functions as ft_fun
import rhs_functions as rhs_fun

def just_save(data, pt, nbds=4, img_size=10, **kwargs):
    """ Saves CS_picture form:
            Parameters
            ----------
                data : CS_matrix
                neighbours : matrix of neighbours 
                pt : index of point to draw neighbours from
            **kwargs:
                path : str path to save the image
                nu_name : int number of the frame in the movie
        """
    # Turn off interactive mode otherwise some matplotlib 
    # instances lingers in RAM and we cannot make longer videos
    plt.ioff()
    
    # matplotlib mandatory parts:
    fig, ax = plt.subplots(1,1, figsize=(img_size, img_size))
    ax.set_aspect(aspect = 1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    spec_data = data.copy()
    #back to square(0,1)
    vects = data[:,2:]
    data = np.divmod(data[:,:2],1)[1]
    
    # plot points in *data
    plt.scatter(*data[:,:2].T, s=1+3*img_size)
    
    # draw numbers of points
    num = range(len(data))

    for i, txt in enumerate(num):
        ax.annotate(txt, np.divmod(data[:,:2],1)[1][i])
    
    # plot speeds arrows
    pos_X = data[:,0]
    pos_Y = data[:,1]
    arr_end_X = vects[:,0]#(data[:,0]+vects[:,0])
    arr_end_Y = vects[:,1]#(data[:,1]+vects[:,1])
    ax.quiver(pos_X,pos_Y,arr_end_X,arr_end_Y, angles='xy')
    
    #plot accelaration arrows
    acc_X = data[:,0]
    acc_Y = data[:,1]
    accs = rhs_fun.rhs_d_vel_with_forecast(spec_data, [1]*len(data), rhs_fun.comm_weight, 3, 0.1)[:,2:]
    acc_end_X = accs[:,0]#acc_X + 
    acc_end_Y = accs[:,1]#acc_Y +
    ax.quiver(acc_X,acc_Y,acc_end_X,acc_end_Y, angles='xy', color='g')
    
    
    #find_neighbours returns pt as first element of the list
    ind_to = ft_fun.find_neighbours(data, nbds)[1][pt]#[1:]
    
    # this loop draws *k intervals from *pt to its closest neighbours
    for neigh_pt in data[ind_to]:
        x = data[pt][:2]
        y = neigh_pt[:2]
        bound_pts = ft_fun.intersections_with_grid(x, y)
        if ft_fun.do_segment_cross_domain_boundary(x, y):

                # below is the index of point closer to x -> draw interval from boundary to x
            ind_x_to_y_1st = np.apply_along_axis(np.linalg.norm, 1, bound_pts - x).argmin()
            x_to_y_1st = np.array(bound_pts[ind_x_to_y_1st])
            plt.plot(*np.array([x,x_to_y_1st]).T, marker = ('.'),c='r')

                # below is the index of point closer to y -> -> draw interval from boundary to y
            ind_x_to_y_2nd = np.apply_along_axis(np.linalg.norm, 1, bound_pts - y).argmin()
            x_to_y_2nd = np.array(bound_pts[ind_x_to_y_2nd])
            plt.plot(*np.array([x_to_y_2nd,y]).T, marker = '.',c='r')

        else:
            plt.plot(*np.array([x,y]).T, marker = '.',c='r')
    
    #saving part
    p = kwargs['path']
    frame = kwargs['nu_name']
    plt.savefig(p/str(frame).zfill(6))
    plt.close(fig)
    
    return