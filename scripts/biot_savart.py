'''
    Follow Welling et al. (2020) calculate the magnetic field at Earth's surface.

 Integrate Biot-Savart over:

    1. All currents within the Vlasov domain
    2. Birkeland currents (FACs) in the “gap region” between the MHD inner boundary and the ionosphere, mapped (assuming J \\propto B) along the field lines connecting ionosphere radius R_IONO to coupling radius r_C
    3. Horizontal Ionospheric currents (altitude 100 km)

 The integration is a discrete summation over the 3 domains.

 Integrating the domain #2 will likely require this mesh to be 'refined' in order to resolve the FAC structures
 see the functions refine_mesh() and graded_mesh()

 This script has been tested with the Vlasiator runs EGL, FHA, FIA
 note that the usefulness for EGL is limited because there is no ionosphere (domain #3) for that run 

 To run this script, require access to the data reducer/variable 'vg_J' in the .vlsv  file
 This may be supplied by a .vlsv file's vlsvReader object, see keyword f_J_sidecar

 This script is written for the UH environment. Adapt file paths as needed.

 ###
 
 EXAMPLE CALL (1 thread):
 python biot_savart.py -nproc 1 -task 1 -run FHA

 Example sidecar .vlsv files,  containing ground magnetic field data, can be found at:
    /wrk-vakka/group/spacephysics/vlasiator/3D/{run name}/sidecars/ig_B/

'''

import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import logging

global R_EARTH
R_EARTH = 6.371e6
global R_IONO
R_IONO = R_EARTH + 1e5       # nominal ionosphere altitude 100km (also assumed in Vlasiator)
global mu_0
mu_0 = 4e-7 * np.pi

# Input parameters
import argparse

global ARGS 



def mkdir_path(path):
    '''
        Make a directory from the stem of an input file name (path)
    '''
    filedir_list = path.split('/')
    filedir = path[:-len(filedir_list[-1])]
    if not(os.path.exists(filedir)):
         os.system('mkdir -p {}'.format(filedir))

def cartesian_to_spherical(x, y, z):
    '''
    r > 0
    0 < theta < pi
    -pi < phi < pi
    all are assumed to be numpy arrays of equal dimensions

    returns:  r, theta, phi  [tuple]
    '''
    r = (x**2 + y**2 + z**2)**0.5
    theta = np.arccos( z / r )
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    '''
    r > 0
    0 < theta < pi
    -pi < phi < pi
    all are assumed to be numpy arrays of equal dimensions

    returns:  x, y, z   [tuple]
    '''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def vec_len_2d(arr_2d):
    '''
        Vector length
    '''
    return np.array([np.linalg.norm(arr_2d, axis = 1)] * 3).transpose()

def vec_unit(arr_2d):
    '''
     assume arr_2d is a numpy array with shape [N, 3]. Return unit vectors with same shape
    '''
    return arr_2d / vec_len_2d(arr_2d)


def b_dip_magnitude(theta, r, mag_mom = 8e22):
    '''
    Inputs:
    theta: colatitude [radians]
    r: radial distance [m]
    keyword mag_mom: magnetic dipole moment, default=8e22 [A / m^2], as in EGI, EGL, FIA, FHA runs

    Outputs: Earth's magnetic field magnitude [Tesla]
    '''
    B_magnitude = mag_mom * (mu_0 / (4 * np.pi * r**3)) * np.sqrt((2*np.cos(theta))**2 + (np.sin(theta))**2)
    return B_magnitude


def b_dip_direction(x, y, z, mag_mom_vector = np.array([0., 0., -8e22])):
    '''
    Inputs: cartesian coordinates x,y,z [m]
        keyword mag_mom_vector: Earth's vector magnetic dipole moment
        
    Outputs: dipole magnetic field unit vector
    '''
    B = b_dip(x, y, z, mag_mom_vector = mag_mom_vector)
    return vec_unit(B)

def b_dip(x, y, z, mag_mom_vector = np.array([0., 0., -8e22])):
    '''
    Inputs: cartesian coordinates x,y,z [m]
        keyword mag_mom_vector: Earth's vector magnetic dipole moment

    Outputs: dipole magnetic field
    '''
    N = x.size
    pos_N = np.array([x, y, z]).transpose()    # shape (N, 3)
    m_N = np.array([list(mag_mom_vector)]*N)  # shape (N, 3)
    r_N = vec_len_2d(pos_N)   # radius, shape (N, 3)
    # dipole field:  B(r) = (mu_0 / 4 pi) * (3r (m dot r) / r^5 - m / r^3)
    B = (mu_0 / (4 * np.pi)) * ( ( 3 * pos_N * np.array([np.sum(m_N * pos_N, axis = 1)]*3).transpose() / r_N**5) - m_N / r_N**3 )
    return B


def refine_mesh(x, y, z, dV, n):
    '''
     refine the mesh in 'inner' FAC region  R_IONO < r < r_C,
     used to calculate FAC contribution to Biot-Savart integral

     x, y, z: initial coordinates (1D numpy float arrays), assumed to be at the center of cubic cell
     dV: cell volume 1D array or scalar value
     dx_0: side length of initial mesh (1D numpy array or single scalar value)
     n (int): the factor by which to refine the mesh
    '''
    size = x.size
    xout = np.zeros([size, n**3])
    yout = np.zeros([size, n**3])
    zout = np.zeros([size, n**3])
    dV_out = np.zeros([size, n**3])
    dx_0 = dV**(1. / 3)
    dx = dx_0 / n
    iarr = np.arange(n)
    ind = 0
    for i_x in range(n):
        for i_y in range(n):
            for i_z in range(n):
                xout[:, ind] = x - (dx_0 / 2) + ( (i_x+0.5) * dx)
                yout[:, ind] = y - (dx_0 / 2) + ( (i_y+0.5) * dx)
                zout[:, ind] = z - (dx_0 / 2) + ( (i_z+0.5) * dx)
                dV_out[:, ind] = dV / n**3
                ind += 1
    return xout.flatten(), yout.flatten(), zout.flatten(), dV_out.flatten()

def graded_mesh(x, y, z, dV, ns = np.array([8, 4, 2]), Rs = np.array([R_EARTH, R_EARTH*2, R_EARTH*4])):
    '''
     Iteratively refine the mesh in 'inner' FAC region  R_IONO < r < r_C,
     used to calculate FAC contribution to Biot-Savart integral

    Calls refine mesh with different refinement factors n, at specified refinement boundaries
    ns: numpy array of refinement factors (ints)
    Rs: numpy array of refinement boundary radii [R_E]
    edge case: arrays are aligned so that for r>Rs[-1], use ns[-1] refinement

    Want the refinement level to roughly scale as r^{-3/2}. To resolve the mapped features at all radii
    This works because magnetic dipole field goes as 1/r^3, roughly, and area A goes as r^2. 
    (Need to set areas of the cell faces so B*A~constant, to resolve features mapped along field lines)

    Example: graded_mesh(x, y, z, dV, ns = np.array([8, 4, 2]), Rs = np.array([R_EARTH, R_EARTH*2, R_EARTH*4]))
        --> refines 1D mesh size by factor of 8 for 1 RE<r<=2 RE, by factor of 4 for 2 RE<r<=4 RE, by factor of 2 for r>4 RE
    '''

    x_out = np.array([])
    y_out = np.array([])
    z_out = np.array([])
    dV_out = np.array([])
    r = np.sqrt(x**2 + y**2 + z**2)

    for i in range(ns.size):
        logging.info('i: '+ str(i))
        if i < ns.size - 1:
            ind, = np.where((r > Rs[i]) & (r <= Rs[i+1]))
        elif i == ns.size -1:
            ind, = np.where(r > Rs[i])
        logging.info('ind size: '+ str(ind.size))
        x_ref, y_ref, z_ref, dV_ref = refine_mesh(x[ind], y[ind], z[ind], dV[ind], ns[i])
        x_out = np.concatenate((x_out, x_ref))
        y_out = np.concatenate((y_out, y_ref))
        z_out = np.concatenate((z_out, z_ref))
        dV_out = np.concatenate((dV_out, dV_ref))
    logging.info("graded mesh has" + str(x_out.size) + " elements")
    return x_out, y_out, z_out, dV_out


def nearest_node_index(f, x, y, z, node_coords_iono = None):
    '''
       helper function for finding nearest ionospheric node
    '''
    if node_coords_iono is None:
        node_coords_iono = f.get_ionosphere_node_coords()
        # find the nearest cell and evaluate the current there (brute force)
        # this approach is probably faster: https://github.com/fmihpc/vlasiator/blob/master/sysboundary/ionosphere.cpp#L381
    dist = np.sqrt((x - node_coords_iono[:,0])**2 + (y - node_coords_iono[:,1])**2 + (z - node_coords_iono[:,2])**2)
    ind_min = np.argmin(dist)
    return ind_min


def fac_map(f, vg_x, vg_y, vg_z, dx, f_J_sidecar = None, r_C = 5 * 6.371e6, mag_mom_vector = np.array([0., 0., -8e22])):
    '''
     Map the FACs along magnetic field lines (J \\propto B).

     Inputs:
        f: VlsvReader object
        f_J_sidecar: vlsvReader object that contains pre-computed current 'vg_J'

            ``if f_J_sidecar = None:``
                here is assumed the data reducer 'ig_fac' exists (such as for runs FHA and FIA)
                In this case, the currents in the FAC region (domain #2) will be mapped UP from the ionosphere 'ig\\_' grid
            otherwise (f_J_sidecar = a \\*.vlsv string, for a file containing the current density J in the vlasov 'vg\\_' grid)
                for run EGL, see files at /wrk-vakka/group/spacephysics/vlasiator/3D/EGL/visualizations_2/ballooning/\\*.vlsv
                In this case, the currents in the FAC region (domain #2) will be mapped DOWN from the vg\\_ grid

        vg_x,vg_y,vg_z position [m], 1D numpy arrays. 
            note: these coordinates do not have to correspond with Vlasiator's vg\\_ grid,
            This is relevant when fac_map() is called by biot_savart(), with keyword mesh='refined' or mesh='graded'

        dx is grid resolution [m], 1D numpy array, This is for the input cells which can be defined arbitrarily---not necessarily same as vg cells.

     ***  coordinates, not just coordinates on the vg_ grid ***

    Returns:
        the vector current density [A/m^2] at specified positions vg_x, vg_y, vg_z position
        output J=0 for cells that fall outside the FAC region (i.e if cell is not in region R_EARTH+dx/2 < r< r_C)
    '''
    cellids = f.read_variable('CellID')
    vg_b_vol = b_dip(vg_x, vg_y, vg_z, mag_mom_vector = mag_mom_vector)

    vg_r, vg_theta, vg_phi = cartesian_to_spherical(vg_x, vg_y, vg_z)
    inner, = np.where((vg_r < r_C) & (vg_r > (R_EARTH + dx/2)))  # only integrate over cells whose centers are at least 1/2 a cell length beyond radius of 1 R_E

    vg_J_eval = np.zeros([vg_x.size, 3])

    vg_lat = (np.pi / 2) - vg_theta         # -pi/2 < lat < pi/2
    L = vg_r /  np.cos(vg_lat)**2           # L-shell (dipole) [m]

    # evaluate FACs in 'inner' FAC region:  R_IONO < r < r_C
    if f_J_sidecar is None:
        # map: initial point -> downmap to ionosphere via dipole formula (ionospheric runs, e.g. FHA)
        node_coords_iono = f.get_ionosphere_node_coords()  # [n_nodes, 3]
        ig_fac = f.read_variable('ig_fac')      # [n_nodes], facs evaluated on ionosphere grid (assumed r=R_IONO)
        vg_b_vol_magnitude = np.sqrt(vg_b_vol[:,0]**2 + vg_b_vol[:,1]**2 + vg_b_vol[:,2]**2 )

        lat0 = np.arccos( np.sqrt(R_IONO / L) ) # latitude at r=R_IONO
        theta0 = (np.pi / 2) - lat0
        b0 = b_dip_magnitude(theta0, R_IONO, mag_mom = 8e22)
        x0, y0, z0 = spherical_to_cartesian(R_IONO, theta0[inner], vg_phi[inner])
        for i in range(x0.size):
            ind_min = nearest_node_index(f, x0[i], y0[i], z0[i], node_coords_iono = node_coords_iono)
            vg_J_eval[inner[i], :] = (vg_b_vol[inner[i],:] / b0[inner[i]]) * ig_fac[ind_min]  # J \propto B. Mapping UP from the FACs evaluated at the ground 
    else: # (use sidecar containing current density "vg_J" in non-ionospheric runs, e.g. EGL)
        # map: initial point -> some point in the simulation domain near the inner boundary (~5 R_E) according to dipole formula
        logging.info('NOTE: Downmapping FACs along constant L-shell via dipole formula!')
        r_up = r_C
        lat_up = np.arccos( np.sqrt(r_up / L) ) # latitude at r=r_up
        theta_up = (np.pi / 2) - lat_up
        x_up, y_up, z_up = spherical_to_cartesian(r_up, theta_up[inner], vg_phi[inner])
        ind_fin, = np.where(np.isfinite(lat_up[inner]))
        coords_temp = np.array([x_up[ind_fin],y_up[ind_fin],z_up[ind_fin]]).T.reshape([ind_fin.size,3])

        vg_b_vol_fin = f_J_sidecar.read_interpolated_variable("vg_b_vol", coords_temp)

        B_up = vec_len_2d(vg_b_vol_fin)
        B_down = np.array([b_dip_magnitude(vg_theta[inner[ind_fin]], vg_r[inner[ind_fin]], mag_mom = 8e22)] * 3).transpose()
        scale_factor = B_down / B_up    # J \propto B

        vg_J = f_J_sidecar.read_interpolated_variable("vg_J", coords_temp)

        J_signed_up = np.array([np.sum(vg_J * vg_b_vol_fin, axis = 1) ] * 3).transpose() / B_up    # magnitude and sign of J   (projection J dot B / |B|)
        b_dir = b_dip_direction(vg_x[inner[ind_fin]], vg_y[inner[ind_fin]], vg_z[inner[ind_fin]])
        vg_J_eval[inner[ind_fin], :] = b_dir * J_signed_up * scale_factor   # Mapping DOWN from the FACs evaluated in the simulation domain near inner boundary

    # only allow currents that can map to the inner boundary (this also avoids numerical artifacts at equator)
    ind_to_zero, = np.where(L < r_C) 
    vg_J_eval[ind_to_zero, :] = 0.
    return vg_J_eval


def biot_savart(coord_list, f, f_J_sidecar = None, r_C = 5 * 6.371e6, mesh = 'graded'):
    '''
    param coord_list:   a list of 3-element arrays of coordinates [ [x1,y1,z1], [x2,y2,z2], ... ], SI units
                        if considering just a single starting point, the code accepts a 3-element array-like object [x1,y1,z1]

    f: vlsvReader object
    f_J_sidecar: vlsvReader object that contains pre-computed current 'vg_J'

        e.g., for EGL, files at: /wrk-vakka/group/spacephysics/vlasiator/3D/EGL/visualizations_2/ballooning/\\*.vlsv

    runtime (FHA): overhead of about 200 sec (setup), plus 0.2 sec for each element of coord_list

    returns a tuple (B_inner, B_outer) 
    B_inner: the B-field [T] generated by FACs at 1 R_E < r< r_C
    B_outer: the B-field [T] generated by currents in simulation domain at r > r_C
    '''

    # standardize input (a list of 3-element arrays/lists)
    if type(coord_list[0]) not in [list, np.ndarray]:
        coord_list = [coord_list]

    ncoords = len(coord_list)

    # load vg_ coordinates and compute cell volumes (dV)
    vg_b_vol = f.read_variable('vg_b_vol')
    coords = f.read_variable("vg_coordinates")
    vg_x, vg_y, vg_z = np.array(coords)[:,0], np.array(coords)[:,1], np.array(coords)[:,2]

    cellids = f.read_variable('CellID')
    ncells = cellids.size
    dx = ((f.read_parameter('xmax') - f.read_parameter('xmin')) / f.read_parameter('xcells_ini')) # refinement lvl 0 (coarsest)
    dV = np.zeros(ncells)
    for i, cellid in enumerate(cellids):
        dV[i] = dx**3 / 2**(3*f.get_amr_level(cellid))

    # load or calculate currents (from B-field Jacobian)
    if f_J_sidecar is None:
        vg_J = f.read_variable('vg_J')
    else:
        vg_J = f_J_sidecar.read_variable('vg_J')

    # compute B at 'coord_list' points according to Biot-Savart law
    vg_r, vg_theta, vg_phi = cartesian_to_spherical(vg_x, vg_y, vg_z)
    ind_r = np.argmin(vg_r)
    dx_1re = dx / 2**np.nanmin(f.get_amr_level(cellids[ind_r]))   # grid resolution dx at radius 1 RE (well inside inner boundary)
    outer, = np.where(vg_r >= r_C)

    vg_J_eval = np.zeros(vg_J.shape) # the currents that will actually be integrated
    vg_J_eval[outer] = vg_J[outer] 

    # evaluate contribution of 'inner' FAC region:  R_IONO < r < r_C
    if mesh == 'graded':
        # assume B~r^-3 (dipole). Use resolution required to resolve FAC structures mapped to at 5 RE. That is, cell size~r^(3/2).
        ns = np.array([16, 8, 4, 2])
        Rs = np.array([R_EARTH, R_EARTH*1.3,  R_EARTH*2, R_EARTH*3.2])  # adjust these refinement boundaries to taste
        inner, = np.where((vg_r < r_C) & (vg_r > (R_EARTH - dx_1re/2)))
        x_inner_ref, y_inner_ref, z_inner_ref, dV_inner_ref = graded_mesh(vg_x[inner], vg_y[inner], vg_z[inner], dV[inner], ns = ns, Rs = Rs)
        vg_J_eval_inner_ref = fac_map(f, x_inner_ref, y_inner_ref, z_inner_ref, dV_inner_ref**(1. / 3),
                                      f_J_sidecar = f_J_sidecar, r_C = r_C,  mag_mom_vector = np.array([0., 0., -8e22]))
    elif mesh == 'refined':
        n_fine = 8
        inner, = np.where((vg_r < r_C) & (vg_r > (R_EARTH - dx_1re/2)))
        x_inner_ref, y_inner_ref, z_inner_ref, dV_inner_ref = refine_mesh(vg_x[inner], vg_y[inner], vg_z[inner], dV[inner], n_fine)  # 2000km / n_fine mesh
        vg_J_eval_inner_ref = fac_map(f, x_inner_ref, y_inner_ref, z_inner_ref, dV_inner_ref**(1. / 3),
                                      f_J_sidecar = f_J_sidecar, r_C = r_C, mag_mom_vector = np.array([0., 0., -8e22]))
    else:  # no refinement
        inner, = np.where((vg_r < r_C) & (vg_r > R_EARTH))
        vg_J_eval[inner] = fac_map(f, vg_x[inner], vg_y[inner], vg_z[inner], dV[inner]**(1. / 3),
                                   f_J_sidecar = f_J_sidecar, r_C = r_C, mag_mom_vector = np.array([0., 0., -8e22]))

    B_outer = integrate_biot_savart(coord_list, vg_x[outer], vg_y[outer], vg_z[outer], np.transpose(vg_J_eval[outer]).copy(order='C'), dV[outer])  # alternative, no @jit decorator

    if mesh == 'graded' or mesh == 'refined':
        B_inner = integrate_biot_savart(coord_list, x_inner_ref, y_inner_ref, z_inner_ref, np.transpose(vg_J_eval_inner_ref).copy(order='C'), dV_inner_ref)
    else: # no refinement
        B_inner = integrate_biot_savart(coord_list, vg_x[inner], vg_y[inner], vg_z[inner], np.transpose(vg_J_eval[inner]).copy(order='C'), dV[inner])
    
    return B_inner, B_outer


@jit(nopython=True, fastmath=True)
def integrate_biot_savart(coord_list, x, y, z, J, delta):
    '''
     integration of the Biot-savart law
     magnetic field is evaluated at coordinates specified in coord_list (for example, the ionospheric coordinates)

     Inputs:
        x,y,z (1D array, size n): Cartesian coordinates
        delta (1D array, size n): the volume or area of the element being integrated
        J (2D array, size [3, n]): current density

      all units SI
     Outputs:
      magnetic field evaluated at coord_list

     Note: the units of J and delta depend on the type of integral (volume or surface), but the equation form is unchanged

     Biot-Savart (volume): B = (mu_0 / 4 * pi) \\int { J x r' / \\|r'\\|^3 } dV  ([J] = A/m^2, delta == dV)
                (surface): B = (mu_0 / 4 * pi) \\int { J x r' / \\|r'\\|^3 } dA  ([J] = A/m, delta = dS)
    '''
    B = np.zeros((len(coord_list), 3))
    r_p = np.zeros((3, x.size))

    J_cross_r_p = np.zeros((3, x.size))
    for i, coord in enumerate(coord_list):
        r_p[0,:] = coord[0] - x
        r_p[1,:] = coord[1] - y
        r_p[2,:] = coord[2] - z

        r_p_mag = np.sqrt(r_p[0,:]**2 + r_p[1,:]**2 + r_p[2,:]**2)
        J_cross_r_p[0,:] = J[1,:] * r_p[2,:] - J[2,:] * r_p[1,:]   # can't use np.cross with jit
        J_cross_r_p[1,:] = J[2,:] * r_p[0,:] - J[0,:] * r_p[2,:]
        J_cross_r_p[2,:] = J[0,:] * r_p[1,:] - J[1,:] * r_p[0,:]

        repeated_terms = (mu_0 / (4 * np.pi)) * delta / (r_p_mag*r_p_mag*r_p_mag)    # supposedly, x*x*x faster than x**3
        B[i,0] += np.nansum( repeated_terms * J_cross_r_p[0,:] )  
        B[i,1] += np.nansum( repeated_terms * J_cross_r_p[1,:] )
        B[i,2] += np.nansum( repeated_terms * J_cross_r_p[2,:] )

    return B


def B_ionosphere(f, coord_list = None, ig_r = None, method = 'integrate'):
    '''
     Ionospheric (domain #3) current contribution to magnetic field

     B_ionosphere() evaluates the magnetic field produced by ionospheric currents

     Inputs:
        f: VlsvReader object

        keyword coord_list: locations where B-field is calculated

        keyword ig_r: the ionospheric mesh

        keyword method:

            ='integrate' (default): integrate over the whole ionosphere using Biot-Savart law 

            ='local': ionosphere produces magnetic field locally, as by an infinite plane of current overhead

     Outputs:
        Magnetic field evaluated at coord_list
    '''
    if ig_r is None:
        ig_r = f.get_ionosphere_element_coords()
    if coord_list is None:
        coord_list = list(ig_r * R_EARTH / R_IONO)  # Default: Rescale ionospheric mesh (radius ~R_IONO) to a smaller grid at radius R_EARTH
    dummy = np.array(coord_list)*0. 
    try:
        ig_inplanecurrent = f.read_variable('ig_inplanecurrent')
        if method == "local":
            # assume local horizontal appear as an infinite plane, to a ground observer looking up
            # B = (mu_0 / 2) * r_hat x J_s , where J_s vector is current per unit length
            ig_r_hat = vec_unit(ig_r)   # approximate (technically |ig_r| not exactly R_IONO)
            if coord_list is not None:
                logging.info('infinite plane approximation not yet implemented for input coord_list!')
                return dummy
            else:
                B_iono = (mu_0 / 2) * np.cross(ig_r_hat, ig_inplanecurrent)
        elif method == 'integrate':
            # integrate Biot-Savart law over ionospheric mesh. More accurate but slower.
            dS = f.get_ionosphere_mesh_area()
            B_iono = integrate_biot_savart(coord_list, ig_r[:, 0], ig_r[:, 1], ig_r[:, 2], np.transpose(ig_inplanecurrent).copy(order='C'), dS)
        return B_iono
    except:
        return dummy  # no ionospheric inplanecurrent data

def B_magnetosphere(f, f_J_sidecar = None, r_C = 5 * 6.371e6, ig_r = None):
    '''
        Inner and outer magnetospheric contributions to Biot-Savart integral (domains #1, #2)
        wrapper for biot_savart()
    '''
    if ig_r is None:
        ig_r = f.get_ionosphere_element_coords()
    B_inner, B_outer = biot_savart( list(ig_r * R_EARTH / R_IONO), f, f_J_sidecar = f_J_sidecar, r_C = r_C, mesh = 'graded' )
    return B_inner, B_outer

def save_B_vlsv(input_tuple):
    '''
        calculate magnetic field at the Earth's surface and save in a .vslv file

        Inputs: input tuple
            input_tuple[0]: run (string)  # 'EGL', 'FHA', or 'FIA'
            input_tuple[1]: fileIndex (int)

        Outputs:
            ig_r: Cartesian ionospheric grid locations (radius R_EARTH + 100km)
                note that B_iono, B_inner, B_outer are in fact evaluated at radius R_EARTH
                ig_r is a copy of the Vlasiator ionosphere mesh, to enable combination with other data reducers

            B_iono: Ionospheric (Domain #3) contribution to ground magnetic field (radius R_EARTH)
            B_inner: Ionospheric (Domain #2) contribution to ground magnetic field (radius R_EARTH)
            B_outer: Ionospheric (Domain #1) contribution to ground magnetic field (radius R_EARTH)

        Note: the input and output .vlsv file paths may need to be modified in this script for different users
    '''
    # get_vlsvfile_fullpath(), and helper functions get_filename(), get_bulklocation()
    # compute the file names based on the name of the run ('EGL', 'FHA', 'FIA') and the time step fileIndex [s]
    def get_vlsvfile_fullpath(run, fileIndex):
        '''
            Returns full path of a .vlsv file, based on the run name and time step (fileIndex)
        '''
        return get_bulklocation(run) + get_filename(run, fileIndex)
    def get_filename(run, fileIndex):
        if run.upper() == 'EGL':
            filename = "bulk1.{}.{}.vlsv".format(run.lower(), str(fileIndex).zfill(7) )
        elif run.upper() == 'FHA':
            filename = "bulk1.{}.vlsv".format(str(fileIndex).zfill(7) )
        elif run.upper() == 'FIA':
            filename = "bulk1.{}.vlsv".format(str(fileIndex).zfill(7) )
        return filename
    def get_bulklocation(run):
        if run.upper() == 'EGL':
            location = "/wrk-vakka/group/spacephysics/vlasiator/3D/EGL/bulk/"
        elif run.upper() == 'FHA':
            location = "/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/"
        elif run.upper() == 'FIA':
            location = "/wrk-vakka/group/spacephysics/vlasiator/3D/FIA/bulk/"
        return location
    # instantiate VlsvWriter object
    run, fileIndex = input_tuple
    filename = get_vlsvfile_fullpath( run, fileIndex)
    f = pt.vlsvfile.VlsvReader( filename )      # f contains the vg_ mesh over which Biot-Savart is integrated
    if run == 'EGL':
        f_J_sidecar = pt.vlsvfile.VlsvReader('/wrk-vakka/group/spacephysics/vlasiator/3D/EGL/visualizations_2/ballooning/jlsidecar_bulk1.egl.{}.vlsv'.format(str(fileIndex).zfill(7)))
        f_iono = pt.vlsvfile.VlsvReader( '/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/misc_sidecars/ionogrid_FHA.vlsv' )
    elif run == 'FHA':
        f_J_sidecar = None
        f_iono = f
    elif run == 'FIA':
        f_J_sidecar = None
        f_iono = f
    save_dir = './'   # USER-DEFINED PATH
    # calculate magnetic fields
    ig_r = f_iono.get_ionosphere_element_coords()        # f_iono contains the ionospheric mesh (the locations where B is evaluated)
    B_iono = B_ionosphere(f, ig_r = ig_r, method = "integrate")
    try:    # FHA, FIA
        r_C = float(f.get_config()['ionosphere']['downmapRadius'][0]) * R_EARTH
    except: # EGL
        r_C = 5 * 6.371e6
    B_inner, B_outer = B_magnetosphere(f, f_J_sidecar = f_J_sidecar, r_C = r_C, ig_r = ig_r)
    # write to file
    filename_vlsv = save_dir + 'ionosphere_B_sidecar_{}.{}.vlsv'.format(run, str(fileIndex).zfill(7))
    mkdir_path(filename_vlsv)
    writer = pt.vlsvfile.VlsvWriter(f_iono, filename_vlsv, copy_meshes=("ionosphere"))
    writer.write(ig_r,'ig_r','VARIABLE','ionosphere')
    writer.write(B_iono,'ig_b_ionosphere','VARIABLE','ionosphere')
    writer.write(B_inner,'ig_b_inner','VARIABLE','ionosphere')
    writer.write(B_outer,'ig_b_outer','VARIABLE','ionosphere')
    return ig_r, B_iono, B_inner, B_outer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-run', default='FHA', help="run name" )
    parser.add_argument('-task', default=0, help="task no." )
    parser.add_argument('-nproc', default=1, help="number of processors to use " )
    ARGS = parser.parse_args()
    
    run = ARGS.run
    if run == 'EGL':
        first = 621
        last = 1760
    elif run == 'FHA':
        first = 501  # 501, 800
        last = 1612
    elif run == 'FIA':
        first = 1
        last = 865

    '''      
    # (TEST) Single file: integrate Biot-Savart and save output into a .vlsv sidecar file
    ig_r, B_iono, B_inner, B_outer = save_B_vlsv(('FHA', 1000))
    '''

    # integrate Biot-Savart and save output into .vlsv files (modify biot_savart.sh to use multiple nodes)
    from multiprocessing import Pool
    pool = Pool(int(ARGS.nproc))
    start = first + (int(ARGS.task) * int(ARGS.nproc))
    stop = start + int(ARGS.nproc)
    logging.info('start:, ' + str(start) + ', stop: ' + str(stop))
    input_list = [(run, i) for i in range(start, stop)]
    f_out = pool.map(save_B_vlsv, input_list)
    pool.close()
    pool.join()
    
