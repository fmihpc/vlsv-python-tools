import numpy as np
from scipy.constants import k, m_e, m_p
import analysator as pt
import warnings
import logging

import random

def epsilon_M(f,cell,pop="proton",m=m_p, bulk=None, B=None,
                model="bimaxwellian",
                normorder=1, norm=2, threshold=0,
                dummy=None):

    ''' Calculates the 'non-maxwellianity' parameter for a distribution function f_i and its corresponding Maxwellian g_M.
    
    :param f:           VlsvReader containing VDF data
    :param cell:        CellID for the queried cell

    :kword pop:         Population to calculate the parameter for
    :kword m:           Species mass (default: m_p)
    :kword bulk:        Bulk file name to use for moments (if available and needed)
    :kword B:           Optional, user-given magnetic field vector for a bimaxwellian model distribution
    :kword model:       VDF model to be used. Available models "maxwellian", "bimaxwellian" (default)
    :kword normorder:   Norm used for model-data distance measure (default: 1)
    :kword norm:        Constant norm (default 2, see below)
    :kword threshold:   Disregard vspace cells under this threshold [0]
    :kword dummy:       If not None, generate dummy data for e.g. integration.

    :returns:           scalar, non-Maxwellianity parameter for given model and norm

    The definition is given by Graham et al. (2021):
    (1/2n) * integral(\\|f_i - g_M\\|) d^3v    
    
    valued between 0 (bi-Maxwellian) and 1 (complete deviation from a bi-Maxwellian).
    
    NOTE that this is different to the definition by Greco et al. (2012):
    (1/n) * sqrt(integral((f_i - g_M)^2) d^3v)    
    
    examples comparing these two definitions can be found in Settino et al. (2021).



    
    '''
    if dummy is not None:
        warnings.warn("Generating dummy value for non-Maxwellianity!")
        return random.random()
    # try getting the distribution from the given cell
    size = f.get_velocity_mesh_size(pop)
    distribution = np.zeros(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))
    try:
        D = f.read_velocity_cells(cell, pop)
        D_keys = list(D.keys())
        D_vals = list(D.values())
        distribution[D_keys] = D_vals
        distribution[distribution<threshold] = 0
    except:
        warnings.warn("Could not get VDF from bulk file!")
        return -1
     # calculate non-maxwellianity

    # names of velocity moments in the bulk file
    n_name = pop+'/vg_rho'         
    v_name = pop+'/vg_v'
    v_para_name = pop+'/vg_v_parallel'
    v_perp_name = pop+'/vg_v_perpendicular'
    T_name = pop+'/vg_temperature'
    T_perp_name = pop+'/vg_t_perpendicular'
    T_para_name = pop+'/vg_t_parallel'

    dV = np.prod(f.get_velocity_mesh_dv(pop))
    # generate a velocity space 
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))
    v = f.get_velocity_cell_coordinates(vids, pop)
    vs = f.get_velocity_cell_coordinates(vids[D_keys],pop)

    if B is None:
        try:
            logging.info("No B vector given, trying a restart reducer from " + f.file_name)
            B = f.read_variable("B",cellids=cell)
        except:
            logging.info("No restart B found, trying for vg_b_vol from " + f.file_name)
            try:
                B = f.read_variable("vg_b_vol",cellids=cell)
            except:
                logging.info("No vg_b_vol available either")
        if bulk is not None and B is None:
            try:
                logging.info("Trying to load vg_b_vol from given bulk file "+bulk)
                bulkfile_for_moments_reader=pt.vlsvfile.VlsvReader(bulk)
                B = bulkfile_for_moments_reader.read_variable("vg_b_vol",cellids=cell)
            except:
                logging.info("Could not load vg_b_vol from bulk file "+bulk)
        if B is None:
            warnings.warn("No B found, cannot proceed at cellid "+str(cell))
            return -1

    if bulk is not None:
        try:
            # get velocity moments from bulk file
            bulkfile_for_moments_reader=pt.vlsvfile.VlsvReader(bulk)
            n = bulkfile_for_moments_reader.read_variable(n_name,cellids=cell)
            v0 = bulkfile_for_moments_reader.read_variable(v_name,cellids=cell)
            v0_para = bulkfile_for_moments_reader.read_variable(v_para_name,cellids=cell)
            v0_perp = bulkfile_for_moments_reader.read_variable(v_perp_name,cellids=cell)

            T = bulkfile_for_moments_reader.read_variable(T_name,cellids=cell)
            T_para = bulkfile_for_moments_reader.read_variable(T_para_name,cellids=cell)
            T_perp = bulkfile_for_moments_reader.read_variable(T_perp_name,cellids=cell)
            # Generate a parallel-perpedicular coordinate basis and corresponding velocity array
            bhat = B/np.linalg.norm(B)
            vperp2hat = np.cross(bhat,v0)/np.linalg.norm(np.cross(bhat,v0))
            vperp1hat = np.cross(vperp2hat, bhat)/np.linalg.norm(np.cross(vperp2hat,bhat))

            calc_moments = False

        except:
            logging.info("Could not get moments from bulk file " + bulk + ". Calculating them from the VDF..")
            calc_moments = True
    else:
        calc_moments = True

    # calculate moments from VDF if bulk is None, or moments could not be fetched from the bulk file
    if calc_moments:
        n = np.sum(np.array(D_vals))*dV
        v0 = np.average(vs, axis=0,weights=np.array(D_vals)*dV)
        # Generate a parallel-perpedicular coordinate basis and corresponding velocity array
        bhat = B/np.linalg.norm(B)
        vperp2hat = np.cross(bhat,v0)/np.linalg.norm(np.cross(bhat,v0))
        vperp1hat = np.cross(vperp2hat, bhat)/np.linalg.norm(np.cross(vperp2hat,bhat))
        
        R = np.array([bhat, vperp1hat, vperp2hat])
        vsb = np.matmul(R,vs.T).T
        vsb_mean = np.matmul(R,v0.T).T
        v0_para = vsb_mean[0]
        v0_perp = vsb_mean[1]

        P_diag = m * np.sum((vsb - vsb_mean) * (vsb - vsb_mean) * np.array(D_vals)[:,np.newaxis],axis=0) * dV
        T = np.sum(P_diag) / (3.0 * n * k)
        T_para = P_diag[0] / (n * k)
        T_perp = (P_diag[1] + P_diag[2]) / (2.0 * n * k)
        
    R = np.array([bhat, vperp1hat, vperp2hat])
    vb_mean = np.matmul(R,v0.T).T
    vb = np.matmul(R,v.T).T
    
    vT_para = np.sqrt(2 * k * T_para / m)
     
    # generate a maxwellian distribution
    if model == "maxwellian":
        v_sq = np.sum((v - v0) * (v - v0), axis=-1)
        model_f = n * (0.5 * m / (np.pi * k * T)) ** 1.5 * np.exp(-0.5 * m * v_sq / (k * T))
    elif model == "bimaxwellian": # Graham et al 2021
        model_f = n * T_para / (np.pi**1.5 * vT_para**3 * T_perp) * np.exp(
        -(vb[:,0] - v0_para)**2/vT_para**2
        -((vb[:,1] - v0_perp)**2 + vb[:,2]**2)/(vT_para**2 * (T_perp/T_para))
        )
    else:
        raise NameError("Unknown VDF model '"+model+"', aborting")

    epsilon = np.linalg.norm(distribution - model_f, ord=normorder)
    epsilon *= dV / (norm * n)

    return epsilon

