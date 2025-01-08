'''
 Given a batch of .vlsv files, containing ground magnetic field vector data at different times,
 calculate the north & east components of induced ground electric field [V/m], and save in corresponding .vlsv files

 This script has been tested with the Vlasiator runs EGL, FHA, FIA
 note that the usefulness for EGL is limited because there is no ionosphere (domain #3) for that run 

 Note that the Geomagnetically Induced Currents (GICs) can be computed immediately from the geoelectric field:
 The surface current density is J = sigma*E, where E, sigma are respectively the ground electric field and conductivity.

 This script is written for the UH environment. Adapt file paths as needed.

 ###
 
 EXAMPLE CALL:
    python gics.py

 Example sidecar .vlsv files,  containing ground magnetic field data (over the ionosphere mesh), can be found at:
    /wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1_sidecars/geoelectric_field/

'''

import analysator as pt
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
    

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

def cartesian_to_spherical_vector(vx, vy, vz, x, y, z):
    '''
    Convert cartesian vector(s) with coordinates (vx, vy, vz)
    at the position(s) theta, phi (note: position r does not affect the vector transformation)
    to spherical coordinates (v_r, v_theta, v_phi)

    dimensions of vx, vy, vz, x, y, z arrays must either match or be a single number
    '''
    r, theta, phi = cartesian_to_spherical(x, y, z)
    v_r =     vx * np.sin(theta) * np.cos(phi) + vy * np.sin(theta) * np.sin(phi) + vz * np.cos(theta)
    v_theta = vx * np.cos(theta) * np.cos(phi) + vy * np.cos(theta) * np.sin(phi) - vz * np.sin(theta)
    v_phi =  -vx * np.sin(phi) + vy * np.cos(phi)
    return v_r, v_theta, v_phi

def mkdir_path(path):
    '''
        Make a directory from the stem of an input file name (path)
    '''
    filedir_list = path.split('/')
    filedir = path[:-len(filedir_list[-1])]
    if not(os.path.exists(filedir)):
         os.system('mkdir -p {}'.format(filedir))

def E_horizontal(dB_dt, pos, time, sigma = 1e-3, method = 'liu'):
    '''
        Calculate the horizontal electric field by integrating components of dB/dt
            References: Cagniard et al 1952 (eq. 12), Pulkinnen et al 2006 (eq. 19)
        Inputs:
            dB_dt: cartesian dB/dt    [T/s] array dimension [3, len(time)]
            pos: cartesian position [m] 1D 3-element array, vector position
            time: 1D array of times [s], monotonically increasing

        Keywords:
            sigma = ground conductivity (siemens/meter)
            method:
            
                'liu': use integration method described in Liu et al., (2009) doi:10.1029/2008SW000439, 2009
                       this method is exact for piecewise linear B (i.e., piecewise constant dB/dt)

                'RH-riemann': use right-handed Riemann sum.
    '''
    mu_0 = 1.25663706e-6    # permeability of free space
    E_north = np.zeros(time.size)
    E_east = np.zeros(time.size)
    dB_dt_r, dB_dt_theta, dB_dt_phi = cartesian_to_spherical_vector(dB_dt[0,:], dB_dt[1,:], dB_dt[2,:], pos[0], pos[1], pos[2])
    dB_dt_north = -dB_dt_theta; dB_dt_east = dB_dt_phi
    t0 = time[1] - time[0]
    for i in range(0, time.size):
        t = time[i]   # monotonically increasing from t[0]
        tp = time[1:i+1]
        dt = tp - time[0:i]
        if method == 'liu':  # implement Liu et al (2009), eq. 5
            t_1 = t - tp
            t_2 = t - time[0:i]   # elementwise, t_2 > t_1
            dB_north = dB_dt_north[0:i] * dt[0:i]
            dB_east = dB_dt_east[0:i] * dt[0:i]
            E_north[i] = np.sum(-(2. / np.sqrt(np.pi * mu_0 * sigma * dt[0:i])) * dB_east * (np.sqrt(t_2) - np.sqrt(t_1) ) )
            E_east[i] = np.sum((2. / np.sqrt(np.pi * mu_0 * sigma * dt[0:i])) * dB_north * (np.sqrt(t_2) - np.sqrt(t_1) ) )
        elif method == 'RH-riemann':
            if i != 0:
                E_north[i] = -(1. / np.sqrt(np.pi * mu_0 * sigma)) * np.sum(dt * dB_dt_east[1:i+1] / np.sqrt(t-tp + t0))
                E_east[i] = (1. / np.sqrt(np.pi * mu_0 * sigma)) * np.sum(dt * dB_dt_north[1:i+1] / np.sqrt(t-tp + t0))  # note the sign
    return E_north, E_east

if __name__ == '__main__':    
    '''
    Before running cell blocks below, requires running biot_savart.py 
    to generate total ground magnetic field sidecar .vlsv files
    '''
    
    R_EARTH = 6371000.

    # user-defined locations of sidecar .vlsv files
    run = "FHA"  # EGL, FHA, FIA
    if run == "FIA":
        dir = "/wrk-vakka/group/spacephysics/vlasiator/3D/FIA/bulk_sidecars/ig_B/"
    elif run == "FHA":
        dir = "/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1_sidecars/ig_B/"
    elif run == "EGL":
        dir = "/wrk-vakka/group/spacephysics/vlasiator/3D/EGL/sidecars/ig_B/"
    
    if run == "FHA":
        nmin = 501    # 501-1000 using _v2 sidecars
        nmax = 1612 
    elif run == "FIA":
        nmin = 1
        nmax = 817         # max 865 (files 818-819 missing)
    elif run == "EGL":
        nmin = 621
        nmax = 1760
    time = np.linspace(nmin, nmax, nmax - nmin + 1)

    # load example file to initialize arrays with
    f = pt.vlsvfile.VlsvReader("/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1_sidecars/ig_B/ionosphere_B_sidecar_FHA.0000784.vlsv")     # FHA: file indices 501 - 1254
    pos = f.read_variable('ig_r')   # ionospheric grid.  array dimensions (43132, 3)
    ig_dB_dt_arr = np.ndarray([pos.shape[0], pos.shape[1], nmax - nmin + 1])
    ig_B_ionosphere_arr = ig_dB_dt_arr * 0.
    ig_dB_dt_ionosphere_arr = ig_dB_dt_arr * 0.
    ig_B_inner_arr = ig_dB_dt_arr * 0.
    ig_dB_dt_inner_arr = ig_dB_dt_arr * 0.
    ig_B_outer_arr = ig_dB_dt_arr * 0.
    ig_dB_dt_outer_arr = ig_dB_dt_arr * 0.
    
    E_north_arr = np.ndarray([pos.shape[0], nmax - nmin + 1])
    E_east_arr = np.ndarray([pos.shape[0], nmax - nmin + 1])
    
    # populate B arrays (read sidecar files)
    for i in range(nmin, nmax+1):
        f = pt.vlsvfile.VlsvReader(dir + "ionosphere_B_sidecar_{}.{}.vlsv".format(run, str(i).zfill(7)))
        try:
            ig_B_ionosphere = f.read_variable('ig_B_ionosphere')
            ig_B_ionosphere_arr[:,:,i-nmin] = ig_B_ionosphere
        except:
            logging.info("couldn't read ionospheric data") # for runs without an ionosphere, leave as zeros
        ig_B_inner = f.read_variable('ig_b_inner')
        ig_B_inner_arr[:,:,i-nmin] = ig_B_inner
        ig_B_outer = f.read_variable('ig_b_outer')
        ig_B_outer_arr[:,:,i-nmin] = ig_B_outer

    # interpolate across any zeros in B arrays (klug) 
    for arr in [ig_B_ionosphere_arr]:
        try:
            ind = np.where(arr[0,0,:] != 0)[0]
            logging.info("Time interpolation: {} points removed".format(arr.shape[2] - ind.size))
            interp_arr = arr[:,:, ind]  # only keep the non-zero times to conduct the interpolation
            for i in range(arr.shape[0]): # positions
                for j in range(3): # vector components
                    arr[i, j, :] = np.interp(time, time[ind], interp_arr[i, j, :], left=None, right=None, period=None)
        except:
            logging.info("error with interpolation. zeroing out array...")
    
    ig_B_arr =  ig_B_ionosphere_arr + ig_B_inner_arr + ig_B_outer_arr
    
    # calculate dB/dt and populate corresponding arrays
    for i in range(nmin, nmax+1):
        #next compute derivatives
        if i>nmin:
            ig_dB_dt_arr[:,:,i-nmin] = ig_B_arr[:,:,i-nmin] - ig_B_arr[:,:,i-nmin-1]
            ig_dB_dt_ionosphere_arr[:,:,i-nmin] = ig_B_ionosphere_arr[:,:,i-nmin] - ig_B_ionosphere_arr[:,:,i-nmin-1]
            ig_dB_dt_inner_arr[:,:,i-nmin] = ig_B_inner_arr[:,:,i-nmin] - ig_B_inner_arr[:,:,i-nmin-1]
            ig_dB_dt_outer_arr[:,:,i-nmin] = ig_B_outer_arr[:,:,i-nmin] - ig_B_outer_arr[:,:,i-nmin-1]

    # Integrate dB/dt to find induced geoelectric field, by Cagniard's formula. See E_horizontal()
    for i_pos in range(ig_dB_dt_arr.shape[0]):
        E_north, E_east = E_horizontal(ig_dB_dt_arr[i_pos,:,:], pos[i_pos,:], time, sigma = 1e-3, method = 'liu')
        E_north_arr[i_pos,:] = E_north
        E_east_arr[i_pos,:] = E_east
    
    # write geoelectric field to .vlsv
    save_dir = './GIC_{}/'.format(run)   # user defined path
    f_iono = pt.vlsvfile.VlsvReader( '/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/misc_sidecars/ionogrid_FHA.vlsv' )
    
    for i, t in enumerate(time):
        # write to file
        filename_vlsv = save_dir + 'ionosphere_gic_{}_{}.vlsv'.format(run, str(int(t)).zfill(7))
        mkdir_path(filename_vlsv)
        writer = pt.vlsvfile.VlsvWriter(f_iono, filename_vlsv)
        writer.write(pos,'ig_r','VARIABLE','ionosphere')
        writer.write(E_north_arr[:,i],'ig_e_north','VARIABLE','ionosphere')
        writer.write(E_east_arr[:,i],'ig_e_east','VARIABLE','ionosphere')
    
    # Plot timeseries of the geoelectric field at different latitudes
    plt.rcParams["figure.figsize"] = (10, 6)
    
    lat_deg = np.arange(60, 91, 1)
    lat = lat_deg * np.pi / 180.
    theta = (np.pi / 2) - lat
    phi = lat * 0
    
    x0, y0, z0 = spherical_to_cartesian(R_EARTH, theta, phi)
    for i in range(x0.size):
        # Find nearest neighbor of the ionosphere grid, index by 'ind_min', to the specified lat and phi
        dist = np.sqrt((x0[i] - pos[:,0])**2 + (y0[i] - pos[:,1])**2 + (z0[i] - pos[:,2])**2)
        ind_min = np.argmin(dist)
        # PLOT:
        # geoelectic field
        plt.title('GIC Lat. = {} deg., noon, {}'.format(int(lat_deg[i]), run))
        plt.xlabel('time [sec]')
        plt.ylabel(r'Geoelectric field [$\mu$V/m]')
        plt.plot(time, 1e6 * E_north_arr[ind_min,:], label = r'northward E [$\mu$V/m]')
        plt.plot(time, 1e6 * E_east_arr[ind_min,:], label = r'eastward E [$\mu$V/m]')
        plt.ylim([-400, 400])
        plt.legend()
        filename =  '{}plots/geolectric_E_timeseries_lat_{}_{}'.format(save_dir,run,int(lat_deg[i]),run)
        mkdir_path(filename)
        plt.savefig(filename)
        plt.close()
        # dB/dt
        plt.title('dB/dt Lat. = {} deg., noon, {}'.format(int(lat_deg[i]), run))
        plt.xlabel('time [sec]')
        plt.ylabel(r'Ground magnetic field [nT/s]]')
        dB_dt_r, dB_dt_theta, dB_dt_phi = cartesian_to_spherical_vector(ig_dB_dt_arr[ind_min, 0,:], ig_dB_dt_arr[ind_min,1,:], ig_dB_dt_arr[ind_min, 2,:], 
                                                                        pos[ind_min, 0], pos[ind_min,1], pos[ind_min, 2])
        dB_dt_north = -dB_dt_theta; dB_dt_east = dB_dt_phi
        plt.plot(time, 1e9 * dB_dt_north, label = r'northward dB/dt [nT/s]')
        plt.plot(time, 1e9 * dB_dt_east, label = r'eastward dB/dt [nT/s]')
        plt.ylim([-8, 8])
        plt.legend()
        filename =  '{}plots/dB_dt_timeseries_lat_{}_{}'.format(save_dir,run,int(lat_deg[i]),run)
        mkdir_path(filename)
        plt.savefig(filename)
        plt.close()
        # |dB/dt| (components)
        try: # won't work for EGL?
            plt.title('dB/dt Lat. = {} deg., noon, {}'.format(int(lat_deg[i]), run))
            plt.xlabel('time [sec]')
            plt.ylabel(r'Ground magnetic field [nT/s]]')
            dB_dt_r_ionosphere, dB_dt_theta_ionosphere, dB_dt_phi_ionosphere = cartesian_to_spherical_vector(ig_dB_dt_ionosphere_arr[ind_min, 0,:], ig_dB_dt_arr[ind_min,1,:], ig_dB_dt_arr[ind_min, 2,:], 
                                                                                                             pos[ind_min, 0], pos[ind_min,1], pos[ind_min, 2])
            dB_dt_north_ionosphere = -dB_dt_theta_ionosphere; dB_dt_east_ionosphere = dB_dt_phi_ionosphere
            dB_dt_r_inner, dB_dt_theta_inner, dB_dt_phi_inner = cartesian_to_spherical_vector(ig_dB_dt_inner_arr[ind_min, 0,:], ig_dB_dt_inner_arr[ind_min,1,:], ig_dB_dt_inner_arr[ind_min, 2,:], 
                                                                                              pos[ind_min, 0], pos[ind_min,1], pos[ind_min, 2])
            dB_dt_north_inner = -dB_dt_theta_inner; dB_dt_east_inner = dB_dt_phi_inner
            dB_dt_r_outer, dB_dt_theta_outer, dB_dt_phi_outer = cartesian_to_spherical_vector(ig_dB_dt_outer_arr[ind_min, 0,:], ig_dB_dt_outer_arr[ind_min,1,:], ig_dB_dt_outer_arr[ind_min, 2,:], 
                                                                                              pos[ind_min, 0], pos[ind_min,1], pos[ind_min, 2])
            dB_dt_north_outer = -dB_dt_theta_outer; dB_dt_east_outer = dB_dt_phi_outer
            plt.plot(time, np.abs(1e9 * np.sqrt(dB_dt_north**2 + dB_dt_east**2) ), label = r'total |dB/dt| [nT/s]')
            plt.plot(time, np.abs(1e9 * np.sqrt(dB_dt_north_ionosphere**2 + dB_dt_east_ionosphere**2) ), label = r'ionospheric |dB/dt| [nT/s]')
            plt.plot(time, np.abs(1e9 * np.sqrt(dB_dt_north_inner**2 + dB_dt_east_inner**2) ), label = r'inner |dB/dt| [nT/s]')
            plt.plot(time, np.abs(1e9 * np.sqrt(dB_dt_north_outer**2 + dB_dt_east_outer**2) ), label = r'outer |dB/dt| [nT/s]')
            plt.ylim([-8, 8])
            plt.legend()
            filename =  '{}plots/component_dB_dt_timeseries_lat_{}_{}'.format(save_dir,run,int(lat_deg[i]),run)
            mkdir_path(filename)
            plt.savefig(filename)
            plt.close()
        except:
            logging.info("can't plot components!")
    
    
