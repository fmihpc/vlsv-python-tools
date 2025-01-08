# 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2018 University of Helsinki
# 
# For details of usage, see the COPYING file and read the "Rules of the Road"
# at http://www.physics.helsinki.fi/vlasiator/
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rotation import rotateVectorToVector
from scipy.interpolate import griddata
from scipy.signal import sepfir2d
from packaging.version import Version
import logging

# Detector data obtained from the Themis ESA instrument paper
# http://dx.doi.org/10.1007/s11214-008-9440-2
detector_opening_angle = 6  # in degrees
detector_angle_bins = 32    # angular bins for 360 degress
detector_energy_bins = 32   # Number of energy bins
detector_min_speed = 17509  # in m/s
detector_max_speed = 2188e3 # in m/s
detector_geometric_factor = 0.00000061 # m^2 sr E
#detector_timestep = 0.003   # in s, readout time
detector_timestep = 0.062   # in s, binning time

proton_mass = 1.67e-27      # in kg

# Themis colormap, as extracted from the themis tools' IDL file
themis_colors=[(0,0,0),(.5,0,.5),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0)]
themis_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("themis",themis_colors)

def get_dv(vlsvReader,pop="proton"):
   # Get velocity grid sizes:
   vel_mesh_size = vlsvReader.get_velocity_mesh_size(pop=pop)
   vel_block_size = vlsvReader.get_velocity_block_size(pop=pop)
   vxcells = vel_mesh_size[0]*vel_block_size[0]
   vycells = vel_mesh_size[1]*vel_block_size[1]
   vzcells = vel_mesh_size[2]*vel_block_size[2]
   
   vel_mesh_limits = vlsvReader.get_velocity_mesh_extent(pop=pop)
   vxmin = vel_mesh_limits[0]
   vymin = vel_mesh_limits[1]
   vzmin = vel_mesh_limits[2]
   vxmax = vel_mesh_limits[3]
   vymax = vel_mesh_limits[4]
   vzmax = vel_mesh_limits[5]

   dvx = (vxmax - vxmin) / (float)(vxcells)
   dvy = (vymax - vymin) / (float)(vycells)
   dvz = (vzmax - vzmin) / (float)(vzcells)
   return [dvx,dvy,dvz]

def simulation_to_spacecraft_frame(spinvector, detector_axis, phi=0):
    ''' Builds a matrix to transform coordinates from simulation frame into spaceraft frame
    :param spinvector  Spacecraft spin axis, in simulation coordinates
    :param detector_axis Detector plane normal axis
    :param phi         Rotate spacecraft around spin axis after setting up coordinate system
    '''
    # TODO: Normalize vectors?
    y = np.cross(detector_axis,spinvector)
    z = np.cross(spinvector, y)
    yr = np.cos(phi)*y - np.sin(phi)*z
    zr = np.sin(phi)*y + np.cos(phi)*z
    m = np.array([spinvector, yr, zr])

    return m

def spacecraft_to_simulation_frame(spinvector, detector_axis, phi=0):
    ''' Builds a matrix to transform coordinates from spaceraft frame back to simulation frame
    :param spinvector  Spacecraft spin axis, in simulation coordinates
    :param detector_axis Detector plane normal axis
    :param phi         Rotate spacecraft around spin axis after setting up coordinate system
    '''
    return simulation_to_spacecraft_frame(spinvector,detector_axis,phi).T

def simulation_to_observation_frame(x_axis,y_axis):
    ''' Builds a 3x3 matrix to transform velocities into an observation plane
    :param x_axis:  x-axis of the observation plane (in simulation coordinates)
    :param y_axis:  y-axis of the observation plane (gets orthonormalized)
    '''
    xn = np.linalg.norm(x_axis)
    x_axis /= xn
    p = x_axis.dot(y_axis)
    y_axis -= p*x_axis
    yn = np.linalg.norm(y_axis)
    y_axis /= yn
    z_axis = np.cross(x_axis,y_axis)
    return np.array([x_axis,y_axis,z_axis])

def themis_plot_detector(vlsvReader, cellID, detector_axis=np.array([0,1,0]), pop="proton"):
    ''' Plots a view of the detector countrates using matplotlib
    :param vlsvReader:        Some VlsvReader class with a file open
    :type vlsvReader:         :class:`vlsvfile.VlsvReader`
    :param cellid:            The cell id where the distribution is supposet to be sampled NOTE: The cell id must have a velocity distribution!
    :param detector_axis:     detector axis direction (note: this is not spacecraft spin axis!)
    '''

    matrix = spacecraft_to_simulation_frame(np.cross(np.array([1.,0,0]),detector_axis),detector_axis)

    logging.info("Getting phasespace data...")
    angles, energies, vmin, vmax, values = themis_observation_from_file( vlsvReader=vlsvReader,
                cellid=cellID, matrix=matrix, binOffset=[-0.5,-0.5],pop=pop)
    if vmin == 0:
        vmin = 1e-3
    if vmax <= vmin:
        vmax = vmin * 10.

    values = abs(values);

    grid_r, grid_theta = np.meshgrid(energies,angles)
    fig,ax=plt.subplots(subplot_kw=dict(projection="polar"),figsize=(12,10))
    ax.set_title("Detector view at cell " + str(cellID))
    logging.info("Plotting...")
    cax = ax.pcolormesh(grid_theta,grid_r,values, norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax), cmap=themis_colormap)
    ax.grid(True)
    fig.colorbar(cax)
    plt.show()

def themis_plot_phasespace_contour(vlsvReader, cellID, plane_x=np.array([1.,0,0]), plane_y=np.array([0,0,1.]), smooth=False, xlabel="Vx", ylabel="Vy", pop="proton"):
    ''' Plots a contour view of phasespace, as seen by a themis detector, at the given cellID
    :param vlsvReader:        Some VlsvReader class with a file open
    :type vlsvReader:         :class:`vlsvfile.VlsvReader`
    :param cellid:            The cell id where the distribution is supposet to be sampled NOTE: The cell id must have a velocity distribution!
    :param plane_x and plane_y: x and y direction of the resulting plot plane
    '''

    matrix = simulation_to_observation_frame(plane_x,plane_y)

    angles, energies, vmin, vmax, values = themis_observation_from_file( vlsvReader=vlsvReader, cellid=cellID, matrix=matrix,pop=pop)

    if vmin == 0:
        vmin = 1e-15
    if vmax <= vmin:
        vmax = vmin * 10.

    # Regrid into cartesian space, 256x256:
    grid_r, grid_theta = np.meshgrid(energies,angles)

    grid_x = -grid_r * np.sin(grid_theta)  # turn radial grid points into (x, y)
    grid_y = -grid_r * np.cos(grid_theta)  # (the - comes from detector-look-direction vs particle-movement-direction)

    hires_x = np.linspace(-2200,2200,256);
    hires_y = np.linspace(-2200,2200,256);
    xi,yi = np.meshgrid(hires_x,hires_y);
    vi = griddata( (grid_x.flatten(),grid_y.flatten()), values.flatten(), (xi,yi))

    if smooth:
        # Convolve the grid data with a gaussian kernel
        blurkernel = np.exp(-.17*np.power([6,5,4,3,2,1,0,1,2,3,4,5,6],2))
        vi = sepfir2d(vi, blurkernel, blurkernel) / 4.2983098411528502

    fig,ax=plt.subplots(figsize=(12,10))
    ax.set_aspect('equal')
    ax.set_title("Phasespace at cell " + str(cellID))
    ax.set_xlabel(xlabel+" (km/s)")
    ax.set_ylabel(ylabel+" (km/s)")
    cax = ax.contour(xi,yi,vi.T, levels=np.logspace(np.log10(vmin),np.log10(vmax),20), norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax))
    ax.grid(True)
    fig.colorbar(cax)
    plt.show()

def themis_plot_phasespace_helistyle(vlsvReader, cellID, plane_x=np.array([1.,0,0]), plane_y=np.array([0,0,1.]), smooth=True, xlabel="Vx", ylabel="Vy"):
    ''' Plots a view of phasespace, as seen by a themis detector, at the given cellID, in the style that heli likes.
    :param vlsvReader:        Some VlsvReader class with a file open
    :type vlsvReader:         :class:`vlsvfile.VlsvReader`
    :param cellid:            The cell id where the distribution is supposet to be sampled NOTE: The cell id must have a velocity distribution!
    :param smooth:            Smooth re-gridded phasespace before plotting
    :param plane_x and plane_y: x and y direction of the resulting plot plane
    '''

    matrix = simulation_to_observation_frame(plane_x,plane_y)

    angles, energies, vmin, vmax, values = themis_observation_from_file( vlsvReader=vlsvReader, cellid=cellID, matrix=matrix, countrates=False)
    if vmin == 0:
        vmin = 1e-15
    if vmax < vmin:
        vmax = vmin*10

    # Regrid into cartesian space, 256x256:
    grid_r, grid_theta = np.meshgrid(energies,angles)

    grid_x = -grid_r * np.sin(grid_theta)  # turn radial grid points into (x, y)
    grid_y = -grid_r * np.cos(grid_theta)

    hires_x = np.linspace(-2200,2200,256);
    hires_y = np.linspace(-2200,2200,256);
    xi,yi = np.meshgrid(hires_x,hires_y);
    vi = griddata( (grid_x.flatten(),grid_y.flatten()), values.flatten(), (xi,yi), method='linear')

    if smooth:
        # Convolve the grid data with a gaussian kernel
        blurkernel = np.exp(-.17*np.power([6,5,4,3,2,1,0,1,2,3,4,5,6],2))
        vi = sepfir2d(vi, blurkernel, blurkernel) / 4.2983098411528502

    fig,ax=plt.subplots(figsize=(12,10))
    ax.set_aspect('equal')
    ax.set_title("Phasespace at cell " + str(cellID))
    ax.set_xlabel(xlabel+" (km/s)")
    ax.set_ylabel(ylabel+" (km/s)")
    if Version(matplotlib.__version__) < Version("3.5.0"):
        cmapuse=matplotlib.cm.get_cmap(name="Blues")
    else:
        cmapuse=matplotlib.colormaps.get_cmap("Blues")
    cax = ax.pcolormesh(xi,yi,vi.T, norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax), vmin=vmin, vmax=vmax, cmap=cmapuse, shading='flat')
    cax2 = ax.contourf(xi,yi,vi.T, levels=np.logspace(np.log10(vmin),np.log10(vmax),20), norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax), vmin=vmin, vmax=vmax, cmap=cmapuse)
    #cax3 = ax.contour(xi,yi,vi.T, levels=np.logspace(np.log10(vmin),np.log10(vmax),20), norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax), cmap=pl.get_cmap("binary"))
    ax.grid(True)
    fig.colorbar(cax)
    plt.show()
def themis_observation_from_file( vlsvReader, cellid, matrix=np.array([[1,0,0],[0,1,0],[0,0,1]]), countrates=True, interpolate=True,binOffset=[0.,0.],pop='proton'):
   ''' Calculates artificial THEMIS EMS observation from the given cell
   :param vlsvReader:        Some VlsvReader class with a file open
   :type vlsvReader:         :class:`vlsvfile.VlsvReader`
   :param cellid:            The cell id where the distribution is supposet to be sampled NOTE: The cell id must have a velocity distribution!
   :param matrix:            Matrix to transform velocities from simulation space into detector space (use simulation_to_spacecraft_frame helper function)
   :param countrates: Transform phase space densities into count rates?
   :param interpolate: interpolate into detector bins?
   :param binOffset: offset bin allocation in (angle, energy) by this amount (used to align polar plots)
   :returns: detector bins angles, energies and counts [bin_thetas,energies,min,max,counts]

   .. code-block:: python

   # Example usage:
   vlsvReader = VlsvReader("fullf.0001.vlsv")
   angles, energies, vmin, vmax, values = themis_observation_from_file( vlsvReader=self.vlsvReader, cellid=cellid)
   # plot:
   grid_r, grid_theta = np.meshgrid(energies,angles)
   fig,ax=pl.subplots(subplot_kw=dict(projection="polar"),figsize=(12,10))
   ax.set_title("Detector view at cell " + str(cellid))
   cax = ax.pcolormesh(grid_theta,grid_r,values, norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax))
   pl.show()
   '''
   # Get velocity space resolution
   dvx,dvy,dvz = get_dv(vlsvReader,pop=pop)

   # Read the velocity cells:
   velocity_cell_data = vlsvReader.read_velocity_cells(cellid,pop=pop)
   if velocity_cell_data == []:
      # No velocity space data here, return empty result
      return [[],[],0,0,[]]
   # Calculate the detector histogram
   # Get cells:
   vcellids = list(velocity_cell_data.keys())
   # Get a list of velocity coordinates:
   velocity_coordinates = vlsvReader.get_velocity_cell_coordinates(vcellids,pop=pop)
   angles,energies,detector_values = themis_observation(velocity_cell_data, velocity_coordinates,matrix,dvx,countrates=countrates, interpolate=interpolate,binOffset=binOffset)

   # Calc min and max
   val_min = np.min(detector_values)
   val_max = np.max(detector_values)
   return [angles,energies,val_min,val_max,detector_values]



def themis_observation(velocity_cell_data, velocity_coordinates, matrix, dv=30e3, countrates=True, interpolate=True, binOffset=[0.,0.]):
   ''' Calculates artificial THEMIS EMS observation from the given velocity space data

   :param velocity_cell_data: velocity cell information as obtained from vlsvReader
   :param velocity_coordinates: coordinates associated with the cells
   :param matrix: Matrix to transform velocities from simulation space into detector space (use simulation_to_spacecraft_frame helper function)
   :param dv: velocity space resolution (in km/s)
   :param countrates: Transform phase space densities into count rates?
   :param interpolate: interpolate into detector bins?
   :param binOffset: offset bin allocation in (angle, energy) by this amount (used to align polar plots)
   :returns: detector bins angles, energies and counts [bin_thetas,energies,counts]

   .. code-block:: python

   # Example usage:
   vlsvReader = VlsvReader("fullf.0001.vlsv")
   angles, energies, vmin, vmax, values = themis_observation_from_file( vlsvReader=self.vlsvReader, cellid=cellid)
   # plot:
   grid_r, grid_theta = np.meshgrid(energies,angles)
   fig,ax=pl.subplots(subplot_kw=dict(projection="polar"),figsize=(12,10))
   ax.set_title("Detector view at cell " + str(cellid))
   cax = ax.pcolormesh(grid_theta,grid_r,values, norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax))
   pl.show()
   '''

   # Get avgs data:
   avgs = list(velocity_cell_data.values())
   # Shift to plasma frame
   #if plasmaframe == True:
   #   velocity_coordinates = velocity_coordinates - bulk_velocity
   # Get velocity absolute values:
   v_abs = np.sum(np.abs(velocity_coordinates)**2,axis=-1)**(1./2)
   # Transform into detector's frame
   v_rotated = matrix.dot(velocity_coordinates.T).T
   v_rotated = np.asarray(v_rotated)
   #v_rotated = velocity_coordinates

   # Calculate phi and theta angles
   phi_angles = (np.arctan2(v_rotated[:,1], v_rotated[:,0]) + np.pi) / (2*np.pi) * 360  # 0 .. 360
   theta_angles = np.arccos(v_rotated[:,2] / v_abs) / (2*np.pi) * 360                   # 0 .. 180

   # now map them all into detector bins
   detector_values = np.zeros([detector_angle_bins+1, detector_energy_bins])
   equator_angles = np.abs(90.-theta_angles)
   angle_bins = phi_angles / 360 * detector_angle_bins + binOffset[0]
   energy_bins = np.log(v_abs / detector_min_speed)/np.log(detector_max_speed/detector_min_speed) * detector_energy_bins + binOffset[1]
   for i in range(0,v_abs.shape[0]):
       if equator_angles[i] > detector_opening_angle:
           continue
       #logging.info("Using cell with velocities " + str(v_rotated[i,:]) + ", phi = " + str(phi_angles[i]) + ", theta = " + str(theta_angles[i]))

       if energy_bins[i] >= detector_energy_bins-1:
           continue

       target_val = avgs[i]
       if countrates:
           # Calculate actual countrate from phasespace density
           # (=rho*g/E*v*dt)
           #deltaOmega = dv*dv/v_abs[i]                                                      # solid angle covered by the phase space cell (approx)
           deltaOmega = np.pi * np.pi / detector_angle_bins * detector_opening_angle / 360. # Opening angle of the detector bin (in sterad)
           #deltaE = proton_mass * v_abs[i] * dv                                             # Energy delta of the phase space cell
           deltaE = 0.32 * proton_mass * v_abs[i] * v_abs[i] / 1.6e-19                      # Energy delta in eV of the detector bin (approx)
           effectiveArea = detector_geometric_factor / deltaOmega / deltaE
           velcell_volume = dv*dv*dv

           target_val = avgs[i] * velcell_volume * effectiveArea * v_abs[i] * detector_timestep

       if interpolate:
           # Linearly interpolate
           angle_int = int(angle_bins[i])
           energy_int = int(energy_bins[i])
           angle_frac = angle_bins[i] - int(angle_bins[i])
           energy_frac = energy_bins[i] - int(energy_bins[i])
           detector_values[angle_int,energy_int] += (1.-angle_frac) * (1.-energy_frac) * target_val
           detector_values[(angle_int+1)%detector_angle_bins,energy_int] += (angle_frac) * (1.-energy_frac)  * target_val
           detector_values[angle_int,energy_int+1] += (1.-angle_frac) * (energy_frac)  * target_val
           detector_values[(angle_int+1)%detector_angle_bins,energy_int+1] += (angle_frac) * (energy_frac)   * target_val
       else:
           # Weigh into nearest cell
           detector_values[int(angle_bins[i]),int(energy_bins[i])] += target_val

   return_angles = np.linspace(0.,2*np.pi,detector_angle_bins+1)
   return_energies = np.logspace(np.log10(detector_min_speed/1e3),np.log10(detector_max_speed/1e3),detector_energy_bins)
   return [return_angles,return_energies,detector_values]
