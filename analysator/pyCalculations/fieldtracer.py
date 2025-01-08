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
import scipy as sp
import analysator as pt
import warnings
from scipy import interpolate
import logging

def dynamic_field_tracer( vlsvReader_list, x0, max_iterations, dx):
   ''' Field tracer in a dynamic time frame

      :param vlsvReader_list:              List of vlsv readers
      :param x0:                           The starting point for the streamlines
      
   '''
   dt = vlsvReader_list[1].read_parameter('t') - vlsvReader_list[0].read_parameter('t')
   # Loop through vlsvreaders:
   v = vlsvReader_list[0].read_interpolated_variable('v', x0)
   iterations = 0
   for vlsvReader in vlsvReader_list:
      stream_plus = static_field_tracer( vlsvReader, x0, max_iterations, dx, direction='+' )
      stream_minus = static_field_tracer( vlsvReader, x0, max_iterations, dx, direction='-' )
      stream = stream_minus[::-1] + stream_plus # Minus reversed
      pt.miscellaneous.write_vtk_file("test" + str(iterations) + ".vtk", stream)
      x0 = x0 + v*dt
      iterations = iterations + 1

def static_field_tracer( vlsvReader, x0, max_iterations, dx, direction='+', bvar='B', centering='default', boundary_inner=-1):
   ''' Field tracer in a static frame

       :param vlsvReader:         An open vlsv file
       :param x:                  Starting point for the field trace
       :param max_iterations:     The maximum amount of iteractions before the algorithm stops
       :param dx:                 One iteration step length
       :param direction:          '+' or '-' or '+-' Follow field in the plus direction or minus direction
       :param bvar:               String, variable name to trace [default 'B']
       :param centering:          String, variable centering: 'face', 'volume', 'node' [defaults to 'face']
       :param boundary_inner:     Float, stop propagation if closer to origin than this value [default -1]
       :returns:                  List of coordinates
   '''

   if(bvar != 'B'):
     warnings.warn("User defined tracing variable detected. fg, volumetric variable results may not work as intended, use face-values instead.")

   if direction == '+-':
     backward = static_field_tracer(vlsvReader, x0, max_iterations, dx, direction='-', bvar=bvar)
     backward.reverse()
     forward = static_field_tracer(vlsvReader, x0, max_iterations, dx, direction='+', bvar=bvar)
     return backward + forward

   f = vlsvReader
   # Read cellids in order to sort variables
   cellids = vlsvReader.read_variable("CellID")
   xsize = f.read_parameter("xcells_ini")
   ysize = f.read_parameter("ycells_ini")
   zsize = f.read_parameter("zcells_ini")
   xmin = f.read_parameter('xmin')
   xmax = f.read_parameter('xmax')
   ymin = f.read_parameter('ymin')
   ymax = f.read_parameter('ymax')
   zmin = f.read_parameter('zmin')
   zmax = f.read_parameter('zmax')

   sizes = np.array([xsize, ysize, zsize])
   maxs = np.array([xmax, ymax, zmax])
   mins = np.array([xmin, ymin, zmin])
   dcell = (maxs - mins)/(sizes.astype('float'))

   # Pick only two coordinate directions to operate in
   if xsize <= 1:
      indices = [2,1]
   if ysize <= 1:
      indices = [2,0]
   if zsize <= 1:
      indices = [1,0]

   if 'vol' in bvar and centering == 'default':
      warnings.warn("Found 'vol' in variable name, assuming volumetric variable and adjusting centering")
      centering = 'volume'

   # Read face_B:
   if centering == 'face' or centering == 'default':
      face_B = f.read_variable(bvar)
      face_Bx = face_B[:,0]
      face_By = face_B[:,1]
      face_Bz = face_B[:,2]

      face_Bx = face_Bx[cellids.argsort()].reshape(sizes[indices])
      face_By = face_By[cellids.argsort()].reshape(sizes[indices])
      face_Bz = face_Bz[cellids.argsort()].reshape(sizes[indices])

      face_B = np.array([face_Bx, face_By, face_Bz])

      # Create x, y, and z coordinates:
      x = np.arange(mins[0], maxs[0], dcell[0]) + 0.5*dcell[0]
      y = np.arange(mins[1], maxs[1], dcell[1]) + 0.5*dcell[1]
      z = np.arange(mins[2], maxs[2], dcell[2]) + 0.5*dcell[2]
      coordinates = np.array([x,y,z])
      # Debug:
      if( len(x) != sizes[0] ):
         logging.info("SIZE WRONG: " + str(len(x)) + " " + str(sizes[0]))

      # Create grid interpolation
      interpolator_face_B_0 = interpolate.RectBivariateSpline(coordinates[indices[0]] - 0.5*dcell[indices[0]], coordinates[indices[1]], face_B[indices[0]], kx=2, ky=2, s=0)
      interpolator_face_B_1 = interpolate.RectBivariateSpline(coordinates[indices[0]], coordinates[indices[1]] - 0.5*dcell[indices[1]], face_B[indices[1]], kx=2, ky=2, s=0)
      interpolators = [interpolator_face_B_0, interpolator_face_B_1]#, interpolator_face_B_2]
   elif centering == 'volume':
      vol_B = f.read_variable(bvar)
      vol_Bx = vol_B[:,0]
      vol_By = vol_B[:,1]
      vol_Bz = vol_B[:,2]

      vol_Bx = vol_Bx[cellids.argsort()].reshape(sizes[indices])
      vol_By = vol_By[cellids.argsort()].reshape(sizes[indices])
      vol_Bz = vol_Bz[cellids.argsort()].reshape(sizes[indices])

      vol_B = np.array([vol_Bx, vol_By, vol_Bz])

      # Create x, y, and z coordinates:
      x = np.arange(mins[0], maxs[0], dcell[0]) + 0.5*dcell[0]
      y = np.arange(mins[1], maxs[1], dcell[1]) + 0.5*dcell[1]
      z = np.arange(mins[2], maxs[2], dcell[2]) + 0.5*dcell[2]
      coordinates = np.array([x,y,z])
      # Debug:
      if( len(x) != sizes[0] ):
         logging.info("SIZE WRONG: " + str(len(x)) + " " + str(sizes[0]))

      # Create grid interpolation
      interpolator_vol_B_0 = interpolate.RectBivariateSpline(coordinates[indices[0]], coordinates[indices[1]], vol_B[indices[0]], kx=2, ky=2, s=0)
      interpolator_vol_B_1 = interpolate.RectBivariateSpline(coordinates[indices[0]], coordinates[indices[1]], vol_B[indices[1]], kx=2, ky=2, s=0)
      interpolators = [interpolator_vol_B_0, interpolator_vol_B_1]#, interpolator_face_B_2]
   elif centering == 'node':
      logging.info("Nodal variables not implemented")
      return
   else:
      logging.info("Unrecognized centering: "+ str(centering))
      return

   #######################################################
   if direction == '-':
      multiplier = -1
   else:
      multiplier = 1

   points = [np.array(x0)]
   for i in range(max_iterations):
      previous_point = points[-1]
      B_unit = np.zeros(3)
      B_unit[indices[0]] = interpolators[0](previous_point[indices[0]], previous_point[indices[1]])
      B_unit[indices[1]] = interpolators[1](previous_point[indices[0]], previous_point[indices[1]])
      B_unit = B_unit / float(np.linalg.norm(B_unit))
      next_point = previous_point + multiplier*B_unit * dx
      if(np.linalg.norm(next_point) < boundary_inner):
         break
      points.append(next_point)
   #######################################################

   return points

# fg tracing for static_field_tracer_3d
def fg_trace(vlsvReader, fg, seed_coords, max_iterations, dx, multiplier, stop_condition, centering = None ):


   # Read field grid variable (denoted 'fg_*' in .vlsv files, no '_vol' suffix)
   # (standardizes input by redefining fg as a numpy array)
   if type(fg) == str:
      if fg == 'fg_b':
          centering = 'face'
      elif fg == 'fg_e':
          centering = 'edge'
      fg = vlsvReader.read_variable(fg)
   else:
      #   fg is already an ndarray
      if not isinstance(fg, np.ndarray):
         raise TypeError("Keyword parameter fg does not seem to be a numpy ndarray.")
      elif fg.ndim!=4 or fg.shape[-1]!=3:
         raise ValueError("Checking array supplied in fg keyword: fg[-1]={} (expected: 3), fg.ndim={} (expected: 4)".format(fg[-1], fg.ndim))


   # Create x, y, and z coordinates:
   xsize = fg.shape[0]
   ysize = fg.shape[1]
   zsize = fg.shape[2]
   xmin = vlsvReader.read_parameter('xmin')
   xmax = vlsvReader.read_parameter('xmax')
   ymin = vlsvReader.read_parameter('ymin')
   ymax = vlsvReader.read_parameter('ymax')
   zmin = vlsvReader.read_parameter('zmin')
   zmax = vlsvReader.read_parameter('zmax')
   sizes = np.array([xsize, ysize, zsize])
   maxs = np.array([xmax, ymax, zmax])
   mins = np.array([xmin, ymin, zmin])
   dcell = (maxs - mins)/(sizes.astype('float'))
   x = np.arange(mins[0], maxs[0], dcell[0]) + 0.5*dcell[0]
   y = np.arange(mins[1], maxs[1], dcell[1]) + 0.5*dcell[1]
   z = np.arange(mins[2], maxs[2], dcell[2]) + 0.5*dcell[2]

   if centering is None:
      logging.info("centering keyword not set! Aborting.")
      return False
  
   # Create grid interpolation object for vector field (V). Feed the object the component data and locations of measurements.
   if centering == 'face':
      interpolator_V_0 = interpolate.RegularGridInterpolator((x-0.5*dcell[0], y, z), fg[:,:,:,0], bounds_error = False, fill_value = np.nan)
      interpolator_V_1 = interpolate.RegularGridInterpolator((x, y-0.5*dcell[1], z), fg[:,:,:,1], bounds_error = False, fill_value = np.nan)
      interpolator_V_2 = interpolate.RegularGridInterpolator((x, y, z-0.5*dcell[2]), fg[:,:,:,2], bounds_error = False, fill_value = np.nan)
   elif centering == 'edge':
      interpolator_V_0 = interpolate.RegularGridInterpolator((x, y-0.5*dcell[1], z-0.5*dcell[2]), fg[:,:,:,0], bounds_error = False, fill_value = np.nan)
      interpolator_V_1 = interpolate.RegularGridInterpolator((x-0.5*dcell[0], y, z-0.5*dcell[2]), fg[:,:,:,1], bounds_error = False, fill_value = np.nan)
      interpolator_V_2 = interpolate.RegularGridInterpolator((x-0.5*dcell[0], y-0.5*dcell[1], z), fg[:,:,:,2], bounds_error = False, fill_value = np.nan)

   interpolators = [interpolator_V_0, interpolator_V_1, interpolator_V_2]

   # Trace vector field lines
   points = seed_coords
   points_traced = np.full((seed_coords.shape[0], max_iterations, 3), np.nan)

   points_traced[:, 0,:] = seed_coords
   mask_update = np.ones(seed_coords.shape[0], dtype = bool)

   # points_traced = [np.array(seed_coords)]              # iteratively append traced trajectories to this list
   # points = points_traced[0]
   # N = len(list(seed_coords))
   V_unit = np.zeros([seed_coords.shape[0], 3])
   for i in range(1, max_iterations):
      V_unit[:, 0] = interpolators[0](points)
      V_unit[:, 1] = interpolators[1](points)
      V_unit[:, 2] = interpolators[2](points)
      V_mag = np.linalg.norm(V_unit, axis=(1))
      V_unit = V_unit / V_mag[np.newaxis,:]
      new_points = points + multiplier*V_unit * dx
      points_traced[mask_update,i,:] = new_points[mask_update,:]

      mask_update[stop_condition(vlsvReader, new_points)] = False
      points = new_points
      # points_traced[~mask_update, i, :] = points_traced[~mask_update, i-1, :]
      
   return points_traced


# vg tracing for static_field_tracer_3d
def vg_trace(vlsvReader, vg, seed_coords, max_iterations, dx, multiplier, stop_condition):
   # Search for the unique coordinates in the given seeds only
   unique_seed_coords,indices = np.unique(seed_coords, axis = 0, return_inverse = True)    # indice here is to reverse the coords order to initial
   n_unique_seeds = unique_seed_coords.shape[0]
   points_traced_unique = np.zeros((n_unique_seeds, max_iterations, 3))
      
   def find_unit_vector(vg, coord):
      val_at_point = vlsvReader.read_interpolated_variable(vg,coord)
      if val_at_point.ndim == 1:
         val_at_point = val_at_point[np.newaxis, :]
      val_mag = np.linalg.norm(val_at_point, axis = 1, keepdims = True)
      return val_at_point/val_mag
      
   unique_seed_coords,indices = np.unique(seed_coords, axis = 0, return_inverse = True)    # indice here is to reverse the coords order to initial
   n_unique_seeds = unique_seed_coords.shape[0]
   points_traced_unique = np.full((n_unique_seeds, max_iterations, 3),np.nan)

   Re = 6371000
   mask_update = np.ones((n_unique_seeds,),dtype = bool) # A mask to determine if the points are still needed to trace further
   points_traced_unique[:, 0, :] = unique_seed_coords


   for i in range(1, max_iterations):

      var_unit = find_unit_vector(vg, points_traced_unique[:, i-1, :])
      next_points = points_traced_unique[:, i-1, :] + multiplier * dx * var_unit

      points_traced_unique[mask_update,i,:] = next_points[mask_update,:]
      # distances = np.linalg.norm(points_traced_unique[:,i,:],axis = 1)
      mask_update[stop_condition(vlsvReader, points_traced_unique[:,i,:])] = False

      # points_traced_unique[~mask_update, i, :] = points_traced_unique[~mask_update, i-1, :]

   points_traced = points_traced_unique[indices,:,:]
   return points_traced

# Default stop tracing condition for the vg tracing, (No stop until max_iteration)
def default_stopping_condition(vlsvReader, points):
   [xmin, ymin, zmin, xmax, ymax, zmax] = vlsvReader.get_spatial_mesh_extent()
   x = points[:, 0]
   y = points[:, 1]
   z = points[:, 2]
   return (x < xmin)|(x > xmax) | (y < ymin)|(y > ymax) | (z < zmin)|(z > zmax)
   # return np.full((points.shape[0]), False)

def static_field_tracer_3d( vlsvReader, seed_coords, max_iterations, dx, direction='+', grid_var = 'vg_b_vol', stop_condition = default_stopping_condition, centering = None ):
   ''' static_field_tracer_3d() integrates along the (static) field-grid vector field to calculate a final position. 
      Code uses forward Euler method to conduct the tracing.
      Based on Analysator's static_field_tracer()
      :Inputs:
       
       param vlsvReader:      A vlsvReader object (~an open .vlsv file)

       param coord_list:      a list of 3-element array-like initial coordinates [ [x1,y1,z1], [x2,y2,z2], ... ] ::
                              if considering just a single starting point, the code accepts a 3-element array-like object [x1,y1,z1]

       param max_iterations:  The maximum number of iterations (int) before the algorithm stops. Total traced length is dx*max_iterations

       param dx:              One iteration step length [meters] (ex. dx=1e4 for typical applications)

       keyword direction:     '+' or '-' or '+-' Follow field in the plus direction, minus direction, or both
       
       keyword grid_var:      Variable to be traced (A string)
                              options include:
                                  grid_var = some string
                                      grid_var = 'vg_b_vol': AMR-grid B-field
                                      grid_var ='fg_b': fieldsolver grid B-field, grid_var='fg_e': fieldsolver grid E-field
                                      static_field_tracer_3d() will load the appropriate variable via the vlsvReader object
                                      NOTE: volumetric variables, with '_vol' suffix, may not work as intended. Use face-centered values: 'fg_b', 'fg_e' etc.
                                  grid_var = some fieldsolver-grid ("fg") array.          dimensions [dimx,dimy,dimz,3]
                                      ex. fg = vlsvobj.read_variable('fg_b')
                                      field grid data is already loaded externally using read_variable() method (see vlsvreader.py).
                                      If fg keyword is set this way, the input vlsvReader is only referred to for metadata (esp. grid dimensions)

                                          keyword stop_condition: Boolean array (seed_coords.shape[0],)

                              Determine when the iteration stop, for the vg trace only
                              If not specified, it will always be True for each seed points.
                              eg. def my_stop(points):
                              
                                    distances = np.linalg.norm(points[:,:],axis = 1)
                                    return (distances <= lower_bound) | (distances >= upper_bound)

      :returns:               points_traced --- a 3d numpy array [len(seed_coords),max_iterations (or 2*max_iterations-1 for '+-'),3]
                              Non-traced sections (e.g. iterations after reaching outer boundaries) filled with np.nan

      keyword centering:     Set to a string ('face' or 'edge') indicating whether the fg variable is face- or edge-centered
                              If keyword fg == 'fg_b', then centering = 'face' (overriding input)
                              If keyword fg == 'fg_e', then centering = 'edge' (overriding input)

      EXAMPLE:            vlsvobj = analysator.vlsvfile.VlsvReader(vlsvfile) 
                          fg_b = vlsvobj.read_variable('fg_b')
                          traces = static_field_tracer_3d( vlsvobj, [[5e7,0,0], [0,0,5e7]], 10, 1e5, direction='+', fg = fg_b )
   '''

   # Standardize input: (N,3) np.array
   if type(seed_coords) != np.ndarray:
      raise TypeError("Please give a numpy array.")

   # Cache and read variables:
   vg = None
   fg = None

   if isinstance(grid_var, str):
      parts = grid_var.split("/")
      for part in parts:
         if part.startswith("fg"):
            fg = grid_var
            break
         elif part.startswith("vg"):   
            vg = grid_var   
            # neighbors_vg = vlsvReader.read_variable_to_cache("vg_regular_interp_neighbors")
            vg_cache = vlsvReader.read_variable_to_cache(vg)
            break
      else:
         raise ValueError("Please give a valid string (eg. 'vg_b_vol')")
   else:
      #   fg is already an ndarray
      if not isinstance(grid_var, np.ndarray):
         raise TypeError("Keyword parameter grid_var does not seem to be a string nor a numpy ndarray.")
      elif fg.ndim!=4 or fg.shape[-1]!=3:
         raise ValueError("Checking array supplied in grid_var keyword: fg[-1]={} (expected: 3), fg.ndim={} (expected: 4)".format(fg[-1], fg.ndim))
      else:
         raise TypeError("Unrecognized grid_var keyword data (expect string or numpy.ndarray)")
         
   # Recursion (trace in both directions and concatenate the results)
   if direction == '+-':
      backward = static_field_tracer_3d(vlsvReader, seed_coords, max_iterations, dx, direction='-', grid_var = grid_var, stop_condition = default_stopping_condition, centering = centering)
      # backward.reverse()
      forward = static_field_tracer_3d(vlsvReader, seed_coords, max_iterations, dx, direction='+', grid_var = grid_var, stop_condition = default_stopping_condition, centering = centering)
      return np.concatenate((backward[:,::-1,:],forward[:, 1:, :]), axis = 1)

   multiplier = -1 if direction == '-' else 1   
   
   if fg is not None:
      points_traced = fg_trace(vlsvReader, fg, seed_coords, max_iterations, dx, multiplier, stop_condition, centering )
   
   elif vg is not None:
      points_traced = vg_trace(vlsvReader, vg, seed_coords, max_iterations, dx, multiplier, stop_condition)


   return points_traced       # list for fg; 3d numpy array(N,maxiterations,3) for vg

