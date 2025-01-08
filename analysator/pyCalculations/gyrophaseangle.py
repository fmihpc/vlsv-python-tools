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
from rotation import rotateVectorToVector
import logging

def gyrophase_angles_from_file( vlsvReader, cellid):
   ''' Calculates the gyrophase angle angle distribution for a given cell with a given file
   :param vlsvReader:        Some VlsvReader class with a file open
   :type vlsvReader:         :class:`vlsvfile.VlsvReader`
   :param cellid:            The cell id whose gyrophase angle the user wants NOTE: The cell id must have a velocity distribution!
   :returns: gyrophase angles and avgs [gyro_angles, avgs]
   
   .. code-block:: python
   
   # Example usage:
   vlsvReader = VlsvReader("fullf.0001.vlsv")
   result = gyrophase_angles_from_file( vlsvReader=vlsvReader, cellid=1924)
   # Plot the data
   import matplotlib.pyplot as plt
   plt.hist(result[0].data, weights=result[1].data, bins=100, log=False)
   '''
   # Read the velocity cells:
   velocity_cell_data = vlsvReader.read_velocity_cells(cellid)
   if velocity_cell_data == []:
      from output import output_1d
      return output_1d([[0.0, 1.0], [1.0, 1.0]], ["Gyrophase_angle", "avgs"], ["", ""])
   # Read bulk velocity:
   if vlsvReader.check_variable( "rho" ):
      if vlsvReader.read_variable("rho", cellid) != 0.0:
         bulk_velocity = np.array(vlsvReader.read_variable("rho_v", cellid) / vlsvReader.read_variable("rho", cellid), copy=False)
      else:
         from output import output_1d
         return output_1d([[0.0, 1.0], [1.0, 1.0]], ["Gyrophase_angle", "avgs"], ["", ""])
   else:
      moments = np.array(vlsvReader.read_variable("moments", cellid))
      if moments[0] != 0.0:
         bulk_velocity = np.array(np.divide(np.array(moments[1:]), moments[0]))
      else:
         from output import output_1d
         return output_1d([[0.0, 1.0], [1.0, 1.0]], ["Gyrophase_angle", "avgs"], ["", ""])
   # Calculate the gyrophase angles for the data:
   if vlsvReader.check_variable( "B" ):
      B = vlsvReader.read_variable("B", cellid)
   else:
      B = vlsvReader.read_variable("background_B", cellid) + vlsvReader.read_variable("perturbed_B", cellid)
   if np.linalg.norm(B) != 0.0:
      B_unit = B / np.linalg.norm(B)
   else:
      from output import output_1d
      return output_1d([[0.0, 1.0], [1.0, 1.0]], ["Gyrophase_angle", "avgs"], ["", ""])
   # Get cells:
   vcellids = list(velocity_cell_data.keys())
   # Get a list of velocity coordinates:
   velocity_coordinates = vlsvReader.get_velocity_cell_coordinates(vcellids)
   return gyrophase_angles(bulk_velocity, B_unit, velocity_cell_data, velocity_coordinates)



def gyrophase_angles(bulk_velocity, B_unit, velocity_cell_data, velocity_coordinates, plasmaframe=True, cosine=False):
   ''' Calculates the gyrophase angle angle distribution for a given cell
   
   :param bulk_velocity: TODO
   :param B_unit: TODO
   :param velocity_coordinates: TODO
   :param cosine:            True if returning the gyrophase angles as a cosine plot
   :param plasmaframe:       True if the user wants to get the gyrophase angle distribution in the plasma frame, default True
   :returns: gyrophase angles and avgs [gyro_angles, avgs]
   
   .. code-block:: python
   
   # Example usage:
   vlsvReader = VlsvReader("fullf.0001.vlsv")
   result = gyrophase_angles_from_file( vlsvReader=vlsvReader, cellid=1924, cosine=True, plasmaframe=False )
   # Plot the data
   import matplotlib.pyplot as plt
   plt.hist(result[0].data, weights=result[1].data, bins=100, log=False)
   '''
   
   # Get avgs data:
   avgs = list(velocity_cell_data.values())
   # Shift to plasma frame
   if plasmaframe == True:
      velocity_coordinates = velocity_coordinates - bulk_velocity
   # Get norms:
   v_norms = np.sum(np.abs(velocity_coordinates)**2,axis=-1)**(1./2)
   # Get the angles:
   v_rotated = rotateVectorToVector(velocity_coordinates, B_unit)
   v_rotated = np.asarray(v_rotated)
   if cosine == True:
      gyro_angles = np.cos(np.arctan2(v_rotated[:,0], v_rotated[:,1]))
      units = ""
   else:
      gyro_angles = np.arctan2(v_rotated[:,0], v_rotated[:,1]) / (2*np.pi) * 360
      units = "degree"
   # Return the gyrophase angles and avgs values:
   from output import output_1d
   return output_1d([gyro_angles, avgs], ["Gyrophase_angle", "avgs"], [units, ""])
   #plt.hist(gyro_angles, weights=avgs, bins=bins, log=log)

