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

# Cut-throughs from vlsv files

import numpy as np
import sys

def lineout( vlsvReader, point1, point2, variable, operator="pass",interpolation_order=1, points=100 ):
   ''' Returns a line cut-through from a given VLSV file for distance, coordinates and variable values. The main difference between this and cut_through is that this function interpolates a given variable.

       :param vlsvReader:            Some open VlsvReader
       :type vlsvReader:             :class:`vlsvfile.VlsvReader`
       :param point1:                The starting point of a cut-through line
       :param point2:                The ending point of a cut-through line
       :param variable:              Variable to return
       :param operator:              The operator for the variable, for example "x" for x-component or "magnitude" for magnitude
       :param interpolation_order:   Order of interpolation (0 or 1), defaults to 1
       :param points:                Number of points to return

       :returns: A tuple with output: (distance, coordinates, variable_values)

       .. code-block:: python

          # Example:
          import analysator as pt # import analysator

          vlsvReader = pt.vlsvfile.VlsvReader(\"testfile.vlsv\") # Open a vlsv file
          lineout_rho = pt.calculations.lineout( vlsvReader=vlsvReader, point1=[1.0e5, 1.0e6, 0], point2=[2.0e5, 2.0e6, 0], variable="rho", interpolation_order=1, points=100 )
          distance = lineout_rho[0]
          coordinates = lineout_rho[1]
          values = lineout_rho[2]

   '''
   # Transform point1 and point2 into numpy array:
   point1 = np.array(point1)
   point2 = np.array(point2)
   # Get parameters from the file to determine a good length between points (step length):

   # Make sure point1 and point2 are inside bounds
   if vlsvReader.get_cellid(point1) == 0:
      print("ERROR, POINT1 IN CUT-THROUGH OUT OF BOUNDS!")
   if vlsvReader.get_cellid(point2) == 0:
      print("ERROR, POINT2 IN CUT-THROUGH OUT OF BOUNDS!")

   value_len=len(np.atleast_1d(vlsvReader.read_interpolated_variable( variable, point1, operator)))
   
   if value_len==1:
      values=np.zeros(points)
   else:
      values=np.zeros((points,value_len))

   distance=np.zeros(points)
   coordinates=np.zeros((points,3))
   
   for i in range(points):
      relative_coordinate=(point2 - point1) * i / (points-1)
      if interpolation_order==1:
         values[i]=vlsvReader.read_interpolated_variable( variable, point1 + relative_coordinate, operator)
      elif interpolation_order==0:
         values[i]=vlsvReader.read_variable(variable, vlsvReader.get_cellid(point1 + relative_coordinate), operator)

      distance[i]=np.sqrt(sum(relative_coordinate**2))
      coordinates[i]=point1 +  relative_coordinate

   return (distance,coordinates,values)





