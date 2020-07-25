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
import sys

def vlsv_intpol_points(vlsvReader,points,varlist,operator="pass",interpolation_order=1):
   '''Returns interpolated values of variables at given points
       :param vlsvReader:            Some open VlsvReader
       :type vlsvReader:             :class:`vlsvfile.VlsvReader`
       :param points:                Coordinate points
       :param varlist:               Variable list, if empty all variables are got
       :param operator:              The operator for the variable, for example "x" for x-component or "magnitude" for magnitude
       :param interpolation_order:   Order of interpolation (0 or 1), defaults to 1
       :returns: A tuple with output: (coordinates,variable_values,header_string)
       .. code-block:: python
          # Example:
          import analysator as pt
          import numpy as np
          f=pt.vlsvfile.VlsvReader(file_name="state00040000.vlsv")
          mesh_limits = f.get_spatial_mesh_extent()
          x = np.linspace(mesh_limits[0],mesh_limits[3],100) # points along x-axis
          y = np.zeros(len(x))
          z = np.zeros(len(x))
          points=(np.array([x,y,z])).transpose()
          varlist = ["cellB","n_H+sw_ave"]
          [crd,cellids,params,hstr]=pt.calculations.vlsv_intpol_points(f,points,varlist)
          x=crd[:,0]
          y=crd[:,1]
          z=crd[:,2]
          Bx=params[:,0]
          By=params[:,1]
          Bz=params[:,2]
          n=params[:,3]
   '''
   N_points = len(points)
   N_vars = len(varlist)
   if N_vars <= 0:
      varlist = vlsvReader.get_all_variables()
      N_vars = len(varlist)
   if N_vars <= 0:
      print("ERROR: len(varlist) = 0")
      return
   if N_points < 0:
      print("ERROR: len(points) = 0")
      return
   header = "x y z cellid " # header string
   for i in range(N_vars): # loop variable list
      var = varlist[i]
      if vlsvReader.check_variable(var) == False:
         print("ERROR: variable " + var + " does not exist in file " + vlsvReader.file_name)
         return
      dim=len(np.atleast_1d(vlsvReader.read_interpolated_variable(var,points[0],operator))) # variable dimensions
      if dim <= 0:
         print("ERROR: bad variable dimension (dim=" + str(dim) + ")")
         return
      values=np.zeros((N_points,dim))
      crds=np.zeros((N_points,3)) # coordinates
      cellids=np.zeros((N_points,1)) # cell ids
      for j in range(N_points):
         cellids[j] = vlsvReader.get_cellid(points[j])
         if cellids[j] == 0: # coordinates of a point out of domain
            values[j]=np.ones(dim)*np.nan
         elif interpolation_order==1:
            values[j]=vlsvReader.read_interpolated_variable(var,points[j],operator)
         elif interpolation_order==0:
            values[j]=vlsvReader.read_variable(var,cellids[j],operator)
         crds[j]=points[j]
      if i==0:
         res=values
      else:
         res=np.concatenate((res,values),axis=1)
      if dim == 1:
         header = header + var + " "
      else:
         for d in range(dim):
            header = header + var + "(" + str(d) + ") "
   return (crds,cellids,res,header)





