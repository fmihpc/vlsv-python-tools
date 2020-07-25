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

import analysator as pt
import numpy as np

def vlsv_intpol_file(file_vlsv,file_orbit,varlist,file_output):
   '''Writes interpolated values of variables in ascii file
       :param file_vlsv:             VLSV file
       :param file_orbit:            Orbit file (columns: x,y,z or t,x,y,z)
       :param varlist:               Variable list
       :param file_output:           Output ascii file (columns: x,y,z,cellid,var1,var2,var3,...)
       :returns: none
       .. code-block:: python
          # Example:
          import analysator as pt
          pt.calculations.vlsv_intpol_file("state00040000.vlsv","orbit.dat",["cellB","n_H+sw_ave"],"output.dat")
   '''
   f=pt.vlsvfile.VlsvReader(file_name=file_vlsv)
   points = np.loadtxt(file_orbit,unpack=True).transpose()
   if points.shape[1] == 4:
      points = np.delete(points,0,1) # remove time column
   if points.shape[1] != 3:
      print("ERROR: orbit file must have 3 (x,y,z) or 4 (t,x,y,z) columns")
      return
   [crd,cellids,params,hstr]=pt.calculations.vlsv_intpol_points(f,points,varlist)
   d=np.concatenate((crd,cellids,params),axis=1)
   np.savetxt(file_output,d,"% 05e",header=hstr,comments="% ")



