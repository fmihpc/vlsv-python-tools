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

# File that creates the output format for arrays
import logging

def output_1d( arrays, names, units="" ):
   ''' Creates an output out of 1d arrays

       :param arrays:    Some arrays with data in them
       :param names:     Names for the array elements
       :param units:     Units for the arrays (optional)
       :returns: the arrays in a new format (dictionary currently)

       .. note::

          arrays and names must be of same length

       .. code-block:: python

          #Example usage:
          output_1d( [[2,1,23], [5,78,4], [2,3,2]], ["rho", "B", "Pressure"] )
          This would interpret [2,1,23] as an array called \"rho\"
   '''
   if units == "":
      units = ["" for i in range(len(arrays))]
   if( (len(arrays) != len(names)) or (len(arrays) != len(units)) ):
      logging.info("BAD ARRAY AND NAME LENGTH IN OUTPUT_1D (pyCalculations/output.py)")
      return []
   new_format = []
   from variable import VariableInfo
   for i in range(len(arrays)):
      variable = VariableInfo(arrays[i])
      variable.name = names[i]
      variable.units = units[i]
      new_format.append(variable)
   if len(new_format) == 1:
      return new_format[0]
   else:
      return new_format
