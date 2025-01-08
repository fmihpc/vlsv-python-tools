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
import logging

class DataReducerVariable:
   ''' A class for creating custom variables that are being read by the VlsvReader class. This is useful for reading in variables that are not written in the vlsv file directly
   '''
   variables = []
   operation = []
   units = []
   latex = ""
   latexunits = ""
   useVspace = False
   useReader = False
   vector_size=1
   def __init__(self, variables, operation, units, vector_size, latex="", latexunits="",useVspace=False,useReader=False):
      ''' Constructor for the class
          :param variables          List of variables for doing calculations with
          :param operation          The operation (function) that operates on the variables
          :param units              Units of the variable
          :param latex              Name of the variable in LaTeX markup
          :param latexunits         Units of the variable in LaTeX markup
          :param vector_size       Length of vector for reducer to return (scalars 1, vectors 3, tensors 9)
          :param useVspace          Flag to determine whether the reducer will use velocity space data
          Example:
          def plus( array ):
             return array[0]+array[1]
          new_var = DataReducerVariable( ["rho", "rho"], plus, "1/m^3" )
      '''

      self.variables = variables
      self.operation = operation
      self.units = units
      self.vector_size = vector_size
      self.latex = latex
      self.latexunits = latexunits
      self.useVspace = useVspace
      self.useReader = useReader

