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

'''In this file there are functions that have something to do with fitting data
'''

import logging
def subtract_1d_polynomial_fit( x, y ):
   ''' Fits 1d polynomial into the data, subtracts it from the given data and returns the subtracted data.

       :param x:           The x-axis
       :param y:           Data to manipulate in either variable or raw form
       :returns: data from which a 1d polynomial fit has been subtracted

       .. note::

          This may be useful for fourier transforms
   '''
   # Fit a polynomial into the data
   from variable import get_data, get_name, get_units
   import numpy as np
   parameters = 2
   def function(args, x, y):
      x = np.array(x)
      y = np.array(y)
      value = args[0] + args[1]*x
      return y - value
   from scipy import optimize
   fit = optimize.leastsq(function, np.ones(parameters), args=(get_data(x), get_data(y)))
   y_fitted = (-1)*function(fit[0], x, 0)
   # Create a new array y2 which has a forced constant amplitude for the (possible) waves:
   y2 = y - y_fitted
   # Return the data
   from output import output_1d
   return output_1d( [y2], [get_name(y)], [get_units(y)] )

