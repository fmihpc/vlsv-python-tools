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

# This class has a function for plotting multiple variables as subplots in one plot figure

###############################
# WARNING!  This function should be considered deprecated. It is provided only as the
# rankine-hugoniot calculation rankine.py assumes it is available.
###############################


import matplotlib.pyplot as plt
import logging
import numpy as np
from matplotlib.ticker import MaxNLocator

def set_yticks( figure, yticks ):
   ''' Sets figure ticks for subplots of given figure

       :param figure: Some figure whose subplots to set
       :param yticks: Number of ticks to set

       :returns: Edited figure
   '''
   from math import ceil
   new_figure = figure
   if yticks <= 1:
      logging.info("BAD YTICKS SET AT SET_YTICKS!")
      return []
   # Get sub axes
   axes = new_figure.get_axes()
   # Iterate thorugh sub axes
   for ax in axes:
      ax.yaxis.set_major_locator(MaxNLocator(yticks))
   return new_figure

def plot_variables( x, y, figure=[] ):
   ''' Plots x and y variables from the input with pylab

       :param x:        Some variable to be plotted in the x-axis
       :param y:        Some variable to be plotted in the y-axis
       :param figure:   If one wants to plot into an existing figure then the matplotlib figure should be passed as an argument (OPTIONAL)
       :returns: a pylab figure with the plot

       .. code-block:: python

          # Example usage:
          plot_variables( distances, rho )
          # This would plot rho as a function of distance
   '''   
   # Get dimensions of x and y
   import numpy as np
   x_dim = len(np.shape(x))
   y_dim = len(np.shape(y))
   if x_dim != y_dim:
      if x_dim == y_dim - 1:
         from variable import get_data, get_name, get_units
         new_x = [get_data(x) for i in range(len(y)-1)]
         new_x.append(x)
         return plot_multiple_variables( new_x, y, figure, clean_xticks=True )
      else:
         logging.info("ERROR; BAD X AND Y DIMENSIONS " + str(x_dim) + " " + str(y_dim))
         return []
   else:
      if x_dim == 0 and y_dim == 0:
         return plot_multiple_variables( [x], [y], figure )
      else:
         return plot_multiple_variables( x, y, figure )


def plot_multiple_variables( variables_x_list, variables_y_list, figure=[], clean_xticks=False ):
   ''' Plots multiple variables from the input with pylab

       :param variables_x_list:        Some list of variables to be plotted in the x-axis
       :param variables_y_list:        Some list of variables to be plotted in the y-axis
       :param figure:                  If one wants to plot into an existing figure then the matplotlib figure should be passed as an argument (OPTIONAL)
       :returns: a pylab figure with the plot

       .. code-block:: python

          #Example usage:
          plot_multiple_variables( [distances, xcoordinates], [rho, B_x] )
          # This would plot rho_values as a function of distance and B_x_values as a function of xcoordinates

       .. note:: Multiplot expects variables to be saved in the VariableInfo class

       .. note:: If for some reason some variable list (x or y) is empty, e.g. variables_x_list = [B_x, [], B_z, rho], then the variable will not be plotted. This can be used if one wants to plot only into certain subplots.
   '''
   yticks = {}
   for i in range(18):
      tick = i+1
      yticks[tick] = 7 - (int)(i)//(int)(4)


   import numpy as np
   variables_x_list = np.ma.asarray(variables_x_list)
   variables_y_list = np.ma.asarray(variables_y_list)
   if len(variables_x_list) != len(variables_y_list):
      # Attempt to fix the lengths:
      if (len(variables_x_list) == 1):
         if (len(np.atleast_1d(variables_x_list[0])) == len(variables_y_list)):
            variables_y_list = [variables_y_list]
   
      if (len(variables_y_list) == 1):
         if (len(np.atleast_1d(variables_y_list[0])) == len(variables_x_list)):
            variables_x_list = [variables_x_list]

   if len(variables_x_list) != len(variables_y_list):
      logging.info("BAD VARIABLE LENGTH: " + str(len(variables_x_list)) + " " + str(len(variables_y_list)))
      return []
   if len(variables_y_list) > 18:
      logging.info("TOO MANY VARIABLES: " + str(len(variables_y_list)))
      return []
      
   length_of_list = len(variables_x_list)

   if figure != []:
      fig = plt.figure()
      if len(fig.get_axes()) < length_of_list:
         for i in (np.arange(length_of_list-len(fig.get_axes())) + len(fig.get_axes())):
            fig.add_subplot(length_of_list,1,i)
   else:
      fig = plt.figure()
      for i in range(length_of_list):
         fig.add_subplot(length_of_list,1,i+1)

   axes = fig.get_axes()
   from variable import get_data, get_name, get_units
   for i in range(length_of_list):
      
      x = variables_x_list[i]
      y = variables_y_list[i]
      # Check the length of the list
      if (len(np.atleast_1d(x)) == 0) or (len(np.atleast_1d(y)) == 0):
         continue
      ax = axes[i]
      ax.plot(get_data(x), get_data(y), lw=2)

      if get_units(x) != "":
         ax.set_xlabel(get_name(x) + " [" + get_units(x) + "]")
      else:
         ax.set_xlabel(get_name(x))

      if get_units(y) != "":
         ax.set_ylabel(get_name(y) + " [" + get_units(y) + "]")
      else:
         ax.set_ylabel(get_name(y))

      # Set limits
      xlength = np.max(get_data(x)) - np.min(get_data(x))
      ylength = np.max(get_data(y)) - np.min(get_data(y))
      ax.set_xlim([np.min(get_data(x)) - 0.01*xlength, np.max(get_data(x)) + 0.01*xlength])
      ax.set_ylim([np.min(get_data(y)) - 0.05*ylength, np.max(get_data(y)) + 0.05*ylength])      
      # Set format
      ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))

   if clean_xticks == True:
      for i in range(len(np.atleast_1d(axes))-1):
         axes[i].set_xticks([])

   # Set yticks:
   fig = set_yticks( fig, yticks[len(axes)] )
   return fig



