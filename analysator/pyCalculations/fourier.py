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

# Function for taking 1d and 2d fourier transforms here.
import logging

def fourier( t, y, kaiserwindowparameter=0 ):
   ''' Function for returning fourier series and frequencies of some given arrays t and y


       :param t:           Time
       :param y:            Some variable data
       :returns: the frequencies, new time variables and frequencies

       .. note::

          return format: [FFT, frequencies, t, y]

       .. note::

          t must have a constant time stepping

       .. code-block:: python

          Example usage:
          fourier_data = fourier( t=np.arange(0,500,0.5), y=rho_data, kaiserwindowparameter=14 )

   '''
   # Get the data
   from variable import get_data
   #t_data = get_data(t)
   y_data = get_data(y)
   # First check the t array whether it has a constant t
   #t = get_data(t)[1] - get_data(t)[0]

   #for i in xrange(len(get_data(t))-1):
   #   if t != get_data(t)[i+1] - get_data(t)[i]:
   #      logging.info "Gave bad timestep to plot_fourier, the time step in array t must be constant (for now)"
   # Use kaiser window on y
   import numpy as np
   y_tmp = get_data(y) * np.kaiser(len(get_data(y)), kaiserwindowparameter)
   # Do FFT on the data
   fourier=np.fft.fft(y_tmp) * (1/(float)(len(y_tmp)))
   # Get frequencies of the fourier
   freq=np.fft.fftfreq(len(fourier), d=t)
   # Declare t2 (Note: This is the same as t but we want the steps to be thicker so the data looks smoother
   t2=t*0.01
   t2=np.arange(len(y_tmp)*100)*t2
   # Declare y2
   y2=np.array([np.sum(fourier*np.exp(complex(0,1)*2*np.pi*freq*T)) for T in t2])
   from output import output_1d
   from variable import get_name
   # Get the indexes:
   toIndex = (int)((len(freq)//2)/2.0 + 1) # the division // truncates to int. Kept this way so consistent with python2.7
   return output_1d([2*np.abs(fourier[1:toIndex]), freq[1:toIndex], t2, y2], ["FFT", "frequency", "time", get_name(y)])

