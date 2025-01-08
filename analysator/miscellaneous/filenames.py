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

'''@package docstring
This module imports glob and os and has some miscellaneous functions for basic file manipulation
'''

import glob
import os

def get_sorted_file_names(name="*.vlsv"):
   ''' Gets the file names in the current directory and sorts them.

         :param name:            Name of the file(s), for example `"*.vlsv"`
         :returns: a list of file names in sorted order
   '''
   fileNames=glob.glob(name)
   fileNames.sort()
   return fileNames
   
