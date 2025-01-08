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
import struct
import xml.etree.ElementTree as ET
import ast
import numpy as np
import os
from reduction import datareducers,data_operators
from vlsvwriter import VlsvWriter
from variable import get_data
from operator import itemgetter
from vlsvreader import VlsvReader

class VlasiatorReader(VlsvReader):
   ''' Class for reading VLSV files with support for Vlasiator velocity space and structures

   '''
   def get_nearest_cellid_with_distribution( self, cellid ):
      # Find the nearest cell id with distribution:
      # Read cell ids with velocity distribution in:
      cell_candidates = self.read("SpatialGrid","CELLSWITHBLOCKS")
      # Read in the coordinates of the cells:
      cell_candidate_coordinates = [self.get_cell_coordinates(cell_candidate) for cell_candidate in cell_candidates]
      # Read in the cell's coordinates:
      cell_coordinates = self.get_cell_coordinates(cellid)
      # Find the nearest:
      norms = np.sum((cell_candidate_coordinates - cell_coordinates)**2, axis=-1)**(1./2)
      norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
      # Get the cell id:
      cellid = cell_candidates[i]
      return cellid

   def get_nearest_coordinates_with_distribution( self, coordinates ):
      # Find the nearest cell id with distribution:
      # Read cell ids with velocity distribution in:
      cell_candidates = self.read("SpatialGrid","CELLSWITHBLOCKS")
      # Read in the coordinates of the cells:
      cell_candidate_coordinates = [self.get_cell_coordinates(cell_candidate) for cell_candidate in cell_candidates]
      # Read in the cell's coordinates:
      cell_coordinates = coordinates
      # Find the nearest:
      norms = np.sum((cell_candidate_coordinates - cell_coordinates)**2, axis=-1)**(1./2)
      norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
      # Get the cell id:
      cellid = cell_candidates[i]
      return self.get_cell_coordinates(cellid)

