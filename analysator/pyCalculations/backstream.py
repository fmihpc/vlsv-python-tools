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
import logging

def extract_velocity_cells_sphere( vlsvReader, cellid, origin, radius ):
   ''' Extracts the velocity cells inside some given sphere
       :param vlsvReader:         Some VlsvFile with a file open
       :param cellid:             The cellid whose rho to calculate
       :param origin:             Origin for the sphere
       :param radius:             Radius for the sphere
       :returns: Non backstream velocity cells and their avgs values as [vcellids, avgs]
   '''
   # Read the velocity cells:
   velocity_cell_data = vlsvReader.read_velocity_cells(cellid)
   # Get cells:
   vcellids = list(velocity_cell_data.keys())
   # Get avgs data:
   avgs = list(velocity_cell_data.values())
   # Get a list of velocity coordinates shifted by the solar wind bulk velocity:
   origin = np.array(origin)
   v = vlsvReader.get_velocity_cell_coordinates(vcellids) - origin
   # Get sum of radiuses:
   radiuses = np.sum(v**2,axis=-1)
   # Check radius condition
   radius2 = radius**2
   condition = (radiuses <= radius2)
   # Get the velocity cells of sphere
   vcellids_sphere = np.extract(condition, vcellids)
   # Get the avgs
   avgs_sphere = np.extract(condition, avgs)
   # Return
   return [vcellids_sphere, avgs_sphere]

def extract_velocity_cells_non_sphere( vlsvReader, cellid, origin, radius ):
   ''' Retrieves the velocity cells within a given sphere and returns the population outside the given sphere
       :param vlsvReader:         Some VlsvFile with a file open
       :param cellid:             The cellid whose rho to calculate
       :param origin:             Origin for the sphere
       :param radius:             Radius for the sphere
       :returns: Non backstream velocity cells and their avgs values as [vcellids, avgs]
   '''
   # Read the velocity cells:
   velocity_cell_data = vlsvReader.read_velocity_cells(cellid)
   # Get cells:
   vcellids = list(velocity_cell_data.keys())
   # Get avgs data:
   avgs = list(velocity_cell_data.values())
   # Get a list of velocity coordinates shifted by the solar wind bulk velocity:
   origin = np.array(origin)
   v = vlsvReader.get_velocity_cell_coordinates(vcellids) - origin
   # Get sum of radiuses:
   radiuses = np.sum(v**2,axis=-1)
   # Check radius condition
   radius2 = radius**2
   condition = (radiuses > radius2)
   # Get the velocity cells of nonsphere
   vcellids_nonsphere = np.extract(condition, vcellids)
   # Get the avgs
   avgs_nonsphere = np.extract(condition, avgs)
   # Return
   return [vcellids_nonsphere, avgs_nonsphere]
