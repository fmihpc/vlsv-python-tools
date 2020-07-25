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

import mayavi.mlab
import mayavi
from tvtk.api import tvtk
import numpy as np



def generate_custom_velocity_grid( vlsvReader, blocks_and_values, iso_surface=False ):
   '''Generates a velocity grid from a given spatial cell id

      :param vlsvReader:           Some vlsv reader with a file open
      :param velocity_cell_map:    Given velocity cell ids and values in python dict() format (see read_velocity_cells function in vlsvReader)
      :param iso_surface:          If true, plots the iso surface

      # Example usage:

      import analysator as pt

      #vlsvReader = pt.vlsvfile.VlsvReader("example.vlsv")

      cellid = 1111
      velocity_cell_map = vlsvReader.read_velocity_cells(cellid)
      velocity_cell_ids = velocity_cell_map.keys()
      velocity_cell_values = velocity_cell_map.values()

      velocity_cell_map[velocity_cell_ids[10]] = 3e-7

      generate_custom_velocity_grid( vlsvReader, velocity_cell_map )

   '''
   # Create nodes
   # Get velocity blocks and avgs:
   # Get helper function:
   blocksAndAvgs = blocks_and_values

   if len(blocksAndAvgs) == 0:
      print("CELL " + str(cellid) + " HAS NO VELOCITY BLOCK")
      return False
   # Create a new scene
   #engine.new_scene()
   #mayavi.mlab.set_engine(engine)
   # Create a new figure
   #figure = mayavi.mlab.clf()
   #figure.scene.disable_render = True
   blocks = blocksAndAvgs[0]
   avgs = blocksAndAvgs[1]
   # Get nodes:
   nodesAndKeys = vlsvReader.construct_velocity_cell_nodes(blocks)
   # Create an unstructured grid:
   points = nodesAndKeys[0]
   tets = nodesAndKeys[1]
   tet_type=tvtk.Voxel().cell_type#VTK_VOXEL

   ug=tvtk.UnstructuredGrid(points=points)
   # Set up the cells
   ug.set_cells(tet_type,tets)
   # Input data
   values=np.ravel(avgs)
   ug.cell_data.scalars=values
   ug.cell_data.scalars.name='avgs'
   #figure.scene.disable_render = False
   d = mayavi.mlab.pipeline.add_dataset(ug)
   if iso_surface == False:
      iso = mayavi.mlab.pipeline.surface(d)
   else:
      ptdata = mayavi.mlab.pipeline.cell_to_point_data(d)
      iso = mayavi.mlab.pipeline.iso_surface(ptdata, contours=[1e-15,1e-14,1e-12], opacity=0.3)

   engine = mayavi.mlab.get_engine()

   from mayavi.modules.axes import Axes 
   axes = Axes()
   axes.name = 'Axes'
   axes.axes.fly_mode = 'none'
   axes.axes.number_of_labels = 8
   axes.axes.font_factor = 0.5
   #module_manager = self.__module_manager()
   # Add the label / marker:
   engine.add_filter( axes )
   from mayavi.modules.outline import Outline
   outline = Outline()
   outline.name = 'Outline'
   engine.add_filter( outline )

   mayavi.mlab.show()

def generate_velocity_grid( vlsvReader, cellid, iso_surface=False ):
   '''Generates a velocity grid from a given spatial cell id
      :param cellid:           The spatial cell's ID
      :param iso_surface:      If true, plots the iso surface
   '''
   # Create nodes
   # Get velocity blocks and avgs:
   blocksAndAvgs = vlsvReader.read_blocks(cellid)
   generate_custom_velocity_grid( vlsvReader, blocksAndAvgs, iso_surface )

