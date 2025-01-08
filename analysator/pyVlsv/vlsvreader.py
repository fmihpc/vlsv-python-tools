#s 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2024 University of Helsinki
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
import sys
import re
import numbers

import vlsvvariables
from reduction import datareducers,multipopdatareducers,data_operators,v5reducers,multipopv5reducers,deprecated_datareducers
try:
   from collections.abc import Iterable
except ImportError:
   from collections import Iterable
from collections import OrderedDict
from vlsvwriter import VlsvWriter
from variable import get_data
import warnings
import time
from interpolator_amr import AMRInterpolator, supported_amr_interpolators
from operator import itemgetter


interp_method_aliases = {"trilinear":"linear"}

class PicklableFile(object):
   def __init__(self, fileobj):
      self.fileobj = fileobj

   def __getattr__(self, key):
      return getattr(self.fileobj, key)

   def __getstate__(self):
      ret = self.__dict__.copy()
      ret['_file_name'] = self.fileobj.name
      ret['_file_mode'] = self.fileobj.mode
      if self.fileobj.closed:
         ret['_file_pos'] = 0
      else:
         ret['_file_pos'] = self.fileobj.tell()
      del ret['fileobj']
      return ret

   def __setstate__(self, dict):
      self.fileobj = open(dict['_file_name'], dict['_file_mode'])
      self.fileobj.seek(dict['_file_pos'])
      del dict['_file_name']
      del dict['_file_mode']
      del dict['_file_pos']
      self.__dict__.update(dict)

def dict_keys_exist(dictionary, query_keys, prune_unique=False):
   if query_keys.shape[0] == 0:
      return np.array([],dtype=bool)
   # this helps quite a lot... for some cases.
   if prune_unique:
      unique_keys, indices = np.unique(query_keys, axis=0, return_inverse=True)

      # these are all about the same...
      # if (unique_keys.ndim == 1):
      mask = np.array([k in dictionary.keys() for k in unique_keys],dtype=bool)
      # else: # and this isn't worth it?
      #    mask = np.array([tuple(k) in dictionary.keys() for k in unique_keys],dtype=bool)

      # mask = np.empty(query_keys.shape, dtype=bool)
      # for i,k in enumerate(query_keys):
      #    mask[i] = [k in dictionary.keys() for k in query_keys],dtype=bool)

      # mask = np.array(list(map(lambda c: c in dictionary.keys(),query_keys)), dtype=bool)

      # dlambda = np.frompyfunc(lambda c: c in dictionary.keys(),1,1)
      # mask = np.array(dlambda(query_keys),dtype=bool)
      mask = mask[indices]
   else:
      mask = np.array([k in dictionary.keys() for k in query_keys],dtype=bool)

   return mask

def fsGlobalIdToGlobalIndex(globalids, bbox):
   indices = np.zeros((globalids.shape[0],3),dtype=np.int64)

   stride = np.int64(1)
   for d in [0,1,2]:
      indices[:,d] = (globalids // stride) % bbox[d]
      stride *= np.int64(bbox[d])

   return indices

# Read in the global ids and indices for FsGrid cells, returns
# min and max corners of the fsGrid chunk by rank
def fsReadGlobalIdsPerRank(reader):
   numWritingRanks = reader.read_parameter("numWritingRanks")
   rawData = reader.read(tag="MESH", name="fsgrid")
   bbox = reader.read(tag="MESH_BBOX", mesh="fsgrid")
   sizes = reader.read(tag="MESH_DOMAIN_SIZES", mesh="fsgrid")

   currentOffset = np.int64(0)
   rankIds = {}
   rankIndices = {}
   for i in range(0,numWritingRanks):
      taskIds = rawData[currentOffset:int(currentOffset+sizes[i,0])]
      rankIds[i] = np.array([min(taskIds),max(taskIds)])
      rankIndices[i] = fsGlobalIdToGlobalIndex(rankIds[i], bbox)
      currentOffset += int(sizes[i,0])
      
   return rankIds, rankIndices

# Read global ID bboxes per rank and figure out the decomposition from
# the number of unique corner coordinates per dimension
def fsDecompositionFromGlobalIds(reader):
   ids, inds = fsReadGlobalIdsPerRank(reader)
   lows = np.array([inds[i][0] for i in inds.keys()])
   xs = np.unique(lows[:,0])
   ys = np.unique(lows[:,1])
   zs = np.unique(lows[:,2])
   return [xs.size, ys.size, zs.size]

def map_vg_onto_fg_loop(arr, vg_cellids, refined_ids_start, refined_ids_end):
   #arr = np.zeros(sz, dtype=np.int64) + 1000000000 # big number to catch errors in the latter code, 0 is not good for that

   for i in range(vg_cellids.shape[0]):
      arr[refined_ids_start[i,0]:refined_ids_end[i,0],
                           refined_ids_start[i,1]:refined_ids_end[i,1],
                           refined_ids_start[i,2]:refined_ids_end[i,2]] = i
   return arr

class VlsvReader(object):
   ''' Class for reading VLSV files
   ''' 


   ''' Meshinfo is an information container for multiple meshes.
       Implemented as an empty class.
   '''
   class MeshInfo:
      pass

   file_name=""
   def __del__(self):
      if (hasattr(self, "__fptr")) and self.__fptr is not None:
         self.__fptr.close()

   def __init__(self, file_name, fsGridDecomposition=None):
      ''' Initializes the vlsv file (opens the file, reads the file footer and reads in some parameters)

          :param file_name:     Name of the vlsv file
          :param fsGridDecomposition: Either None or a len-3 list of ints.
                                       List (length 3): Use this as the decomposition directly. Product needs to match numWritingRanks.
      '''
      # Make sure the path is set in file name: 
      file_name = os.path.abspath(file_name)

      self.file_name = file_name
      try:
         self.__fptr = PicklableFile(open(self.file_name,"rb"))
      except FileNotFoundError as e:
         logging.info("File not found: " + self.file_name)
         raise e
      
      self.__xml_root = ET.fromstring("<VLSV></VLSV>")
      self.__fileindex_for_cellid={}

      self.__max_spatial_amr_level = -1
      self.__fsGridDecomposition = fsGridDecomposition

      self.use_dict_for_blocks = False
      self.__fileindex_for_cellid_blocks={} # [0] is index, [1] is blockcount
      self.__cells_with_blocks = {} # per-pop
      self.__blocks_per_cell = {} # per-pop
      self.__blocks_per_cell_offsets = {} # per-pop
      self.__order_for_cellid_blocks = {} # per-pop
      self.__vg_indexes_on_fg = np.array([]) # SEE: map_vg_onto_fg(self)

      self.variable_cache = {} # {(varname, operator):data}

      self.__available_reducers = set() # Set of strings of datareducer names
      self.__unavailable_reducers = set() # Set of strings of datareducer names
      self.__current_reducer_tree_nodes = set() # Set of strings of datareducer names

      self.__read_xml_footer()
                              # vertex-indices is a 3-tuple of integers
      self.__dual_cells = {(0,0,0):(1,1,1,1,1,1,1,1)} # vertex-indices : 8-tuple of cellids at each corner (for x for y for z)
      self.__dual_bboxes = {} # vertex-indices : 6-list of (xmin, ymin, zmin, xmax, ymax, zmax) for the bounding box of each dual cell
      self.__cell_vertices = {} # cellid : varying-length tuple of vertex-indices - includes hanging nodes!
      self.__cell_corner_vertices = {} # cellid : varying-length tuple of vertex-indices - no hanging nodes!
      self.__cell_neighbours = {} # cellid : set of cellids (all neighbors sharing a vertex)
      self.__cell_duals = {} # cellid : tuple of vertex-indices that span this cell
      self.__regular_neighbor_cache = {} # cellid-of-low-corner : (8,) np.array of cellids)

      # Check if the file is using new or old vlsv format
      # Read parameters (Note: Reading the spatial cell locations and
      # storing them will anyway take the most time and memory):

      meshName="SpatialGrid"
      bbox = self.read(tag="MESH_BBOX", mesh=meshName)
      if bbox is None:
          try:
              #read in older vlsv files where the mesh is defined with parameters
              self.__xcells = (int)(self.read_parameter("xcells_ini"))
              self.__ycells = (int)(self.read_parameter("ycells_ini"))
              self.__zcells = (int)(self.read_parameter("zcells_ini"))
              self.__xblock_size = 1
              self.__yblock_size = 1
              self.__zblock_size = 1
              self.__xmin = self.read_parameter("xmin")
              self.__ymin = self.read_parameter("ymin")
              self.__zmin = self.read_parameter("zmin")
              self.__xmax = self.read_parameter("xmax")
              self.__ymax = self.read_parameter("ymax")
              self.__zmax = self.read_parameter("zmax")
          except:
              # Apparently, SpatialGrid doesn't even exist in this file (because it is, for example an ionosphere test output)
              # Fill in dummy values.
              self.__xcells = 1
              self.__ycells = 1
              self.__zcells = 1
              self.__xblock_size = 1
              self.__yblock_size = 1
              self.__zblock_size = 1
              self.__xmin = 0
              self.__ymin = 0
              self.__zmin = 0
              self.__xmax = 1
              self.__ymax = 1
              self.__zmax = 1


      else:
         #new style vlsv file with 
         nodeCoordinatesX = self.read(tag="MESH_NODE_CRDS_X", mesh=meshName)   
         nodeCoordinatesY = self.read(tag="MESH_NODE_CRDS_Y", mesh=meshName)   
         nodeCoordinatesZ = self.read(tag="MESH_NODE_CRDS_Z", mesh=meshName)   
         self.__xcells = bbox[0]
         self.__ycells = bbox[1]
         self.__zcells = bbox[2]
         self.__xblock_size = bbox[3]
         self.__yblock_size = bbox[4]
         self.__zblock_size = bbox[5]
         self.__xmin = nodeCoordinatesX[0]
         self.__ymin = nodeCoordinatesY[0]
         self.__zmin = nodeCoordinatesZ[0]
         self.__xmax = nodeCoordinatesX[-1]
         self.__ymax = nodeCoordinatesY[-1]
         self.__zmax = nodeCoordinatesZ[-1]

      self.__dx = (self.__xmax - self.__xmin) / (float)(self.__xcells)
      self.__dy = (self.__ymax - self.__ymin) / (float)(self.__ycells)
      self.__dz = (self.__zmax - self.__zmin) / (float)(self.__zcells)

      self.__meshes = {}

      # Iterate through the XML tree, find all populations
      # (identified by their BLOCKIDS tag)
      self.active_populations=[]
      for child in self.__xml_root:
          if child.tag == "BLOCKIDS":
              if "name" in child.attrib:
                  popname = child.attrib["name"] 
              else:
                  popname = "avgs"

              # Create a new (empty) MeshInfo-object for this population
              pop = self.MeshInfo()
              
              # Update list of active populations
              if not popname in self.active_populations: self.active_populations.append(popname)

              bbox = self.read(tag="MESH_BBOX", mesh=popname)
              if bbox is None:
                 if self.read_parameter("vxblocks_ini") is not None:
                    #read in older vlsv files where the mesh is defined with
                    #parameters (only one possible)
                    pop.__vxblocks = (int)(self.read_parameter("vxblocks_ini"))
                    pop.__vyblocks = (int)(self.read_parameter("vyblocks_ini"))
                    pop.__vzblocks = (int)(self.read_parameter("vzblocks_ini"))
                    pop.__vxblock_size = 4 # Old files will always have WID=4, newer files read it from bbox
                    pop.__vyblock_size = 4
                    pop.__vzblock_size = 4
                    pop.__vxmin = self.read_parameter("vxmin")
                    pop.__vymin = self.read_parameter("vymin")
                    pop.__vzmin = self.read_parameter("vzmin")
                    pop.__vxmax = self.read_parameter("vxmax")
                    pop.__vymax = self.read_parameter("vymax")
                    pop.__vzmax = self.read_parameter("vzmax")
                    # Velocity cell lengths
                    pop.__dvx = ((pop.__vxmax - pop.__vxmin) / (float)(pop.__vxblocks)) / (float)(pop.__vxblock_size)
                    pop.__dvy = ((pop.__vymax - pop.__vymin) / (float)(pop.__vyblocks)) / (float)(pop.__vyblock_size)
                    pop.__dvz = ((pop.__vzmax - pop.__vzmin) / (float)(pop.__vzblocks)) / (float)(pop.__vzblock_size)

                 else:
                    #no velocity space in this file, e.g., file not written by Vlasiator 
                    pop.__vxblocks = 0
                    pop.__vyblocks = 0
                    pop.__vzblocks = 0
                    pop.__vxblock_size = 4
                    pop.__vyblock_size = 4
                    pop.__vzblock_size = 4
                    pop.__vxmin = 0
                    pop.__vymin = 0
                    pop.__vzmin = 0
                    pop.__vxmax = 0
                    pop.__vymax = 0
                    pop.__vzmax = 0
                    # Velocity cell lengths
                    pop.__dvx = 1
                    pop.__dvy = 1
                    pop.__dvz = 1

              else:
                 #new style vlsv file with bounding box
                 nodeCoordinatesX = self.read(tag="MESH_NODE_CRDS_X", mesh=popname)   
                 nodeCoordinatesY = self.read(tag="MESH_NODE_CRDS_Y", mesh=popname)   
                 nodeCoordinatesZ = self.read(tag="MESH_NODE_CRDS_Z", mesh=popname)   
                 pop.__vxblocks = bbox[0]
                 pop.__vyblocks = bbox[1]
                 pop.__vzblocks = bbox[2]
                 pop.__vxblock_size = bbox[3]
                 pop.__vyblock_size = bbox[4]
                 pop.__vzblock_size = bbox[5]
                 pop.__vxmin = nodeCoordinatesX[0]
                 pop.__vymin = nodeCoordinatesY[0]
                 pop.__vzmin = nodeCoordinatesZ[0]
                 pop.__vxmax = nodeCoordinatesX[-1]
                 pop.__vymax = nodeCoordinatesY[-1]
                 pop.__vzmax = nodeCoordinatesZ[-1]
                 # Velocity cell lengths
                 pop.__dvx = ((pop.__vxmax - pop.__vxmin) / (float)(pop.__vxblocks)) / (float)(pop.__vxblock_size)
                 pop.__dvy = ((pop.__vymax - pop.__vymin) / (float)(pop.__vyblocks)) / (float)(pop.__vyblock_size)
                 pop.__dvz = ((pop.__vzmax - pop.__vzmin) / (float)(pop.__vzblocks)) / (float)(pop.__vzblock_size)

              self.__meshes[popname]=pop
              if not os.getenv('PTNONINTERACTIVE'):
                 logging.info("Found population " + popname)
              
              # Precipitation energy bins
              i = 0
              energybins = []
              binexists = True
              while binexists:
                 binexists = self.check_parameter("{}_PrecipitationCentreEnergy{}".format(popname,i))
                 if binexists:
                    binvalue = self.read_parameter("{}_PrecipitationCentreEnergy{}".format(popname,i))
                    energybins.append(binvalue)
                 i = i + 1
              if i > 1:
                 pop.__precipitation_centre_energy = np.asarray(energybins)
                 vlsvvariables.speciesprecipitationenergybins[popname] = energybins

      vlsvvariables.cellsize = self.__dx

      if self.check_parameter("j_per_b_modifier"):
         vlsvvariables.J_per_B_modifier = self.read_parameter("j_per_b_modifier")

      self.__fptr.close()


   def __read_xml_footer(self):
      ''' Reads in the XML footer of the VLSV file and store all the content
      ''' 
      #(endianness,) = struct.unpack("c", fptr.read(1))
      if self.__fptr.closed:
         fptr = open(self.file_name,"rb")
      else:
         fptr = self.__fptr
      # Eight first bytes indicate whether the system is big_endianness or something else
      endianness_offset = 8
      fptr.seek(endianness_offset)
      # Read 8 bytes as unsigned long long (uint64_t in this case) after endianness, this tells the offset of the XML file.
      uint64_byte_amount = 8
      (offset,) = struct.unpack("Q", fptr.read(uint64_byte_amount))
      # Move to the xml offset
      fptr.seek(offset)
      # Read the xml data
      xml_data = bytearray()
      for chunk in iter(lambda: fptr.read(4096), ''):
          if chunk == b'':
             break
          xml_data += chunk
      # Read the xml as string
      (xml_string,) = struct.unpack("%ds" % len(xml_data), xml_data)
      # Input the xml data into xml_root
      self.__xml_root = ET.fromstring(xml_string)
      fptr.close()

   def __read_fileindex_for_cellid(self):
      """ Read in the cell ids and create an internal dictionary to give the index of an arbitrary cellID
      """
      if not self.__fileindex_for_cellid == {}:
         return
      
      cellids=self.read(mesh="SpatialGrid",name="CellID", tag="VARIABLE")

      #Check if it is not iterable. If it is a scale then make it a list
      if(not isinstance(cellids, Iterable)):
         cellids=[ cellids ]
      # self.__fileindex_for_cellid = {cellid:index for index,cellid in enumerate(cellids)}
      for index,cellid in enumerate(cellids):
         self.__fileindex_for_cellid[cellid] = index

   def __read_blocks(self, cellid, pop="proton"):
      ''' Read raw velocity block data from the open file.
      
      :param cellid: Cell ID of the cell whose velocity blocks are read
      :returns: A numpy array with block ids and their data
      '''
      if self.use_dict_for_blocks: #deprecated version
         if not pop in self.__fileindex_for_cellid_blocks:
            self.__set_cell_offset_and_blocks(pop)

         if( (cellid in self.__fileindex_for_cellid_blocks[pop]) == False ):
            # Cell id has no blocks
            return []
         offset = self.__fileindex_for_cellid_blocks[pop][cellid][0]
         num_of_blocks = self.__fileindex_for_cellid_blocks[pop][cellid][1]
      else:
         # Uses arrays (much faster to initialize)
         if not pop in self.__cells_with_blocks:
            self.__set_cell_offset_and_blocks_nodict(pop) 
         # Check that cells has vspace
         try:
            cells_with_blocks_index = self.__order_for_cellid_blocks[pop][cellid]
         except:
            logging.info("Cell does not have velocity distribution")
            return []
         offset = self.__blocks_per_cell_offsets[pop][cells_with_blocks_index]
         num_of_blocks = self.__blocks_per_cell[pop][cells_with_blocks_index]


      if self.__fptr.closed:
         fptr = open(self.file_name,"rb")
      else:
         fptr = self.__fptr

      # Read in avgs and velocity cell ids:
      for child in self.__xml_root:
         # Read in block values
         if ("name" in child.attrib) and (child.attrib["name"] == pop) and (child.tag == "BLOCKVARIABLE"):
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            #array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]

            # Navigate to the correct position
            offset_avgs = int(offset * vector_size * element_size + ast.literal_eval(child.text))
#            for i in range(0, cells_with_blocks_index[0]):
#               offset_avgs += blocks_per_cell[i]*vector_size*element_size

            fptr.seek(offset_avgs)
            if datatype == "float" and element_size == 4:
               data_avgs = np.fromfile(fptr, dtype = np.float32, count = vector_size*num_of_blocks)
            if datatype == "float" and element_size == 8:
               data_avgs = np.fromfile(fptr, dtype = np.float64, count = vector_size*num_of_blocks)
            data_avgs = data_avgs.reshape(num_of_blocks, vector_size)

         # Read in block coordinates:
         # (note the special treatment in case the population is named 'avgs'
         if (pop == 'avgs' or ("name" in child.attrib) and (child.attrib["name"] == pop)) and (child.tag == "BLOCKIDS"):
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            #array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]

            offset_block_ids = int(offset * vector_size * element_size + ast.literal_eval(child.text))

            fptr.seek(offset_block_ids)
            if datatype == "uint" and element_size == 4:
               data_block_ids = np.fromfile(fptr, dtype = np.uint32, count = vector_size*num_of_blocks)
            elif datatype == "uint" and element_size == 8:
               data_block_ids = np.fromfile(fptr, dtype = np.uint64, count = vector_size*num_of_blocks)
            else:
               logging.info("Error! Bad block id data!")
               logging.info("Data type: " + datatype + ", element size: " + str(element_size))
               return

            data_block_ids = np.reshape(data_block_ids, (len(data_block_ids),) )

      fptr.close()

      # Check to make sure the sizes match (just some extra debugging)
      logging.info("data_avgs = " + str(data_avgs) + ", data_block_ids = " + str(data_block_ids))
      if len(data_avgs) != len(data_block_ids):
         logging.info("BAD DATA SIZES")

      return [data_block_ids, data_avgs]

   def __set_cell_offset_and_blocks(self, pop="proton"):
      ''' Read blocks per cell and the offset in the velocity space arrays for
          every cell with blocks into a private dictionary.
          Deprecated in favor of below version.
      '''
      logging.info("Getting offsets for population " + pop)
      if pop in self.__fileindex_for_cellid_blocks:
         # There's stuff already saved into the dictionary, don't save it again
         return
      #these two arrays are in the same order: 
      #list of cells for which dist function is saved
      cells_with_blocks = self.read(mesh="SpatialGrid",tag="CELLSWITHBLOCKS", name=pop)
      #number of blocks in each cell for which data is stored
      blocks_per_cell = self.read(mesh="SpatialGrid",tag="BLOCKSPERCELL", name=pop)

      # Navigate to the correct position:
      from copy import copy
      offset = 0
      #self.__fileindex_for_cellid_blocks[pop] = {}
      self.__fileindex_for_cellid_blocks[pop] = dict.fromkeys(cells_with_blocks) # should be faster but negligible difference
      for i in range(0, len(cells_with_blocks)):
         self.__fileindex_for_cellid_blocks[pop][cells_with_blocks[i]] = [copy(offset), copy(blocks_per_cell[i])]
         offset += blocks_per_cell[i]

   def __set_cell_offset_and_blocks_nodict(self, pop="proton"):
      ''' Read blocks per cell and the offset in the velocity space arrays for every cell with blocks.
          Stores them in arrays. Creates a private dictionary with addressing to the array.
          This method should be faster than the above function.
      '''
      if pop in self.__cells_with_blocks:
         # There's stuff already saved into the dictionary, don't save it again
         return

      logging.info("Getting offsets for population " + pop)

      self.__cells_with_blocks[pop] = np.atleast_1d(self.read(mesh="SpatialGrid",tag="CELLSWITHBLOCKS", name=pop))
      self.__blocks_per_cell[pop] = np.atleast_1d(self.read(mesh="SpatialGrid",tag="BLOCKSPERCELL", name=pop))

      self.__blocks_per_cell_offsets[pop] = np.empty(len(self.__cells_with_blocks[pop]))
      self.__blocks_per_cell_offsets[pop][0] = 0.0
      self.__blocks_per_cell_offsets[pop][1:] = np.cumsum(self.__blocks_per_cell[pop][:-1])
      self.__order_for_cellid_blocks[pop] = {}
      for index,cellid in enumerate(self.__cells_with_blocks[pop]):
         self.__order_for_cellid_blocks[pop][cellid]=index

   def __check_datareducer(self, name, reducer):

      reducer_ok = True
      if name in self.__available_reducers: return True
      if name in self.__unavailable_reducers: return False
      if name in self.__current_reducer_tree_nodes: raise RuntimeError("Cyclical datareduction deteced with "+name+", this is weird and undefined!")

      self.__current_reducer_tree_nodes.add(name)

      for var in reducer.variables:
         if len(var) > 3 and var[0:3] == "pop":
            in_vars = False
            for pop in self.active_populations:
               popvar = pop+var[3:]
               if popvar in self.__available_reducers:
                  in_vars = True
               elif popvar in self.__unavailable_reducers:
                  in_vars = False
               else:
                  in_vars = self.check_variable(popvar)

               # print(popvar," is in vars: ",in_vars)
               if in_vars:
                  self.__available_reducers.add(popvar)
                  break
               else:
                  self.__unavailable_reducers.add(popvar)
         else:
            in_vars = self.check_variable(var)

         # reducer_ok = reducer_ok and in_vars
         if in_vars:
            continue

         in_reducers = ((var in datareducers.keys()) or
                        (var in multipopdatareducers.keys()) or
                        (var in v5reducers.keys()) or
                        (var in multipopv5reducers.keys()))
         if in_reducers:
            reducer = None
            for reducer_reg in [datareducers, multipopdatareducers, v5reducers, multipopv5reducers]:
               try:
                  reducer = reducer_reg[var]
               except:
                  pass

            reducer_ok = reducer_ok and self.__check_datareducer(var, reducer)
         else: # Not in variables not in datareducers, break
            reducer_ok = False
            break

         if not reducer_ok: break

      if reducer_ok: self.__available_reducers.add(name)
      else: self.__unavailable_reducers.add(name)

      return reducer_ok


   def list(self, parameter=True, variable=True, mesh=False, datareducer=False, operator=False, other=False):
      ''' Print out a description of the content of the file. Useful
         for interactive usage. Default is to list parameters and variables, query selection can be adjusted with keywords:

         Default and supported keywords:

         parameter=True 
         variable=True
         mesh=False
         datareducer=False 
         operator=False 
         other=False
      '''
      if parameter:
         print("tag = PARAMETER")
         for child in self.__xml_root:
            if child.tag == "PARAMETER" and "name" in child.attrib:
               print("   ", child.attrib["name"])
      if variable:
         print("tag = VARIABLE")
         for child in self.__xml_root:
            if child.tag == "VARIABLE" and "name" in child.attrib:
               print("   ", child.attrib["name"])
      if mesh:
         print("tag = MESH")
         for child in self.__xml_root:
            if child.tag == "MESH" and "name" in child.attrib:
               print("   ", child.attrib["name"])
      if datareducer:
         print("Datareducers (replace leading pop with a population name):")

         reducer_max_len = 0
         units_max_len = 0
         for reducer_reg in [datareducers, multipopdatareducers, v5reducers, multipopv5reducers]:
            for k in reducer_reg.keys():
               reducer_max_len = max(reducer_max_len, len(k))
               units_max_len = max(units_max_len, len(reducer_reg[k].units))


         for reducer_reg in [datareducers, multipopdatareducers, v5reducers, multipopv5reducers]:
            for name, reducer in reducer_reg.items():
               self.__current_reducer_tree_nodes.clear()
               if self.__check_datareducer(name,reducer):
                  print((("    %-"+str(reducer_max_len)+"s")% name) + ("\t%-"+str(units_max_len+2)+"s")%("["+reducer.units+"]")+"\t based on " + str(reducer_reg[name].variables))

      if operator:
         print("Data operators:")
         for name in data_operators:
            if type(name) is str:
               if not name.isdigit():
                  print("   ",name)
      if other:
         print("Other:")
         for child in self.__xml_root:
            if child.tag != "PARAMETER" and child.tag != "VARIABLE" and child.tag != "MESH":
               print("    tag = ", child.tag, " mesh = ", child.attrib["mesh"])

   def check_parameter( self, name ):
      ''' Checks if a given parameter is in the vlsv reader

          :param name:             Name of the parameter
          :returns:                True if the parameter is in the vlsv file, false if not

          .. note:: This should be used for checking if a parameter exists, e.g. for different Vlasiator versions and time output

          .. code-block:: python

             # Example usage:
             vlsvReader = pt.vlsvfile.VlsvReader("test.vlsv")
             if vlsvReader.check_parameter( "time" ):
                time = vlsvReader.read_parameter("time")
             elif vlsvReader.check_parameter( "t" ):
                time = vlsvReader.read_parameter("t")
             else:
                time = None
      '''
      for child in self.__xml_root:
         if child.tag == "PARAMETER" and "name" in child.attrib:
            if child.attrib["name"].lower() == name.lower():
               return True
      return False

   def check_variable( self, name ):
      ''' Checks if a given variable is in the vlsv reader

          :param name:             Name of the variable
          :returns:                True if the variable is in the vlsv file, false if not

          .. note:: This should be used for checking if a variable exists in case a function behaves differently for ex. if B vector is in the vlsv and if not

          .. code-block:: python

             # Example usage:
             vlsvReader = pt.vlsvfile.VlsvReader("test.vlsv")
             if vlsvReader.check_variable( "B" ):
                # Variable can be plotted
                plot_B()
             else:
                # Variable not in the vlsv file
                plot_B_vol()
      '''
      for child in self.__xml_root:
         if child.tag == "VARIABLE" and "name" in child.attrib:
            if child.attrib["name"].lower() == name.lower():
               return True
      return False

   def check_population( self, popname ):
      ''' Checks if a given population is in the vlsv file

          :param name:             Name of the population
          :returns:                True if the population is in the vlsv file, false if not

          .. code-block:: python

             # Example usage:
             vlsvReader = pt.vlsvfile.VlsvReader("test.vlsv")
             if vlsvReader.check_population( "avgs" ):
                plot_population('avgs')
             else:
                if vlsvReader.check_population( "proton" ):
                   # File is newer with proton population
                   plot_population('proton')
      '''
      blockidsexist = False
      foundpop = False
      for child in self.__xml_root:
         if child.tag == "BLOCKIDS":
            if "name" in child.attrib:
               if popname.lower() == child.attrib["name"].lower():
                  foundpop = True
            else:
               blockidsexist = True
      if blockidsexist:
         for child in self.__xml_root:
            if child.tag == "BLOCKVARIABLE":
               if "name" in child.attrib:
                  if popname.lower() == child.attrib["name"].lower(): # avgs
                     foundpop = True
      return foundpop

   def get_all_variables( self ):
      ''' Returns all variables in the vlsv reader and the data reducer
          :returns:                List of variable is in the vlsv file
          .. code-block:: python

             # Example usage:
             vlsvReader = pt.vlsvfile.VlsvReader("test.vlsv")
             vars = vlsvReader.get_variables()
      '''
      varlist = [];
      for child in self.__xml_root:
         if child.tag == "VARIABLE" and "name" in child.attrib:
            varlist.append(child.attrib["name"])
      return varlist

   def get_cellid_locations(self):
      ''' Returns a dictionary with cell id as the key and the index of the cell id as the value. The index is used to locate the cell id's values in the arrays that this reader returns
      '''
      # if len( self.__fileindex_for_cellid ) == 0:
      self.__read_fileindex_for_cellid()
      return self.__fileindex_for_cellid

   def print_version(self):
      '''
      Prints version information from VLSV file.
      TAG is hardcoded to VERSION

      :returns True if version is found otherwise returns False
      '''
      import sys
      tag="VERSION"
      # Seek for requested data in VLSV file
      for child in self.__xml_root:
         if child.tag != tag:
            continue
         if child.tag == tag:
            # Found the requested data entry in the file
            array_size = ast.literal_eval(child.attrib["arraysize"])
            variable_offset = ast.literal_eval(child.text)

            if self.__fptr.closed:
               fptr = open(self.file_name,"rb")
            else:
               fptr = self.__fptr
         
            fptr.seek(variable_offset)
            info = fptr.read(array_size).decode("utf-8")

            print("Version Info for " + self.file_name)
            print(info)
            return True

      #if we end up here the file does not contain any version info
      print("File ",self.file_name," contains no version information")
      return False
  
   def get_config_string(self):
      '''
      Gets config information from VLSV file.
      TAG is hardcoded to CONFIG

      :returns configuration file string if config is found otherwise returns None
      '''
      tag="CONFIG"
      # Seek for requested data in VLSV file
      for child in self.__xml_root:
         if child.tag != tag:
            continue
         if child.tag == tag:
            # Found the requested data entry in the file
            array_size = ast.literal_eval(child.attrib["arraysize"])
            variable_offset = ast.literal_eval(child.text)

            if self.__fptr.closed:
               fptr = open(self.file_name,"rb")
            else:
               fptr = self.__fptr

            fptr.seek(variable_offset)
            configuration = fptr.read(array_size).decode("utf-8")

            return configuration

      #if we end up here the file does not contain any config info
      return None

   def get_config(self):
      '''
      Gets config information from VLSV file

      :returns a nested dictionary of dictionaries,
        where keys (str) are config file group headings (appearing in '[]')
        and values are dictionaries which contain (lists of) strings 

      If the same heading/parameter pair appears >once in the config file,
      the different values are appended to the list .

      EXAMPLE:
      if the config contains these lines:
      
         [proton_precipitation]
         nChannels = 9

      then the following returns ['9']:
      vlsvReader.get_config()['proton_precipitation']['nChannels']
      '''
      confstring = self.get_config_string()
      if confstring is None:
         return None
      fa = re.findall(r'\[\w+\]|\w+ = \S+', confstring)
      heading = ''
      output = {heading:{}}

      for i, sfa in enumerate(fa):
         if (sfa[0] == '[') and (sfa[-1] == ']'):
            heading = sfa[1:-1]
            output[heading] = {}
         else:
            var_name = sfa.split('=')[0].strip()
            var_value = sfa.split('=')[1].strip()
            if var_name in output[heading]:
               # when the same parameter is assigned a value multiple times
               output[heading][var_name].append(var_value)
            else:
               output[heading][var_name] = [var_value]

      return output

   def print_config(self):
      '''
      Prints config information from VLSV file.

      :returns True if config is found otherwise returns False
      '''
      config_string = self.get_config_string()
      if config_string is not None:
         logging.info(config_string)
         return True
      else:
         #if we end up here the file does not contain any config info
         logging.info("File ",self.file_name," contains no config information")
         return False

   def read_variable_vectorsize(self, name):

      if name.startswith('fg_'):
          mesh = "fsgrid"
      elif name.startswith('ig_'):
          mesh = "ionosphere"
      else:
          mesh = "SpatialGrid"

      return self.read_attribute(name=name, mesh=mesh,attribute="vectorsize", tag="VARIABLE")

   def read_attribute(self, name="", mesh="", attribute="", tag=""):
      ''' Read data from the open vlsv file. 
      
      :param name: Name of the data array
      :param tag:  Tag of the data array.
      :param mesh: Mesh for the data array
      :param operator: Datareduction operator. "pass" does no operation on data.
      :param cellids:  If -1 then all data is read. If nonzero then only the vector
                       for the specified cell id or cellids is read
      :returns: numpy array with the data

      .. seealso:: :func:`read_variable` :func:`read_variable_info`
      '''
      if tag == "" and name == "":
         logging.info("Bad (empty) arguments at VlsvReader.read")
         raise ValueError()

      # Force lowercase name for internal checks
      name = name.lower()       

      # Seek for requested data in VLSV file
      for child in self.__xml_root:
         if tag != "":
            if child.tag != tag:
               continue
         if name != "":
            if "name" in child.attrib and child.attrib["name"].lower() != name:
               continue
         if mesh != "":
            if "mesh" in child.attrib and child.attrib["mesh"] != mesh:
               continue
         if child.tag == tag:
            # Found the requested data entry in the file
            return ast.literal_eval(child.attrib[attribute])
         
      raise ValueError("Variable or attribute not found")


   def read(self, name="", tag="", mesh="", operator="pass", cellids=-1):
      ''' Read data from the open vlsv file. 
      
      :param name: Name of the data array
      :param tag:  Tag of the data array.
      :param mesh: Mesh for the data array
      :param operator: Datareduction operator. "pass" does no operation on data.
      :param cellids:  If -1 then all data is read. If nonzero then only the vector
                       for the specified cell id or cellids is read
      :returns: numpy array with the data

      .. seealso:: :func:`read_variable` :func:`read_variable_info`
      '''
      if tag == "" and name == "":
         logging.info("Bad (empty) arguments at VlsvReader.read")
         raise ValueError()

      if mesh == None:
         mesh = ''

      # Force lowercase name for internal checks
      name = name.lower()

      if tag == "VARIABLE":
         if (name,operator) in self.variable_cache.keys():
            return self.read_variable_from_cache(name, cellids, operator)

      if (len( self.__fileindex_for_cellid ) == 0):
         # Do we need to construct the cellid index?
         if isinstance(cellids, numbers.Number): # single or all cells
            if cellids >= 0: # single cell
               self.__read_fileindex_for_cellid()
         else: # list of cellids
            self.__read_fileindex_for_cellid()
               
      if self.__fptr.closed:
         fptr = open(self.file_name,"rb")
      else:
         fptr = self.__fptr


         
      # Get population and variable names from data array name 
      if '/' in name:
         popname = name.split('/')[0]
         if popname in self.active_populations:
            varname = name.split('/',1)[1]
         else:
            popname = 'pop'
            varname = name
      else:
         popname = 'pop'
         varname = name

      # Seek for requested data in VLSV file
      for child in self.__xml_root:
         if tag != "":
            if child.tag != tag:
               continue
         # Verify that any requested name or mesh matches those of the data
         if name != "":
            if not "name" in child.attrib:
               continue
            if child.attrib["name"].lower() != name:
               continue
         if mesh != "":
            if not "mesh" in child.attrib:
               continue
            if child.attrib["mesh"] != mesh:
               continue
         if child.tag == tag:
            # Found the requested data entry in the file
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]
            variable_offset = ast.literal_eval(child.text)

            # Define efficient method to read data in
            try: # try-except to see how many cellids were given
               lencellids=len(cellids) 
               # Read multiple specified cells
               # If we're reading a large amount of single cells, it'll be faster to just read all
               # data from the file system and sort through it. For the CSC disk system, this
               # becomes more efficient for over ca. 5000 cellids.
               arraydata = []
               if lencellids>5000: 
                  result_size = len(cellids)
                  read_size = array_size
                  read_offsets = [0]
               else: # Read multiple cell ids one-by-one
                  result_size = len(cellids)
                  read_size = 1
                  read_offsets = [self.__fileindex_for_cellid[cid]*element_size*vector_size for cid in cellids]
            except: # single cell or all cells
               if cellids < 0: # -1, read all cells
                  result_size = array_size
                  read_size = array_size
                  read_offsets = [0]
               else: # single cell id
                  result_size = 1
                  read_size = 1
                  read_offsets = [self.__fileindex_for_cellid[cellids]*element_size*vector_size]
                  
            for r_offset in read_offsets:
               use_offset = int(variable_offset + r_offset)
               fptr.seek(use_offset)
               if datatype == "float" and element_size == 4:
                  data = np.fromfile(fptr, dtype = np.float32, count=vector_size*read_size)
               if datatype == "float" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.float64, count=vector_size*read_size)
               if datatype == "int" and element_size == 4:
                  data = np.fromfile(fptr, dtype=np.int32, count=vector_size*read_size)
               if datatype == "int" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.int64, count=vector_size*read_size)
               if datatype == "uint" and element_size == 4:
                  data = np.fromfile(fptr, dtype=np.uint32, count=vector_size*read_size)
               if datatype == "uint" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.uint64, count=vector_size*read_size)
               if len(read_offsets)!=1:
                  arraydata.append(data)
            
            if len(read_offsets)==1 and result_size<read_size:
               # Many single cell id's requested
               # Pick the elements corresponding to the requested cells
               for cid in cellids:
                  append_offset = self.__fileindex_for_cellid[cid]*vector_size
                  arraydata.append(data[append_offset:append_offset+vector_size])
               data = np.squeeze(np.array(arraydata))

            if len(read_offsets)!=1:
               # Not-so-many single cell id's requested
               data = np.squeeze(np.array(arraydata))

            fptr.close()

            if vector_size > 1:
               data=data.reshape(result_size, vector_size)
            
            # If variable vector size is 1, and requested magnitude, change it to "absolute"
            if vector_size == 1 and operator=="magnitude":
               logging.info("Data variable with vector size 1: Changed magnitude operation to absolute")
               operator="absolute"

            if result_size == 1:
               return data_operators[operator](data[0])
            else:
               return data_operators[operator](data)
      
      # Check which set of datareducers to use
      if '/' in name and popname in self.active_populations:
         checkname = 'pop/'+varname
      else:
         checkname = varname

      if checkname in deprecated_datareducers.keys():
         raise ValueError(deprecated_datareducers[checkname] )

      if varname[0:3]=="vg_" or varname[0:3]=="ig_":
         reducer_reg = v5reducers
         reducer_multipop = multipopv5reducers
      else:
         reducer_reg = datareducers
         reducer_multipop = multipopdatareducers

         
            
      # If this is a variable that can be summed over the populations (Ex. rho, PTensorDiagonal, ...)
      if hasattr(self, 'active_populations') and len(self.active_populations) > 0 and self.check_variable(self.active_populations[0]+'/'+name):
         tmp_vars = []
         for pname in self.active_populations:
            vlsvvariables.activepopulation = pname
            tmp_vars.append( self.read( pname+'/'+name, tag, mesh, "pass", cellids ) )
         return data_operators[operator](data_operators["sum"](tmp_vars))

      # Check if the name is in datareducers
      if name in reducer_reg:
         reducer = reducer_reg[name]
         # Read the necessary variables:

         # If variable vector size is 1, and requested magnitude, change it to "absolute"
         if reducer.vector_size == 1 and operator=="magnitude":
            logging.info("Data reducer with vector size 1: Changed magnitude operation to absolute")
            operator="absolute"

         # Return the output of the datareducer
         if reducer.useVspace and not reducer.useReader:
            actualcellids = self.read(mesh="SpatialGrid", name="CellID", tag="VARIABLE", operator=operator, cellids=cellids)
            output = np.zeros(len(actualcellids))
            index = 0
            for singlecellid in actualcellids:
               velocity_cell_data = self.read_velocity_cells(singlecellid)
               # Get cells:
               vcellids = list(velocity_cell_data.keys())
               # Get coordinates:
               velocity_coordinates = self.get_velocity_cell_coordinates(vcellids)
               tmp_vars = []
               for i in np.atleast_1d(reducer.variables):
                  tmp_vars.append( self.read( i, tag, mesh, "pass", singlecellid ) )
               output[index] = reducer.operation( tmp_vars , velocity_cell_data, velocity_coordinates )
               index+=1
               logging.info(str(index)+"/"+str(len(actualcellids)))
            
            if reducer.useReader:
               logging.info("Combined useVspace and useReader reducers not implemented!")
               raise NotImplementedError()
            else:
               return data_operators[operator](output)
         else:
            tmp_vars = []
            for i in np.atleast_1d(reducer.variables):
               tmp_vars.append( self.read( i, tag, mesh, "pass", cellids ) )
            if reducer.useReader:
               return data_operators[operator](reducer.operation( tmp_vars, self ))
            else:
               return data_operators[operator](reducer.operation( tmp_vars ))

      # Check if the name is in multipop datareducers
      if 'pop/'+varname in reducer_multipop:

         reducer = reducer_multipop['pop/'+varname]
         # If variable vector size is 1, and requested magnitude, change it to "absolute"
         if reducer.vector_size == 1 and operator=="magnitude":
            logging.info("Data reducer with vector size 1: Changed magnitude operation to absolute")
            operator="absolute"

         if reducer.useVspace:
            logging.info("Error: useVspace flag is not implemented for multipop datareducers!") 
            return

         # sum over populations
         if popname=='pop':
            # Read the necessary variables:
            tmp_vars = []
            for pname in self.active_populations:
               vlsvvariables.activepopulation = pname
               tmp_vars.append( self.read( pname+'/'+varname, tag, mesh, "pass", cellids ) )
            return data_operators[operator](data_operators["sum"](tmp_vars))
         else:
            vlsvvariables.activepopulation = popname

         # Read the necessary variables:
         tmp_vars = []
         for i in np.atleast_1d(reducer.variables):
            if '/' not in i:
               tmp_vars.append( self.read( i, tag, mesh, "pass", cellids ) )
            else:
               tvar = i.split('/',1)[1]
               tmp_vars.append( self.read( popname+'/'+tvar, tag, mesh, "pass", cellids ) )
         return data_operators[operator](reducer.operation( tmp_vars ))

      fptr.close()
      if name!="":
         raise ValueError("Error: variable "+name+"/"+tag+"/"+mesh+"/"+operator+" not found in .vlsv file or in data reducers!") 



   def read_metadata(self, name="", tag="", mesh=""):
      ''' Read variable metadata from the open vlsv file. 
      
      :param name: Name of the data array
      :param tag:  Tag of the data array.
      :param mesh: Mesh for the data array
      :returns: four strings:
                the unit of the variable as a regular string
                the unit of the variable as a LaTeX-formatted string
                the description of the variable as a LaTeX-formatted string
                the conversion factor to SI units as a string                  
      '''

      if tag == "" and name == "":
         logging.info("Bad arguments at read")

      if self.__fptr.closed:
         fptr = open(self.file_name,"rb")
      else:
         fptr = self.__fptr

      # Force lowercase name for internal checks
      name = name.lower()
      
      # Seek for requested data in VLSV file
      for child in self.__xml_root:         
         if tag != "":
            if child.tag != tag:
               continue
         if name != "":
            if "name" in child.attrib and child.attrib["name"].lower() != name:
               continue
         if mesh != "":
            if "mesh" in child.attrib and child.attrib["mesh"] != mesh:
               continue
         if not "name" in child.attrib:
            continue
         # Found the requested data entry in the file
         try:
            unit = child.attrib["unit"]
         except:
            unit = ""
         try:
            unitLaTeX = child.attrib["unitLaTeX"]
         except:
            unitLaTeX = ""
         try:
            variableLaTeX = child.attrib["variableLaTeX"]
         except:
            variableLaTeX = ""
         try:
            unitConversion = child.attrib["unitConversion"] 
         except:
            unitConversion = ""
         return unit, unitLaTeX, variableLaTeX, unitConversion
            
      if name!="":
         logging.info("Error: variable "+name+"/"+tag+"/"+mesh+" not found in .vlsv file!" )
      fptr.close()
      return -1
         

   def read_interpolated_fsgrid_variable(self, name, coordinates, operator="pass",periodic=[True,True,True], method="linear"):
      ''' Read a linearly interpolated FSgrid variable value from the open vlsv file. Feel free to vectorize!
      Note that this does not account for varying centerings of fsgrid data.
      Arguments:
      :param name: Name of the (FSgrid) variable
      :param coords: Coordinates from which to read data 
      :param periodic: Periodicity of the system. Default is periodic in all dimension
      :param operator: Datareduction operator. "pass" does no operation on data
      :returns: numpy array with the data

      .. seealso:: :func:`read` :func:`read_variable_info`
      '''

      if method != "Linear":
         raise NotImplementedError("interpolation method "+method+" not implemented for read_interpolated_fsgrid_variable, only linear supported so far.")
      
      warnings.warn("read_interpolated_fsgrid_variable: face- vs. edge- centered variables not accounted for!")

      if name[0:3] != 'fg_':
         raise ValueError("Interpolation of FsGrid called on non-FsGrid data; exiting.")
      
      if (len(periodic)!=3):
         raise ValueError("Periodic must be a list of 3 booleans.")

      #First off let's fetch the data and some meta
      fg_data=self.read_fsgrid_variable( name,operator=operator)
      fg_size=self.get_fsgrid_mesh_size()
      nx,ny,nz=fg_size
      extents=self.get_fsgrid_mesh_extent()
      xmin,ymin,zmin,xmax,ymax,zmax=extents
      dx=abs((xmax-xmin)/nx)
      dy=abs((ymax-ymin)/ny)
      dz=abs((zmax-zmin)/nz)

      def getFsGridIndices(indices):
         ''' 
         Returns indices based on boundary conditions
         '''
         ind=-1*np.ones((3))
         for c,index in enumerate(indices):
            #Non periodic case
            if ((index<0 or index>fg_size[c]-1) and not periodic[c]):
               # Returns False, interpolateSingle converts that to nans
               warnings.warn("Requested fsgrid index for interpolation outside simulation domain.", UserWarning)
               return False
             # Here we are either periodic or (not periodic and inside the domain)
            if (index >= fg_size[c] or index <0):
                ind[c] = index%fg_size[c]
            elif (index>=0 and index<=fg_size[c]-1):
                ind[c]=index
            else:
                #If we end up here then something is really wrong
                raise ValueError("FsGrid interpolation ran into a failure and could not locate all neighbors.","Indices in question= ",indices)

         return int(ind[0]),int(ind[1]),int(ind[2]) 



      def interpolateSingle(r):
         ''' 
         Simple trilinear routine for interpolating fsGrid quantities 
         at arbitrary coordinates r.
         Inputs:
             r: array of coordinates at which to perform the interpolation. 
                Example: r=[x,y,z] in meters
         Outputs:
             Numpy array with interpolated data at r. Can be scalar or vector.
         '''
         import sys
         if (len(r) !=3 ):
            raise ValueError("Interpolation cannot be performed. Exiting")

         x,y,z=r
         xl=int(np.floor((x-xmin)/dx))
         yl=int(np.floor((y-ymin)/dy))
         zl=int(np.floor((z-zmin)/dz))
    
         #Normalize distances in a unit cube 
         xd=(x-xmin)/dx - xl
         yd=(y-ymin)/dy - yl
         zd=(z-zmin)/dz - zl
       
         # Calculate Neighbors' Weights
         w=np.zeros(8)
         w[0] = (1.0-xd)*(1.0-yd)*(1.0-zd)
         w[1] = (xd)*(1.0-yd)*(1.0-zd)
         w[2] = (1.0-xd)*(yd)*(1.0-zd)
         w[3] = (xd)*(yd)*(1.0-zd)
         w[4] = (1.0-xd)*(1.0-yd)*(zd)
         w[5] = (xd)*(1.0-yd)*(zd)
         w[6] = (1.0-xd)*(yd)*(zd)
         w[7] = (xd)*(yd)*(zd)

         retval = np.zeros_like(fg_data[0,0,0])
         for k in [0,1]:
            for j in [0,1]:
               for i in [0,1]:
                  retind=getFsGridIndices([xl+i,yl+j,zl+k])
                  if (not retind): retval.fill(np.nan) # outside of a non periodic domain
                  retval += w[4*k+2*j+i]*fg_data[retind]

         return retval

      ret=[]
      for r in coordinates:
         ret.append(interpolateSingle(r))
      return np.asarray(ret)

   def read_interpolated_ionosphere_variable(self, name, coordinates, operator="pass", method="linear"):
      ''' Read a linearly interpolated ionosphere variable value from the open vlsv file.
      Arguments:
      :param name: Name of the (ionosphere) variable
      :param coords: Coordinates (x,y,z) from which to read data 
      :param operator: Datareduction operator. "pass" does no operation on data
      :param method: Interpolation method. Not implemented; barycentric interp would fall under linear.
      :returns: numpy array with the data

      .. seealso:: :func:`read` :func:`read_variable_info`
      '''

      # At this stage, this function has not yet been implemented -- logging.info a warning and exit
      raise NotImplementedError('Interpolation of ionosphere variables has not yet been implemented; exiting.')

   # These are the 8 cells that span the upper corner vertex on a regular grid
   def get_vg_regular_interp_neighbors(self, cellids):

      
      len_cellids = np.atleast_1d(cellids).shape[0]
      cellid_neighbors = np.zeros((len_cellids, 8))

      in_cache = dict_keys_exist(self.__regular_neighbor_cache, cellids)
      
      if(np.any(in_cache)):
         cellid_neighbors[in_cache,:] = np.array(itemgetter(*cellids[in_cache])(self.__regular_neighbor_cache), dtype=np.int64)
      n_not_in_cache = np.sum(~in_cache)

      if n_not_in_cache > 0:
         offsets = np.zeros((8,3), dtype=np.int32)
         ii = 0
         for x in [0,1]:
            for y in [0,1]:
               for z  in [0,1]:
                  offsets[ii,:] = np.array((x,y,z), dtype=np.int32)
                  ii+=1

         cellids_rep = np.reshape(np.repeat(np.atleast_2d(cellids[~in_cache]), 8, axis=1).T,n_not_in_cache*8)
         offsets = np.tile(offsets, (n_not_in_cache, 1))
         cellid_neighbors_new = self.get_cell_neighbor(cellids_rep, offsets, [True,True,True], prune_uniques=False)
         cellid_neighbors_new = cellid_neighbors_new.reshape((-1,8))
         self.__regular_neighbor_cache.update( {c:cellid_neighbors_new[i,:] for i,c in enumerate(cellids[~in_cache])})
         cellid_neighbors[~in_cache,:] = cellid_neighbors_new
      
      return cellid_neighbors

   def read_interpolated_variable(self, name, coords, operator="pass",periodic=[True, True, True], method="linear"):
      ''' Read a linearly interpolated variable value from the open vlsv file.
      Arguments:
      :param name: Name of the variable
      :param coords: Coordinates from which to read data 
      :param periodic: Periodicity of the system. Default is periodic in all dimension
      :param operator: Datareduction operator. "pass" does no operation on data
      :param method: Interpolation method, default "linear", options: ["nearest", "linear"]
                     
      :returns: numpy array with the data

      .. seealso:: :func:`read` :func:`read_variable_info`
      '''



      if (len(periodic)!=3):
            raise ValueError("Periodic must be a list of 3 booleans.")

      if method.lower() in interp_method_aliases.keys():
         warnings.warn("Updated alias " +method+" -> "+interp_method_aliases[method.lower()])
         method = interp_method_aliases[method.lower()]

      # First test whether the requested variable is on the FSgrid or ionosphre, and redirect to the dedicated function if needed
      if name[0:3] == 'fg_':
         return self.read_interpolated_fsgrid_variable(name, coords, operator, periodic, method)
      if name[0:3] == 'ig_':
         return self.read_interpolated_ionosphere_variable(name, coords, operator, periodic, method)

      # case vg

      coordinates = get_data(coords)
      coordinates = np.array(coordinates)
      
      stack = True
      if len(np.shape(coordinates)) == 1:
         stack = False
         coordinates = np.atleast_2d(coordinates)

      # Check one value for the length
      test_variable = self.read_variable(name,cellids=[1],operator=operator)
      if isinstance(test_variable,np.ma.core.MaskedConstant):
         value_length=1
      elif isinstance(test_variable, Iterable):
         value_length=len(test_variable)
      else:
         value_length=1

      ncoords = coordinates.shape[0]
      if(coordinates.shape[1] != 3):
         raise IndexError("Coordinates are required to be three-dimensional (coords.shape[1]==3 or convertible to such))")
      closest_cell_ids = self.get_cellid(coordinates)

      if method.lower() == "nearest":
         final_values = self.read_variable(name, cellids=closest_cell_ids, operator=operator)
         if stack:
            return final_values.squeeze()
         else:
            if value_length == 1:
               return final_values.squeeze()[()] # The only special case to return a scalar instead of an array
            else:
               return final_values.squeeze()
      elif method.lower() != 'linear':
         raise NotImplementedError(method + ' is not a valid interpolation method')

      batch_closest_cell_coordinates=self.get_cell_coordinates(closest_cell_ids)
      
      offsets = np.zeros(coordinates.shape,dtype=np.int32)
      offsets[coordinates <= batch_closest_cell_coordinates] = -1

      # This we cannot cache, unfortunately. np.unique on (closest_cell_ids,offsets) helps a bit in some cases.
      lower_cell_ids = self.get_cell_neighbor(closest_cell_ids, offsets, periodic, prune_uniques=True)

      lower_cell_ids_unique, unique_cell_indices = np.unique(lower_cell_ids, return_inverse=True)
      ngbrvalues=np.full((len(lower_cell_ids_unique)*2*2*2,value_length),np.nan)

      cellid_neighbors = np.zeros((lower_cell_ids_unique.shape[0],8))
      cellid_neighbors[lower_cell_ids_unique != 0, :] = self.get_vg_regular_interp_neighbors(lower_cell_ids_unique[lower_cell_ids_unique != 0])
      cellid_neighbors = cellid_neighbors.reshape((-1))
   

      lower_cell_coordinatess=self.get_cell_coordinates(lower_cell_ids_unique)

      upper_cell_ids = cellid_neighbors[7::8]
      upper_cell_coordinatess=self.get_cell_coordinates(upper_cell_ids)

      scaled_coordinates = np.zeros_like(coordinates)
      nonperiodic = lower_cell_coordinatess != upper_cell_coordinatess
      nonperiodic_all = nonperiodic[unique_cell_indices]
      scaled_coordinates[nonperiodic_all] = (coordinates[nonperiodic_all] - lower_cell_coordinatess[unique_cell_indices][nonperiodic_all])/(upper_cell_coordinatess[unique_cell_indices][nonperiodic_all] - lower_cell_coordinatess[unique_cell_indices][nonperiodic_all])

      # neighbour tuples can also overlap, so choose only unique ones here as well for reading (should be moved to read_variable)
      cellids_neighbors_unique, indices = np.unique(cellid_neighbors[cellid_neighbors!=0], return_inverse=True)

      # Read the corner values for required cells and apply repeats to match the initial coordinates
      if value_length > 1:
         read_vals = self.read_variable(name, cellids=cellids_neighbors_unique, operator=operator)
         ngbrvalues[cellid_neighbors!=0,:] = read_vals[indices,:]
         # ngbrvalues[cellid_neighbors!=0,:] = self.read_variable(name, cellids=cellid_neighbors[cellid_neighbors!=0], operator=operator)
      else:
         read_vals = self.read_variable(name, cellids=cellids_neighbors_unique, operator=operator)[:,np.newaxis]
         ngbrvalues[cellid_neighbors!=0,:] = read_vals[indices,:]
         # ngbrvalues[cellid_neighbors!=0,:] = self.read_variable(name, cellids=cellid_neighbors[cellid_neighbors!=0], operator=operator)[:,np.newaxis]
      # ngbrvalues = np.reshape(ngbrvalues, (ncoords,2,2,2,value_length))
      ngbrvalues = np.reshape(ngbrvalues, (len(lower_cell_ids_unique),2,2,2,value_length))

      ngbrvalues = ngbrvalues[unique_cell_indices,...]
      
      c2ds=ngbrvalues[:,0,:,:,:]* (1- scaled_coordinates[:,0][:,np.newaxis,np.newaxis,np.newaxis]) +  ngbrvalues[:,1,:,:,:]*scaled_coordinates[:,0][:,np.newaxis,np.newaxis,np.newaxis]
      c1ds = c2ds[:,0,:,:]*(1 - scaled_coordinates[:,1][:,np.newaxis,np.newaxis]) + c2ds[:,1,:,:] * scaled_coordinates[:,1][:,np.newaxis,np.newaxis]
      final_values=c1ds[:,0,:] * (1 - scaled_coordinates[:,2][:,np.newaxis]) + c1ds[:,1,:] * scaled_coordinates[:,2][:,np.newaxis]

      if np.any(cellid_neighbors==0):
         warnings.warn("Coordinate in interpolation out of domain, output contains nans",UserWarning)

      
      refs0 = np.reshape(self.get_amr_level(cellid_neighbors),(-1,8))
      if np.any(np.any(refs0 != refs0[:,0][:,np.newaxis],axis =1)):
         irregs = np.any(refs0 != refs0[:,0][:,np.newaxis],axis =1)[unique_cell_indices]
         final_values[irregs,:] = np.reshape(self.read_interpolated_variable_irregular(name, coordinates[irregs], operator, method=method.lower()),(-1,value_length))
         # warnings.warn("Interpolation across refinement levels. Results are now better, but some discontinuitues might appear. If that bothers, try the read_interpolated_variable_irregular variant directly.",UserWarning)

      if stack:
         return final_values.squeeze()
      else:
         if value_length == 1:
            return final_values.squeeze()[()] # The only special case to return a scalar instead of an array
         else:
            return final_values.squeeze()

   def get_duals(self,cids):
      ''' Get the union of dual cells that cover each of CellIDs in cids.
          Assumes all required duals are defined! TODO handling of missing duals, do not call separately.

      :returns: Dict of vertex-indices v (3-tuple) : 8-tuple of cellids (corners of dual cells indexed by v)
      '''

      fo = {c: self.__cell_duals[c] for c in cids}
      vset = set()
      vset.union(*fo.values())
      return {v: self.__dual_cells[v] for v in vset}


   def read_interpolated_variable_irregular(self, name, coords, operator="pass",periodic=[True, True, True],
                                            method="linear",
                                            methodargs={
                                             "rbf":{"neighbors":64},
                                             "delaunay":{"qhull_options":"QJ"}
                                             }):
      ''' Read a linearly interpolated variable value from the open vlsv file.
      Arguments:

      :param name:         Name of the variable

      :param coords:       Coordinates from which to read data 

      :param periodic:     Periodicity of the system. Default is periodic in all dimension

      :param operator:     Datareduction operator. "pass" does no operation on data


      :param method:       Method for interpolation, default "linear" ("nearest", "rbf, "delaunay")

      :param methodargs:   Dict of dicts to pass kwargs to interpolators. Default values for "rbf", "delaunay";

                           see scipy.interpolate.RBFInterpolator for rbf and scipy.interpolate.LinearNDInterpolator for delaunay
      :returns: numpy array with the data

      .. seealso:: :func:`read` :func:`read_variable_info`
      '''

      stack = True
      if coords.ndim == 1:
         stack = False
         coords = coords[np.newaxis,:]

      if (len(periodic)!=3):
            raise ValueError("Periodic must be a list of 3 booleans.")

      if method.lower() in interp_method_aliases.keys():
         warnings.warn("Updated alias " +method+" -> "+interp_method_aliases[method])
         method = interp_method_aliases[method]
      

      # First test whether the requested variable is on the FSgrid or ionosphre, and redirect to the dedicated function if needed
      if name[0:3] == 'fg_':
         return self.read_interpolated_fsgrid_variable(name, coords, operator, periodic, method)
      if name[0:3] == 'ig_':
         return self.read_interpolated_ionosphere_variable(name, coords, operator, periodic, method)

      # Default case: AMR grid

      coordinates = get_data(coords)
      coordinates = np.array(coordinates)

      
      ncoords = coordinates.shape[0]
      if(coordinates.shape[1] != 3):
         raise IndexError("Coordinates are required to be three-dimensional (coords.shape[1]==3 or convertible to such))")
      cellids = self.get_cellid(coordinates)

      if method == "nearest":
         # Check one value for the length
         test_variable = self.read_variable(name,cellids=[1],operator=operator)
         if isinstance(test_variable,np.ma.core.MaskedConstant):
            value_length=1
         elif isinstance(test_variable, Iterable):
            value_length=len(test_variable)
         else:
            value_length=1
         final_values = self.read_variable(name, cellids=cellids, operator=operator)
         if stack:
            return final_values.squeeze()
         else:
            if value_length == 1:
               return final_values.squeeze()[()] # The only special case to return a scalar instead of an array
            else:
               return final_values.squeeze()      # Other methods
      else:
         if method.lower() not in supported_amr_interpolators:
            raise NotImplementedError(method + ' is not a valid interpolation method for AMR grids')

      containing_cells = np.unique(cellids)
      self.build_duals(containing_cells)
      duals = self.get_duals(containing_cells)

      cells_set = set()
      cells_set = cells_set.union(*duals.values())

      cells_set.discard(0)
      intp_wrapper = AMRInterpolator(self,cellids=np.array(list(cells_set)))
      intp = intp_wrapper.get_interpolator(name,operator, coords, method=method.lower(), methodargs=methodargs)
      
      final_values = intp(coords, cellids=cellids)[:,np.newaxis]

      if stack:
         return final_values.squeeze() # this will be an array as long as this is still a multi-cell codepath!
      else:
         final_value = final_values[0,:]
         if len(final_value)==1:
            return final_value[0]
         else:
            return final_value


   def read_fsgrid_variable_cellid(self, name, cellids=-1, operator="pass"):
      ''' Reads fsgrid variables from the open vlsv file.
       Arguments:
       :param name:     Name of the variable
       :param cellids:  SpatialGrid cellids for which to fetch data. Default: return full fsgrid data
       :param operator: Datareduction operator. "pass" does no operation on data
       :returns: *ordered* list of numpy arrays with the data

       ... seealso:: :func:`read_fsgrid_variable`
       '''
      var = self.read_fsgrid_variable(name, operator=operator)
      if cellids == -1:
         return var
      else:
         return [self.downsample_fsgrid_subarray(cid, var) for cid in cellids]

   def read_fsgrid_variable(self, name, operator="pass"):
       ''' Reads fsgrid variables from the open vlsv file.
       Arguments:
       :param name: Name of the variable
       :param operator: Datareduction operator. "pass" does no operation on data
       :returns: *ordered* numpy array with the data

       ... seealso:: :func:`read_variable`
       '''

       # Get fsgrid domain size (this can differ from vlasov grid size if refined)
       bbox = self.read(tag="MESH_BBOX", mesh="fsgrid")
       bbox = np.int32(bbox)

       # Read the raw array data
       rawData = self.read(mesh='fsgrid', name=name, tag="VARIABLE", operator=operator)

       # Determine fsgrid domain decomposition
       numWritingRanks = self.read_parameter("numWritingRanks")
       if len(rawData.shape) > 1:
         orderedData = np.zeros([bbox[0],bbox[1],bbox[2],rawData.shape[1]])
       else:
         orderedData = np.zeros([bbox[0],bbox[1],bbox[2]])

       def calcLocalStart(globalCells, ntasks, my_n):
           n_per_task = globalCells//ntasks
           remainder = globalCells%ntasks
           if my_n < remainder:
               return my_n * (n_per_task+1)
           else:
               return my_n * n_per_task + remainder
       def calcLocalSize(globalCells, ntasks, my_n):
           n_per_task = globalCells//ntasks
           remainder = globalCells%ntasks
           if my_n < remainder:
               return n_per_task+1;
           else:
               return n_per_task

       currentOffset = 0;
       if self.__fsGridDecomposition is None:
         self.__fsGridDecomposition = self.read(tag="MESH_DECOMPOSITION",mesh='fsgrid')
         if self.__fsGridDecomposition is not None:
            logging.info("Found FsGrid decomposition from vlsv file: " + str(self.__fsGridDecomposition))
         else:
            logging.info("Did not find FsGrid decomposition from vlsv file.")
       
       # If decomposition is None even after reading, we need to calculate it:
       if self.__fsGridDecomposition is None:
          logging.info("Calculating fsGrid decomposition from the file")
          self.__fsGridDecomposition = fsDecompositionFromGlobalIds(self)
          logging.info("Computed FsGrid decomposition to be: " + str(self.__fsGridDecomposition))
       else:
          # Decomposition is a list (or fail assertions below) - use it instead
          pass
          
       assert len(self.__fsGridDecomposition) == 3, "Manual FSGRID decomposition should have three elements, but is "+str(self.__fsGridDecomposition)
       assert np.prod(self.__fsGridDecomposition) == numWritingRanks, "Manual FSGRID decomposition should have a product of numWritingRanks ("+str(numWritingRanks)+"), but is " + str(np.prod(self.__fsGridDecomposition)) + " for decomposition "+str(self.__fsGridDecomposition)
               
          

       for i in range(0,numWritingRanks):
           x = (i // self.__fsGridDecomposition[2]) // self.__fsGridDecomposition[1]
           y = (i // self.__fsGridDecomposition[2]) % self.__fsGridDecomposition[1]
           z = i % self.__fsGridDecomposition[2]
 	   
           thatTasksSize = [calcLocalSize(bbox[0], self.__fsGridDecomposition[0], x), \
                            calcLocalSize(bbox[1], self.__fsGridDecomposition[1], y), \
                            calcLocalSize(bbox[2], self.__fsGridDecomposition[2], z)]
           thatTasksStart = [calcLocalStart(bbox[0], self.__fsGridDecomposition[0], x), \
                             calcLocalStart(bbox[1], self.__fsGridDecomposition[1], y), \
                             calcLocalStart(bbox[2], self.__fsGridDecomposition[2], z)]
           
           thatTasksEnd = np.array(thatTasksStart) + np.array(thatTasksSize)
           totalSize = int(thatTasksSize[0]*thatTasksSize[1]*thatTasksSize[2])
           # Extract datacube of that task... 
           if len(rawData.shape) > 1:
               thatTasksData = rawData[currentOffset:currentOffset+totalSize,:]
               thatTasksData = thatTasksData.reshape([thatTasksSize[0],thatTasksSize[1],thatTasksSize[2],rawData.shape[1]],order='F')

               # ... and put it into place 
               orderedData[thatTasksStart[0]:thatTasksEnd[0],thatTasksStart[1]:thatTasksEnd[1],thatTasksStart[2]:thatTasksEnd[2],:] = thatTasksData
           else:
               # Special case for scalar data
               thatTasksData = rawData[currentOffset:currentOffset+totalSize]
               if (len(thatTasksData)>0):
                  thatTasksData = thatTasksData.reshape([thatTasksSize[0],thatTasksSize[1],thatTasksSize[2]], order='F')

                  # ... and put it into place
                  orderedData[thatTasksStart[0]:thatTasksEnd[0],thatTasksStart[1]:thatTasksEnd[1],thatTasksStart[2]:thatTasksEnd[2]] = thatTasksData

           currentOffset += totalSize

       return np.squeeze(orderedData)

   def read_fg_variable_as_volumetric(self, name, centering=None, operator="pass"):
      fgdata = self.read_fsgrid_variable(name, operator)

      fssize=list(self.get_fsgrid_mesh_size())
      if 1 in fssize:
         #expand to have a singleton dimension for a reduced dim - lets roll happen with ease
         singletons = [i for i, sz in enumerate(fssize) if sz == 1]
         for dim in singletons:
            fgdata=np.expand_dims(fgdata, dim)
      return self.fg_array_to_volumetric(fgdata, name, centering=centering, operator=operator)

   def fg_array_to_volumetric(self, fgdata, name, centering=None,operator="pass"):
      celldata = np.zeros_like(fgdata)
      known_centerings = {"fg_b":"face", "fg_e":"edge"}
      if centering is None:
         try:
            centering = known_centerings[name.lower()]
         except KeyError:
            logging.info("A variable ("+name+") with unknown centering! Aborting.")
            return False
         
      #vector variable
      if fgdata.shape[-1] == 3:
         if centering=="face":
            celldata[:,:,:,0] = (fgdata[:,:,:,0] + np.roll(fgdata[:,:,:,0],-1, 0))/2.0
            celldata[:,:,:,1] = (fgdata[:,:,:,1] + np.roll(fgdata[:,:,:,1],-1, 1))/2.0
            celldata[:,:,:,2] = (fgdata[:,:,:,2] + np.roll(fgdata[:,:,:,2],-1, 2))/2.0
            # Use Leo's reconstuction for fg_b instead
         elif centering=="edge":
            celldata[:,:,:,0] = (fgdata[:,:,:,0] + np.roll(fgdata[:,:,:,0],-1, 1) + np.roll(fgdata[:,:,:,0],-1, 2) + np.roll(fgdata[:,:,:,0],-1, (1,2)))/4.0
            celldata[:,:,:,1] = (fgdata[:,:,:,1] + np.roll(fgdata[:,:,:,1],-1, 0) + np.roll(fgdata[:,:,:,1],-1, 2) + np.roll(fgdata[:,:,:,1],-1, (0,2)))/4.0
            celldata[:,:,:,2] = (fgdata[:,:,:,2] + np.roll(fgdata[:,:,:,2],-1, 0) + np.roll(fgdata[:,:,:,2],-1, 1) + np.roll(fgdata[:,:,:,2],-1, (0,1)))/4.0
         else:
            logging.info("Unknown centering ('" +centering+ "')! Aborting.")
            return False
      else:
         logging.info("A scalar variable! I don't know what to do with this! Aborting.")
         return False
      return celldata

   def read_ionosphere_variable(self, name, operator="pass"):
       ''' Reads fsgrid variables from the open vlsv file.
       Arguments:
       :param name: Name of the variable
       :param operator: Datareduction operator. "pass" does no operation on data
       :returns: numpy array with the data in node order

       ... seealso:: :func:`read_variable`
       '''
       # Read the raw array data
       rawData = self.read(mesh='ionosphere', name=name, tag="VARIABLE", operator=operator)

       return rawData


   def read_variable_from_cache(self, name, cellids, operator):
      ''' Read variable from cache instead of the vlsv file.
         :param name: Name of the variable
         :param cellids: a value of -1 reads all data
         :param operator: Datareduction operator. "pass" does no operation on data
         :returns: numpy array with the data, same format as read_variable

         .. seealso:: :func:`read_variable`
      '''

      var_data = self.variable_cache[(name,operator)]
      if var_data.ndim == 2:
         value_len = var_data.shape[1]
      else:
         value_len = 1
         
      if isinstance(cellids, numbers.Number):
         if cellids == -1:
            return var_data
         else:
            return var_data[self.__fileindex_for_cellid[cellids]]
      else:
         if(len(cellids) > 0):
            indices = np.array(itemgetter(*cellids)(self.__fileindex_for_cellid),dtype=np.int64)
         else:
            indices = np.array([],dtype=np.int64)
         if value_len == 1:
            return var_data[indices]
         else:
            return var_data[indices,:]
         
         
   def read_variable_to_cache(self, name, operator="pass"):
      ''' Read variable from vlsv file to cache, for the whole grid and after applying
          operator.
          :param name: Name of the variable (or datareducer)
          :param operator: Datareduction operator. "pass" does no operation on data.

      '''

      # add data to dict, use a tuple of (name,operator) as the key [tuples are immutable and hashable]
      self.variable_cache[(name,operator)] = self.read_variable(name, cellids=-1,operator=operator)
      # Also initialize the fileindex dict at the same go because it is very likely something you want to have for accessing cached values
      self.__read_fileindex_for_cellid()

   def read_variable(self, name, cellids=-1,operator="pass"):
      ''' Read variables from the open vlsv file. 
      Arguments:
      :param name: Name of the variable
      :param cellids: a value of -1 reads all data
      :param operator: Datareduction operator. "pass" does no operation on data
      :returns: numpy array with the data

      .. seealso:: :func:`read` :func:`read_variable_info`
      '''
      cellids = get_data(cellids)

      # Wrapper, check if requesting an fsgrid variable
      if (self.check_variable(name) and (name.lower()[0:3]=="fg_")):
         if not cellids == -1:
            logging.info("Warning, CellID requests not supported for FSgrid variables! Aborting.")
            return False
         return self.read_fsgrid_variable(name=name, operator=operator)

      #if(self.check_variable(name) and (name.lower()[0:3]=="ig_")):
      if name.lower()[0:3]=="ig_":
         if not cellids == -1:
            logging.info("Warning, CellID requests not supported for ionosphere variables! Aborting.")
            return False
         return self.read_ionosphere_variable(name=name, operator=operator)
      
      if((name,operator) in self.variable_cache.keys()):
         return self.read_variable_from_cache(name,cellids,operator)

      # Passes the list of cell id's onwards - optimization for reading is done in the lower level read() method
      return self.read(mesh="SpatialGrid", name=name, tag="VARIABLE", operator=operator, cellids=cellids)

   def read_variable_info(self, name, cellids=-1, operator="pass"):
      ''' Read variables from the open vlsv file and input the data into VariableInfo

      :param name: Name of the variable
      :param cellids: a value of -1 reads all data
      :param operator: Datareduction operator. "pass" does no operation on data
      :returns: numpy array with the data

      .. seealso:: :func:`read_variable`
      '''
      from variable import VariableInfo

      # Force lowercase
      name = name.lower()

      # Get population and variable names from data array name 
      if '/' in name:
         popname = name.split('/')[0]
         if popname in self.active_populations:
            varname = name.split('/',1)[1]
         else:
            popname = 'pop'
            varname = name
      else:
         popname = "pop"
         varname = name

      # Check which set of datareducers to use
      if varname[0:3]=="vg_" or varname[0:3]=="ig_":
         reducer_reg = v5reducers
         reducer_multipop = multipopv5reducers
      else:
         reducer_reg = datareducers
         reducer_multipop = multipopdatareducers

      if (self.check_variable(name) and (varname[0:3]=="vg_" or varname[0:3]=="fg_" or varname[0:3]=="ig_")):
         # For Vlasiator 5 vlsv files, metadata is included
         units, latexunits, latex, conversion = self.read_metadata(name=name)
         # Correction for early version incorrect number density (extra backslash)
         if latex[0:3]==r"$\n":
            latex = r"$n"+latex[3:]
      elif (self.check_variable(name) and (name in vlsvvariables.unitsdict)):
         units = vlsvvariables.unitsdict[name]
         latex = vlsvvariables.latexdict[name]
         latexunits = vlsvvariables.latexunitsdict[name]            
      elif name in reducer_reg:
         units = reducer_reg[name].units
         latex = reducer_reg[name].latex
         latexunits = reducer_reg[name].latexunits
      elif 'pop/'+varname in reducer_multipop:
         poplatex='i'
         if popname in vlsvvariables.speciesdict:
            poplatex = vlsvvariables.speciesdict[popname]
         units = reducer_multipop['pop/'+varname].units
         latex = (reducer_multipop['pop/'+varname].latex).replace('REPLACEPOP',poplatex)
         latexunits = reducer_multipop['pop/'+varname].latexunits
      elif varname in vlsvvariables.unitsdict:
         poplatex='i'
         if popname in vlsvvariables.speciesdict:
            poplatex = vlsvvariables.speciesdict[popname]
         units = vlsvvariables.unitsdict[varname]
         latex = vlsvvariables.latexdictmultipop[varname].replace('REPLACEPOP',poplatex)
         latexunits = vlsvvariables.latexunitsdict[varname]
      else:
         units = ""
         latex = r""+name.replace(r"_",r"\_")
         latexunits = ""

      if name.startswith('fg_'):
          data = self.read_fsgrid_variable(name=name, operator=operator)
      elif name.startswith('ig_'):
          data = self.read_ionosphere_variable(name=name, operator=operator)
      else:
          data = self.read_variable(name=name, operator=operator, cellids=cellids)

      if operator != "pass":
         if operator=="magnitude":
            latex = r"$|$"+latex+r"$|$"
         else:
            latex = latex+r"${_{"+operator+r"}}$"
         return VariableInfo(data_array=data, name=name + "_" + operator, units=units, latex=latex, latexunits=latexunits)
      else:
         return VariableInfo(data_array=data, name=name, units=units, latex=latex, latexunits=latexunits)


   def get_max_refinement_level(self):
      ''' Returns the maximum refinement level of the AMR
      '''
      if self.__max_spatial_amr_level < 0:
         # Read the file index for cellid
         cellids=self.read(mesh="SpatialGrid",name="CellID", tag="VARIABLE")
         maxcellid = np.amax([cellids])

         AMR_count = 0
         while (maxcellid > 0):
            maxcellid -= 2**(3*(AMR_count))*(self.__xcells*self.__ycells*self.__zcells)
            AMR_count += 1
            
         self.__max_spatial_amr_level = AMR_count - 1
      return self.__max_spatial_amr_level

   def get_amr_level(self,cellid):
      '''Returns the AMR level of a given cell defined by its cellid
      
      :param cellid:        The cell's cellid
      :returns:             The cell's refinement level in the AMR
      '''
      stack = True
      if not hasattr(cellid,"__len__"):
         cellid = np.atleast_1d(cellid)
         stack = False

      AMR_count = np.zeros(np.array(cellid).shape, dtype=np.int64)
      cellids = cellid.astype(np.int64)
      iters = 0
      while np.any(cellids > 0):
         mask = cellids > 0
         sub = 2**(3*AMR_count)*(self.__xcells*self.__ycells*self.__zcells)
         np.subtract(cellids, sub.astype(np.int64), out = cellids, where = mask)
         np.add(AMR_count, 1, out = AMR_count, where = mask)
         iters = iters+1
         if(iters > self.get_max_refinement_level()+1):
            logging.info("Can't have that large refinements. Something broke.")
            break

      if stack:
         return AMR_count - 1 
      else:
         return (AMR_count - 1)[0]

   def get_cell_dx(self, cellid):
      '''Returns the dx of a given cell defined by its cellid
      
      :param cellid:        The cell's cellid
      :returns:             The cell's size [dx, dy, dz]
      '''

      stack = True
      if not hasattr(cellid,"__len__"):
         cellid = np.atleast_1d(cellid)
         stack = False

      cellid = np.array(cellid, dtype=np.int64)

      dxs = np.array([[self.__dx,self.__dy,self.__dz]])

      dxs = dxs.repeat(cellid.shape[0], axis=0)

      amrs = np.array([self.get_amr_level(cellid)]).transpose()
      amrs = amrs.repeat(3,axis=1)
      amrs[amrs < 0] = 0

      ret = dxs/2**amrs

      if stack:
         return ret
      else:
         return ret[0]

   def get_cell_bbox(self, cellid):
      '''Returns the bounding box of a given cell defined by its cellid
      
      :param cellid:        The cell's cellid
      :returns:             The cell's bbox [xmin,ymin,zmin],[xmax,ymax,zmax]
      '''
      
      hdx = self.get_cell_dx(cellid)*0.5
      mid = self.get_cell_coordinates(cellid)
      return mid-hdx, mid+hdx

   def get_cell_fsgrid_slicemap(self, cellid):
      '''Returns a slice tuple of fsgrid indices that are contained in the SpatialGrid
      cell.
      '''
      low, up = self.get_cell_bbox(cellid)
      lowi, upi = self.get_fsgrid_slice_indices(low, up)
      return lowi, upi

   def get_bbox_fsgrid_slicemap(self, low, up):
      '''Returns a slice tuple of fsgrid indices that are contained in the (low, up) bounding box.
      '''
      lowi, upi = self.get_fsgrid_slice_indices(low, up)
      return lowi, upi

   def get_cell_fsgrid_subarray(self, cellid, array):
      '''Returns a subarray of the fsgrid array, corresponding to the fsgrid
      covered by the SpatialGrid cellid.
      '''
      lowi, upi = self.get_cell_fsgrid_slicemap(cellid)
      if array.ndim == 4:
         return array[lowi[0]:upi[0]+1, lowi[1]:upi[1]+1, lowi[2]:upi[2]+1, :]
      else:
         return array[lowi[0]:upi[0]+1, lowi[1]:upi[1]+1, lowi[2]:upi[2]+1]

   def get_bbox_fsgrid_subarray(self, low, up, array):
      '''Returns a subarray of the fsgrid array, corresponding to the (low, up) bounding box.
      '''
      lowi, upi = self.get_bbox_fsgrid_slicemap(low,up)
      if array.ndim == 4:
         return array[lowi[0]:upi[0]+int(1), lowi[1]:upi[1]+int(1), lowi[2]:upi[2]+int(1), :]
      else:
         return array[lowi[0]:upi[0]+1, lowi[1]:upi[1]+1, lowi[2]:upi[2]+1]


   def downsample_fsgrid_subarray(self, cellid, array):
      '''Returns a mean value of fsgrid values underlying the SpatialGrid cellid.
      '''
      fsarr = self.get_cell_fsgrid_subarray(cellid, array)
      n = fsarr.size
      if fsarr.ndim == 4:
         n = n/3
      ncells = 8**(self.get_max_refinement_level()-self.get_amr_level(cellid))
      if(n != ncells):
         warnings.warn("Weird fs subarray size", n, 'for amrlevel', self.get_amr_level(cellid), 'expect', ncells)
      return np.mean(fsarr,axis=(0,1,2))

   def fsgrid_array_to_vg(self, array):
      ''' Downsample, via averaging, an fsgrid array to the Vlasov grid
      of this reader.

      :param array:  array with first three dimensions corresponding to the
                     dimensions of the fsgrid associated with this reader.
      :returns: Vlasov grid data (in file order) of array averaged to Vlasov Grid.
      '''
      cellIds=self.read_variable("CellID")

      self.map_vg_onto_fg()
      counts = np.bincount(np.reshape(self.__vg_indexes_on_fg, self.__vg_indexes_on_fg.size))
      if array.ndim == 4:
         numel = array.shape[3]
         vgarr = np.zeros((len(cellIds),numel))
         reshaped_mapping = np.reshape(self.__vg_indexes_on_fg,self.__vg_indexes_on_fg.size)
         for i in range(numel):
            wgts = np.reshape(array[:,:,:,i],array[:,:,:,i].size)
            sums = np.bincount(reshaped_mapping, weights=wgts)
            vgarr[:,i] = np.divide(sums,counts)
      else:
         sums = np.bincount(np.reshape(self.__vg_indexes_on_fg, self.__vg_indexes_on_fg.size), weights=np.reshape(array,array.size))
         vgarr = np.divide(sums,counts)
      return vgarr

   def vg_uniform_grid_process(self, variable, expr, exprtuple):
      cellIds=self.read_variable("CellID")
      array = self.read_variable_as_fg(variable)
      array = expr(*exprtuple)
      return self.fsgrid_array_to_vg(array)

   def get_cellid_at_fsgrid_index(self, i,j,k):
      coords = self.get_fsgrid_coordinates([i,j,k])
      return self.get_cellid(coords)

   def upsample_fsgrid_subarray(self, cellid, var, array):
      '''Set the elements of the fsgrid array to the value of corresponding SpatialGrid
      cellid. Mutator for array.
      '''
      lowi, upi = self.get_cell_fsgrid_slicemap(cellid)
      value = self.read_variable(var, cellids=[cellid])
      if array.ndim == 4:
         array[lowi[0]:upi[0]+1,lowi[1]:upi[1]+1,lowi[2]:upi[2]+1,:] = value
      else:
         array[lowi[0]:upi[0]+1,lowi[1]:upi[1]+1,lowi[2]:upi[2]+1] = value
      return

   def read_variable_as_fg(self, var, operator='pass'):
      vg_cellids = self.read_variable('CellID')
      sz = self.get_fsgrid_mesh_size()
      sz_amr = self.get_spatial_mesh_size()
      vg_var = self.read_variable(var, operator=operator)
      varsize = vg_var[0].size
      if(varsize > 1):
         fg_var = np.zeros([sz[0], sz[1], sz[2], varsize], dtype=vg_var.dtype)
      else:
         fg_var = np.zeros(sz, dtype=vg_var.dtype)
      self.map_vg_onto_fg()
      fg_var = vg_var[self.__vg_indexes_on_fg]
      return fg_var


   # Builds fsgrid array that contains indices to the SpatialGrid data that are colocated with the fsgrid cells.
   # Many fsgrid cells may map to the same index of SpatialGrid data.
   # Example: for fsgrid cell at indices [i,j,k], find the overlaying SpatialGrid cell by:
   # vg_overlaying_CellID_at_ijk = self.read_variable('CellID')[self.__vg_indexes_on_fg[i,j,k]]
   # or, for all fsgrid cells:
   # vg_CellIDs_on_fg = self.read_variable('CellID')[self.__vg_indexes_on_fg]
   def map_vg_onto_fg(self):
      if(len(self.__vg_indexes_on_fg)==0):
         vg_cellids = self.read_variable('CellID')
         sz = self.get_fsgrid_mesh_size()
         sz_amr = self.get_spatial_mesh_size()
         max_amr_level = int(np.log2(sz[0] / sz_amr[0]))
         self.__vg_indexes_on_fg = np.zeros(sz, dtype=np.int64) + 1000000000 # big number to catch errors in the latter code, 0 is not good for that
         amr_levels = self.get_amr_level(vg_cellids)
         cell_indices = np.array(self.get_cell_indices(vg_cellids,amr_levels),dtype=np.int64)
         refined_ids_start = np.array(cell_indices * 2**(max_amr_level-amr_levels[:,np.newaxis]), dtype=np.int64)
         refined_ids_end = np.array(refined_ids_start + 2**(max_amr_level-amr_levels[:,np.newaxis]), dtype=np.int64)
            
         self.__vg_indexes_on_fg = map_vg_onto_fg_loop(self.__vg_indexes_on_fg,vg_cellids, refined_ids_start, refined_ids_end)

      return self.__vg_indexes_on_fg

   def get_cell_fsgrid(self, cellid):
      '''Returns a slice tuple of fsgrid indices that are contained in the SpatialGrid
      cell.
      '''
      low, up = self.get_cell_bbox(cellid)
      lowi, upi = self.get_fsgrid_slice_indices(low, up)
      return lowi, upi

   def get_fsgrid_coordinates(self, ri):
      '''Returns real-space center coordinates of the fsgrid 3-index.
      '''
      lowerlimit = self.get_fsgrid_mesh_extent()[0:3]
      dxs = self.get_fsgrid_cell_size()

      return lowerlimit+dxs*(np.array(ri)+0.5)

   def get_unique_cellids(self, coords):
      ''' Returns a list of cellids containing all the coordinates in coords,
          with no duplicate cellids. Relative order of elements is conserved.

      :param coords:         A list of coordinates

      :returns: a list of unique cell ids
      '''
      # cids = [int(self.get_cellid(coord)) for coord in coords]
      cids = self.get_cellid(coords)

      #choose unique cids, keep ordering. This requires a bit of OrderedDict magic (python 2.7+)
      cidsout = np.array(list(OrderedDict.fromkeys(cids)))
      return cidsout
   
   def get_cellid(self, coords):
      ''' Returns the cell ids at given coordinates

      :param coords:        The cells' coordinates
      :returns: the cell ids

      .. note:: Returns 0 if the cellid is out of bounds!
      '''

      stack = True
      coordinates = np.array(coords)
      if len(coordinates.shape) == 1:
         coordinates = np.atleast_2d(coordinates)
         stack = False

      if coordinates.shape[1] != 3:
         raise IndexError("Coordinates are required to be 3-dimensional (coords were %d-dimensional)" % coordinates.shape[1])

      # If needed, read the file index for cellid
      # if len(self.__fileindex_for_cellid) == 0:
      self.__read_fileindex_for_cellid()
      #good_ids = self.read_variable("CellID")
      # good_ids = np.array(list(self.__fileindex_for_cellid.keys()))
      # good_ids.sort()

      cellids = np.zeros((coordinates.shape[0]), dtype=np.int64)

      # mask for cells that are unresolved - out-of-bounds coordinates stay at zero
      mask = (
               (self.__xmax > coordinates[:,0]) & (self.__xmin < coordinates[:,0]) &
               (self.__ymax > coordinates[:,1]) & (self.__ymin < coordinates[:,1]) &
               (self.__zmax > coordinates[:,2]) & (self.__zmin < coordinates[:,2])
      )

      # Get cell lengths:
      cell_lengths = np.array([self.__dx, self.__dy, self.__dz])

      cellindices = np.zeros(coordinates.shape, dtype=np.int64)
      
      

      # Get cell indices:
      cellindices[mask,0] = (coordinates[mask,0] - self.__xmin)/cell_lengths[0]
      cellindices[mask,1] = (coordinates[mask,1] - self.__ymin)/cell_lengths[1]
      cellindices[mask,2] = (coordinates[mask,2] - self.__zmin)/cell_lengths[2]
      # Get the cell id:
      cellids[mask] = cellindices[mask,0] + cellindices[mask,1] * self.__xcells + cellindices[mask,2] * self.__xcells * self.__ycells + 1

      # Going through AMR levels as needed
      AMR_count = 0
      ncells_lowerlevel = 0
      refmax = self.get_max_refinement_level()


      while AMR_count < refmax +1:
         drop = ~dict_keys_exist(self.__fileindex_for_cellid, cellids[mask])

         mask[mask] = mask[mask] & drop
         
         ncells_lowerlevel += 2**(3*AMR_count)*(self.__xcells*self.__ycells*self.__zcells) # Increment of cellID from lower lvl             
         AMR_count += 1
         # Get cell lengths:
         cell_lengths = np.array([self.__dx, self.__dy, self.__dz]) / 2**AMR_count # Check next AMR level

         # Get cell indices:
         cellindices[mask,0] = (coordinates[mask,0] - self.__xmin)/cell_lengths[0]
         cellindices[mask,1] = (coordinates[mask,1] - self.__ymin)/cell_lengths[1]
         cellindices[mask,2] = (coordinates[mask,2] - self.__zmin)/cell_lengths[2]
         # Get the cell id:
         cellids[mask] = ncells_lowerlevel + cellindices[mask,0] + 2**(AMR_count)*cellindices[mask,1] * self.__xcells + 4**(AMR_count) * cellindices[mask,2] * self.__xcells * self.__ycells + 1

      drop = ~dict_keys_exist(self.__fileindex_for_cellid, cellids[mask])
      mask[mask] = mask[mask] & drop
      cellids[mask] = 0 # set missing cells to null cell
      if stack:
         return cellids
      else:
         return cellids[0]

   def get_cellid_with_vdf(self, coords, pop = 'proton'):
      ''' Returns the cell ids nearest to test points, that contain VDFs

      :param coords:    Test coordinates [meters] of N_in points in ND-dimensional space
                        array with shape [N_in, ND] or [ND]

      Example: cellid = vlsvReader.get_cellid_with_vdf(np.array([1e8, 0, 0]))
      :returns: the cell ids

      '''
      stack = True
      coords_in = np.array(coords)
      if len(coords_in.shape) == 1:
         coords_in = np.atleast_2d(coords_in)
         stack = False

      if not pop in self.__cells_with_blocks:
         self.__set_cell_offset_and_blocks_nodict(pop)
      cid_w_vdf = self.__cells_with_blocks[pop]
      coords_w_vdf = self.get_cell_coordinates(cid_w_vdf)
      N_in = coords_in.shape[0]; N_w_vdf = len(cid_w_vdf)

      if N_w_vdf==0:
         logging.info("Error: No velocity distributions found!")
         sys.exit()

      # Boolean array flag_empty_in indicates if queried points (coords_in) don't already lie within vdf-containing cells, 
      output = self.get_cellid(coords_in)
      flag_empty_in = np.array( [cid not in self.__order_for_cellid_blocks[pop] for cid in output] )
      N_empty_in = sum(flag_empty_in)

      if N_empty_in == 0:   # every element of coords_in already within a vdf-containing cell
         if stack:
            return output
         else:
            return output[0]
      
      # Direct search: calculate distances for each pair points (test <--> vdf cells)
      # Only calculate nearest distance if there is no VDF already in the cell (using flag_empty_in) 
      '''
      try:
         # Vectorized approach: 
         dist2 = np.nansum((coords_in[flag_empty_in, None, :] - coords_w_vdf[None, :, :])**2, axis = -1)   # distance^2, shape [N_empty_in, N_w_vdf]
         output[flag_empty_in] = cid_w_vdf[np.argmin(dist2, axis = 1)]
      except MemoryError:
      '''
      # Loop approach:
      logging.info('Not enough memory to broadcast arrays! Using a loop instead...')
      ind_emptycell = np.arange(len(flag_empty_in))[flag_empty_in]
      for ind in ind_emptycell:
         this_coord = coords_in[ind, :]
         dist2 = np.nansum((coords_w_vdf - this_coord)**2, axis = -1)
         output[ind] = cid_w_vdf[np.argmin(dist2)]

      # return cells that minimize the distance to the test points
      if stack:
         return output
      else:
         return output[0]
      
   def cellid_has_vdf(self, cid, pop = 'proton')->bool:
      ''' Returns whether the cid in question has a vdf or not
      :param coords:    the cellid to test for
      :returns: bool 
      '''
      self.__set_cell_offset_and_blocks_nodict(pop)
      cid_w_vdf = self.__cells_with_blocks[pop]
      return cid in cid_w_vdf

   def get_vertex_indices(self, coordinates):
      ''' Get dual grid vertex indices for all coordinates.
      
      Works by truncation to integer indices at fsgrid resolution, for cell low-corners.
      :param coordinates: np.array of coordinates, shaped either (3,) or (N,3)
      '''
      coordinates = np.array(coordinates)
      stack = True
      if(len(coordinates.shape) == 1):
         stack = False
         coordinates = coordinates[np.newaxis,:]
      cell_lengths = np.array([self.__dx, self.__dy, self.__dz]) / 2**self.get_max_refinement_level()
      extents = self.get_fsgrid_mesh_extent()
      mins = extents[0:3]
      maxs = extents[3:5]
      eps = np.mean(cell_lengths)/1000
      crds = coordinates - mins[np.newaxis,:] + eps
      indices = crds/cell_lengths[np.newaxis,:]
      indices = indices.astype(int)

      if stack:
         return [tuple(inds) for inds in indices]
      else:
         coordinates = coordinates[0,:]
         return tuple(indices[0,:])
      
   def get_vertex_coordinates_from_indices(self, indices):
      ''' Convert vertex indices to physical coordinates.
      :param indices: numpy array of vertex indices, either (3,) or (N,3)

      :returns: numpy array of coordinates, with matching shape to indices
      '''
      stack = True
      inds = np.array(indices)
      if(len(inds.shape) == 1):
         stack = False
         inds = inds[np.newaxis,:]
      cell_lengths = np.array([self.__dx, self.__dy, self.__dz]) / 2**self.get_max_refinement_level()
      extents = self.get_fsgrid_mesh_extent()
      mins = extents[0:3]
      crds = inds*cell_lengths[np.newaxis,:]
      crds = crds + mins[np.newaxis,:]

      if stack:
         return crds
      else:
         return crds[0,:]

   # this should then do the proper search instead of intp for in which dual of the cell the point lies
   def get_dual(self, pts, cellids=None):
      ''' Find the duals that contain the coordinate points pts. This will call the iterative find_ksi function
      to see if the resulting interpolation weights for the coordinate are in the range [0,1]; if not, it will iterate 
      through neighbouring duals until a dual is found.
      :parameter pts: numpy array of coordinates (N,3)

      :returns: duals (numpy array of N 3-tuples), ksis (numpy array of interpolation weights (N, 8))
      '''

      from pyCalculations.interpolator_amr import find_ksi

      # start the search from the vertices 
      if cellids is None:
         cid = self.get_cellid(np.atleast_2d(pts))
      else:
         cid = cellids

      self.build_cell_vertices(np.atleast_1d(cid),prune_unique=True)
      
      vverts = [self.__cell_vertices[c] for c in cid]
      set_of_verts = set()
      # Loops over duals indexed by vertex tuple
      self.build_dual_from_vertices(list(set_of_verts.union(*vverts)))

      for i,verts in enumerate(vverts):
         bboxes = np.array(itemgetter(*verts)(self.__dual_bboxes))
         vmask = np.all(pts[i,:] >= bboxes[:,0:3],axis=1) & np.all(pts[i,:] <= bboxes[:,3:6],axis=1)
         okverts = [verts[ind] for ind, val in enumerate(vmask) if val]
         vverts[i] = okverts

      # Breaks degeneracies by expanding the dual cells vertices along
      #  main-grid diagonals
      offset_eps = 1.0
      offsets = np.array([[-1.0, -1.0, -1.0],
                           [-1.0, -1.0,  1.0],
                           [-1.0,  1.0, -1.0],
                           [-1.0,  1.0,  1.0],
                           [ 1.0, -1.0, -1.0],
                           [ 1.0, -1.0,  1.0],
                           [ 1.0,  1.0, -1.0],
                           [ 1.0,  1.0,  1.0],
                        ]) * offset_eps

      verts_per_pt = [len(verts) for verts in vverts]
      ppts = pts.repeat(verts_per_pt, axis=0)
      pinds = np.indices((pts.shape[0],)).repeat(verts_per_pt)

      all_verts = [v for vs in vverts for v in vs]

      all_vcoords = self.get_cell_coordinates(np.array(itemgetter(*all_verts)(self.__dual_cells))[np.newaxis,:].reshape(-1))

      all_vcoords = offsets[np.newaxis,:,:]+all_vcoords.reshape(-1,8,3)
      all_vksis = find_ksi(ppts, all_vcoords)

      foundmask = np.all(all_vksis <=1, axis=1) & np.all(all_vksis >= 0, axis=1)

      ind = np.nonzero(foundmask)[0]
      all_vksis = all_vksis[ind,:]

      found_pts = pinds[ind]
      found_pts, inds = np.unique(found_pts, return_index = True)



      ksis = np.full_like(pts, np.nan)
      duals = np.empty((pts.shape[0],),dtype="i,i,i")
      duals[:] = (0,0,0)
      dduals = np.array(np.array(all_verts,dtype="i,i,i")[ind], dtype="i,i,i")

      ksis[found_pts,:] = all_vksis[inds,:]
      duals[found_pts] = dduals[inds]

      return duals.astype(object), ksis
      
   # For now, combined caching accessor and builder
   def build_cell_vertices(self, cid, prune_unique=False):
      ''' Builds, caches and returns the vertices that lie on the surfaces of CellIDs cid.
      
      :parameter cid: numpy array of CellIDs
      :parameter prune_unique: bool [False], if you suspect you might be calling the function many times with the 
                               same CellID in the list, it might be beneficial to enable this and not repeat the operation for duplicate entries.

      :returns: Dictionary of cell c (int) : set of vertex indices (3-tuple) that touch the cell c.

      '''
      if prune_unique:
         cid = np.unique(cid)

      mask = ~dict_keys_exist(self.__cell_vertices,cid,prune_unique=False)
      # {cid : 8-tuple of vertex_inds}
      
      cell_neighbors = self.build_cell_neighborhoods(cid[mask])
      corner_vertices = self.get_cell_corner_vertices(cid[mask])

      levels = self.get_amr_level(cid[mask])

      # find irregular cells
      # irregular_mask = np.array([np.any(levels[i] != self.get_amr_level(np.array(list(cell_neighbors[c])))) for i,c in enumerate(cid[mask])],dtype=bool)
      mask[mask] = mask[mask] #& irregular_mask # beautiful
      
      cell_hanging_nodes = {c: () for c in cid[mask]}

      if(len(cid[mask])==1):
         self.get_cell_corner_vertices(np.array(list(set().union(itemgetter(*cid[mask])(cell_neighbors)))))
      elif(len(cid[mask])>=1):
         self.get_cell_corner_vertices(np.array(list(set().union(*itemgetter(*cid[mask])(cell_neighbors)))))
      # Loop over all the irregular cells (with finer neighbours) that may have hanging nodes
      # for i,c in enumerate(irregular_cells):
      for i,c in enumerate(cid[mask]):
         corners = np.array(self.__cell_corner_vertices[c], dtype=np.int64)
         # finer_neighbors = [n for n in cell_neighbors[c] if self.get_amr_level(n) > levels[i]]
         hanging_set = set()
         # ncorners = self.get_cell_corner_vertices(np.array(finer_neighbors))
         ncorners = itemgetter(*cell_neighbors[c])(self.__cell_corner_vertices) #self.get_cell_corner_vertices(np.array(list(cell_neighbors[c])))

         # Possible hanging nodes are those that are the vertices of finer neighbours.
         for vertex_inds_set in ncorners:#.values():
            hanging_set.update(set(vertex_inds_set) - set(corner_vertices[c])) # no need to add current corners
         
         cmin = np.min(corners,axis=0)
         cmax = np.max(corners,axis=0)

         pruned_hangers = {v for v in hanging_set if np.all(np.array(v,dtype=np.int64) <= cmax) and np.all(np.array(v,dtype=np.int64) >= cmin)}
         cell_hanging_nodes[c] = tuple(pruned_hangers)

      vertices = {c: corner_vertices[c]+cell_hanging_nodes[c] for c in cid[mask]}
      self.__cell_vertices.update(vertices)
      
      for c in cid[~mask]:
         vertices[c] = self.__cell_vertices[c]
      
      return vertices

   def get_cell_corner_vertices(self, cids):
      ''' Builds, caches and returns the vertices that lie on the corners of CellIDs cid.
      :parameter cid: numpy array of CellIDs

      :returns: Dictionary of cell c (int) : 8-tuple of vertex indices (3-tuples).

      '''

      mask = ~dict_keys_exist(self.__cell_vertices,cids,prune_unique=False)
      coords = self.get_cell_coordinates(cids[mask])
      vertices = np.zeros((len(cids[mask]), 8, 3),dtype=int)
      cell_vertex_sets = {}

      if(len(cids[mask]) > 0): 
         ii = 0
         for x in [-1,1]:
            for y in [-1,1]:
               for z  in [-1,1]:
                  if x == 0 and y == 0 and z == 0:
                     continue
                  vertices[:,ii,:] = np.array(self.get_vertex_indices(coords + np.array((x,y,z))[np.newaxis,:]*self.get_cell_dx(cids[mask])/2))
                  ii += 1

         for i, c in enumerate(cids[mask]):
            vlist = vertices[i,:,:]
            vtuple = tuple([tuple(inds) for inds in vlist])
            cell_vertex_sets[c] = vtuple
            
         self.__cell_corner_vertices.update(cell_vertex_sets)

      for i, c in enumerate(cids[~mask]):
         cell_vertex_sets[c] = self.__cell_corner_vertices[c]

      return cell_vertex_sets


   # again, combined getter and builder..
   def build_cell_neighborhoods(self, cids):

      mask = ~dict_keys_exist(self.__cell_neighbours, cids, prune_unique=False)
      cell_vertex_sets = self.get_cell_corner_vertices(cids[mask]) # these are enough to fetch the neighbours

      cell_neighbor_sets = {c: set() for c in cell_vertex_sets.keys()}
      vertices_todo = set().union(*cell_vertex_sets.values())
      neighbor_tuples_dict = self.build_dual_from_vertices(list(vertices_todo))
      for c,verts in cell_vertex_sets.items():
         # neighbor_tuples = self.build_dual_from_vertices(verts)
         # cell_neighbor_sets[c].update(set().union(*neighbor_tuples.values()))
         cell_neighbor_sets[c].update(set().union(*itemgetter(*cell_vertex_sets[c])(neighbor_tuples_dict)))
      
      self.__cell_neighbours.update(cell_neighbor_sets)

      for c in cids[~mask]:
         cell_neighbor_sets[c] = self.__cell_neighbors[c]

      return cell_neighbor_sets



   def build_dual_from_vertices(self, vertices):

      vertices = list(set(vertices))

      # I don't like this, but alternatives seem worse.
      done = []
      todo = []
      # mask = dict_keys_exist(self.__dual_cells, vertices)
      # mask = np.array([v not in self.__dual_cells.keys() for v in vertices],dtype=bool)
      for i,v in enumerate(vertices):
         if v in self.__dual_cells.keys():
            done.append(v)
         else:
            todo.append(v)
      # vertices=np.array(vertices,dtype="i,i,i")
      # done = [tuple(v) for v in vertices[mask]]
      # todo = [tuple(v) for v in vertices[~mask]]

      dual_sets_done   = {v : self.__dual_cells[v] for v in done}
      
      dual_sets = {}

      len_todo = len(todo)

      if len_todo > 0:#np.sum(mask) > 0:
         
         dual_bboxes = {}
         eps = 1
         v_cells = np.zeros((len_todo, 8),dtype=int)
         v_cellcoords = np.zeros((len_todo, 8,3))
         ii = 0
         vcoords = self.get_vertex_coordinates_from_indices(todo)

         # TODO get rid of the loop
         offsets = []
         for x in [-1,1]:
            for y in [-1,1]:
               for z  in [-1,1]:
                  offsets.append([x,y,z])
                  v_cellcoords[:,ii,:] = eps*np.array((x,y,z))[np.newaxis,:] + vcoords
                  v_cells[:,ii] = self.get_cellid(v_cellcoords[:,ii])
                  ii += 1

         v_cellcoords = self.get_cell_coordinates(v_cells.reshape((-1))).reshape((-1,8,3))

         dual_sets.update({vinds: tuple(v_cells[i,:]) for i,vinds in enumerate(todo)})
         mins = np.min(v_cellcoords, axis=1)
         maxs = np.max(v_cellcoords, axis=1)
         dual_bboxes.update({vinds: np.hstack((mins[i,:],maxs[i,:])) for i,vinds in enumerate(todo)})

         self.__dual_cells.update(dual_sets)
         self.__dual_bboxes.update(dual_bboxes)
      dual_sets_done.update(dual_sets)

      return dual_sets_done

   # build a dual coverage to enable interpolation to each coordinate
   def build_duals_from_coordinates(self, coordinates):
      
      coordinates = np.atleast_2d(coordinates)
      cid = self.get_unique_cellids(coordinates)
      coords = self.get_cell_coordinates(cid)
      
      ncoords = coords.shape[0]
      if(coords.shape[1] != 3):
         raise IndexError("Coordinates are required to be three-dimensional (coords.shape[1]==3 or convertible to such))")
      
      vertices = set()
      vsets = self.build_cell_vertices(cid,prune_unique=False)
      vertices = vertices.union(*vsets.values())
      
      return self.build_dual_from_vertices(list(vertices))

   # build a dual coverage to enable interpolation to each coordinate
   def build_duals(self, cid):
      
      cid = np.atleast_1d(cid)

      mask = ~dict_keys_exist(self.__cell_duals, cid, prune_unique=False)

      if(np.sum(mask) > 0):
         coords = self.get_cell_coordinates(cid[mask])
         
         ncoords = coords.shape[0]
         if(coords.shape[1] != 3):
            raise IndexError("Coordinates are required to be three-dimensional (coords.shape[1]==3 or convertible to such))")
         
         vertices = set()
         vsets = self.build_cell_vertices(cid[mask])
         vertices = vertices.union(*vsets.values())
         for c in cid[mask]:
            self.__cell_duals[c] = vsets[c]
         
         self.build_dual_from_vertices(list(vertices))



   def get_cell_coordinates(self, cellids):
      ''' Returns a given cell's coordinates as a numpy array

      :param cellids:            The array of cell IDs
      :returns: a numpy array with the coordinates

      .. seealso:: :func:`get_cellid`

      .. note:: The cell ids go from 1 .. max not from 0
      '''

      stack = True
      if not hasattr(cellids,"__len__"):
         cellids = np.atleast_1d(cellids)
         stack = False

      # Get cell lengths:
      xcells = np.zeros((self.get_max_refinement_level()+1), dtype=np.int64)
      ycells = np.zeros((self.get_max_refinement_level()+1), dtype=np.int64)
      zcells = np.zeros((self.get_max_refinement_level()+1), dtype=np.int64)
      for r in range(self.get_max_refinement_level()+1):
         xcells[r] = self.__xcells*2**(r)
         ycells[r] = self.__ycells*2**(r)
         zcells[r] = self.__zcells*2**(r)

      reflevels = self.get_amr_level(cellids)
      cellindices = self.get_cell_indices(cellids)

      # Get cell coordinates:
      cell_lengths = np.array([(self.__xmax - self.__xmin)/(xcells[reflevels]),
                               (self.__ymax - self.__ymin)/(ycells[reflevels]),
                               (self.__zmax - self.__zmin)/(zcells[reflevels])]).T
      mins = np.array([self.__xmin,self.__ymin,self.__zmin])
      cellcoordinates = mins + (cellindices + 0.5)*cell_lengths
      # Return the coordinates:
      if stack:
         return np.array(cellcoordinates)
      else:
         return np.array(cellcoordinates)[0,:]

   def get_cell_indices(self, cellids, reflevels=None):
      ''' Returns a given cell's indices as a numpy array

      :param cellid:            The cell's ID, numpy array
      :param reflevel:          The cell's refinement level in the AMR
      :returns: a numpy array with the coordinates

      .. seealso:: :func:`get_cellid`

      .. note:: The cell ids go from 1 .. max not from 0
      '''

      stack = True
      if not hasattr(cellids,"__len__"):
         cellids = np.atleast_1d(cellids)
         stack = False

      if reflevels is None:
         reflevels = self.get_amr_level(cellids)
      else:
         reflevels = np.atleast_1d(reflevels)

      mask = reflevels >= 0
      # Calculating the index of the first cell at this reflevel
      index_at_reflevel = np.zeros(self.get_max_refinement_level()+1, dtype=np.int64)
      isum = 0
      for i in range(0,self.get_max_refinement_level()):
         isum = isum + 2**(3*i) * self.__xcells * self.__ycells * self.__zcells
         index_at_reflevel[i+1] = isum


      # Get cell indices:
      cellids = np.array(cellids - 1 - index_at_reflevel[reflevels], dtype=np.int64)
      cellindices = np.full((len(cellids),3), -1)
      cellindices[mask,0] = (cellids[mask])%(np.power(2,reflevels[mask])*self.__xcells)
      cellindices[mask,1] = ((cellids[mask])//(np.power(2,reflevels[mask])*self.__xcells))%(np.power(2,reflevels[mask])*self.__ycells)
      cellindices[mask,2] = (cellids[mask])//(np.power(4,reflevels[mask])*self.__xcells*self.__ycells)

      # Return the indices:
      if stack:
         return np.array(cellindices)
      else:
         return np.array(cellindices)[0]

   def get_cell_neighbor(self, cellidss, offsetss, periodic, prune_uniques=False):
      ''' Returns a given cells neighbor at offset (in indices)

      :param cellids:            The cell's ID
      :param offsets:            The offset to the neighbor in indices
      :param periodic:          For each dimension, is the system periodic
      :returns: the cellid of the neighbor

      .. note:: Returns 0 if the offset is out of bounds!

      '''
      stack = True
      if not hasattr(cellidss,"__len__"):
         cellidss = np.atleast_1d(cellidss)
         offsetss = np.atleast_2d(offsetss)
         stack = False

      if prune_uniques:
         fullargs = np.array(np.hstack((cellidss[:,np.newaxis],offsetss)))
         uniqueargs, inverse_indices = np.unique(fullargs,axis=0, return_inverse=True)
         cellids = uniqueargs[:,0]
         offsets = uniqueargs[:,1:4]
      else:
         cellids = cellidss
         offsets = offsetss
         inverse_indices = np.indices((len(cellids),))


      reflevel = self.get_amr_level(cellids)
      indices = self.get_cell_indices(cellids, reflevel)

      cellid_neighbors = np.zeros_like(cellids)
      mask = np.ones(cellids.shape, dtype=bool)
      # Special case if no offset (return self in that case); and require in-domain reflevel
      mask = ~((offsets[:,0]==0) & (offsets[:,1]==0) & (offsets[:,2]==0)) & (reflevel >= 0)

      # Getting the neighbour at the same refinement level
      ngbr_indices = np.zeros((len(cellids),3))
      sys_sizes= np.ones(ngbr_indices.shape, dtype=np.float64)
      sys_sizes[mask,0] = 2**reflevel[mask]*self.__xcells
      sys_sizes[mask,1] = 2**reflevel[mask]*self.__ycells
      sys_sizes[mask,2] = 2**reflevel[mask]*self.__zcells
      for i in range(3):
         ngbr_indices[:,i] = indices[:,i] + offsets[:,i]
         if periodic[i]:
            lowmask = mask & (ngbr_indices[:,i] < 0)
            ngbr_indices[lowmask,i] = ngbr_indices[lowmask,i] % sys_sizes[lowmask,i]
            himask = mask & (ngbr_indices[:,i] >= sys_sizes[:,i])
            ngbr_indices[himask,i] = ngbr_indices[himask,i] % sys_sizes[himask,i]
   
         elif np.any((ngbr_indices[mask, i] < 0) or (ngbr_indices[mask,i] >= sys_sizes[mask,i])):
            raise ValueError("Error in Vlsvreader get_cell_neighbor: neighbor out of bounds")

      coord_neighbor = np.zeros(ngbr_indices.shape, dtype=np.float64)
      coord_neighbor[mask,:] = np.array([self.__xmin,self.__ymin,self.__zmin]) + (ngbr_indices[mask,:] + np.array((0.5,0.5,0.5))) * np.array([self.__dx,self.__dy,self.__dz])/2**np.repeat(np.atleast_2d(reflevel[mask]).T,3,axis=1)
      
      cellid_neighbors[mask] = self.get_cellid(coord_neighbor[mask,:])
      cellid_neighbors[(offsets[:,0]==0) & (offsets[:,1]==0) & (offsets[:,2]==0)] = cellids[(offsets[:,0]==0) & (offsets[:,1]==0) & (offsets[:,2]==0)]

      # if np.any(self.get_amr_level(cellid_neighbors)!=reflevel):
      #    warnings.warn("A neighboring cell found at a different refinement level. Behaviour is janky, and results will vary.")

      # Return the neighbor cellids/cellid:
      if stack:
         return np.array(cellid_neighbors[inverse_indices])
      else:
         return np.array(cellid_neighbors)[0]

   def get_WID(self):
      # default WID=4
      widval=4
      if self.check_parameter("velocity_block_width"):
         widval = self.read_parameter("velocity_block_width")
      return widval

   def get_velocity_cell_ids(self, vcellcoord, pop="proton"):
      ''' Returns velocity cell ids of given coordinate

      Arguments:
      :param vcellcoords: One 3d coordinate
      :returns: Velocity cell id

      .. seealso:: :func:`get_velocity_cell_coordinates`
      '''
      WID=self.get_WID()
      WID2=WID*WID
      WID3=WID2*WID
      vmin = np.array([self.__meshes[pop].__vxmin, self.__meshes[pop].__vymin, self.__meshes[pop].__vzmin])
      dv = np.array([self.__meshes[pop].__dvx, self.__meshes[pop].__dvy, self.__meshes[pop].__dvz])
      block_index = np.floor((vcellcoord - vmin) / (WID * dv))
      cell_index = np.floor(np.remainder(vcellcoord - vmin, WID * dv) / dv)
      vcellid = int(block_index[0])
      vcellid += int(block_index[1] * self.__meshes[pop].__vxblocks)
      vcellid += int(block_index[2] * self.__meshes[pop].__vxblocks * self.__meshes[pop].__vyblocks)
      vcellid *= WID3
      vcellid += int(cell_index[0])
      vcellid += int(cell_index[1] * WID)
      vcellid += int(cell_index[2] * WID2)
      return vcellid

   def get_velocity_cell_coordinates(self, vcellids, pop="proton"):
      ''' Returns a given velocity cell's coordinates as a numpy array

      Arguments:
      :param vcellids:       The velocity cell's ID
      :returns: a numpy array with the coordinates

      .. seealso:: :func:`get_cell_coordinates` :func:`get_velocity_block_coordinates`
      '''
      vcellids = np.atleast_1d(vcellids)
      WID=self.get_WID()
      WID2=WID*WID
      WID3=WID2*WID
      # Get block ids:
      blocks = vcellids.astype(int) // WID3
      # Get block coordinates:
      blockIndicesX = np.remainder(blocks.astype(int), (int)(self.__meshes[pop].__vxblocks))
      blockIndicesY = np.remainder(blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks), (int)(self.__meshes[pop].__vyblocks))
      blockIndicesZ = blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks*self.__meshes[pop].__vyblocks)
      blockCoordinatesX = blockIndicesX.astype(float) * self.__meshes[pop].__dvx * WID + self.__meshes[pop].__vxmin
      blockCoordinatesY = blockIndicesY.astype(float) * self.__meshes[pop].__dvy * WID + self.__meshes[pop].__vymin
      blockCoordinatesZ = blockIndicesZ.astype(float) * self.__meshes[pop].__dvz * WID + self.__meshes[pop].__vzmin
      # Get cell indices:
      cellids = np.remainder(vcellids.astype(int), (int)(WID3))
      cellIndicesX = np.remainder(cellids.astype(int), (int)(WID))
      cellIndicesY = np.remainder((cellids.astype(int)//(int)(WID)).astype(int), (int)(WID))
      cellIndicesZ = cellids.astype(int)//(int)(WID2)
      # Get cell coordinates:
      cellCoordinates = np.array([blockCoordinatesX.astype(float) + (cellIndicesX.astype(float) + 0.5) * self.__meshes[pop].__dvx,
                                  blockCoordinatesY.astype(float) + (cellIndicesY.astype(float) + 0.5) * self.__meshes[pop].__dvy,
                                  blockCoordinatesZ.astype(float) + (cellIndicesZ.astype(float) + 0.5) * self.__meshes[pop].__dvz])

      return cellCoordinates.transpose()

   def get_velocity_block_indices( self, blocks, pop="proton"):
      ''' Returns the block indices of the given blocks in a numpy array

          :param blocks:         list of block ids
          :returns: a numpy array containing the block indices e.g. np.array([np.array([2,1,3]), np.array([5,6,6]), ..])

          .. seealso:: :func:`get_velocity_block_coordinates`
      '''
      WID=self.get_WID()
      blockIndicesX = np.remainder(blocks.astype(int), (int)(self.__meshes[pop].__vxblocks))
      blockIndicesY = np.remainder(blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks), (int)(self.__meshes[pop].__vyblocks))
      blockIndicesZ = blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks*self.__meshes[pop].__vyblocks)
      # Return the indices:
      return np.array([blockIndicesX,
                       blockIndicesY,
                       blockIndicesZ]).transpose()

   def get_velocity_blockGID(self, blockindices, pop="proton"):
      WID=self.get_WID()

      bIX = blockindices[:,0]
      bIY = blockindices[:,1]
      bIZ = blockindices[:,2]

      GIDs = bIZ + bIY*self.__meshes[pop].__vzblocks + bIX*self.__meshes[pop].__vzblocks*self.__meshes[pop].__vyblocks
      return GIDs

   def get_velocity_block_coordinates( self, blocks, pop="proton"):
      ''' Returns the block coordinates of the given blocks in a numpy array

          :param blocks:         list of block ids
          :returns: a numpy array containing the block coordinates e.g. np.array([np.array([2,1,3]), np.array([5,6,6]), ..])

          .. seealso:: :func:`get_velocity_cell_coordinates`
      '''
      WID=self.get_WID()
      blockIndicesX = np.remainder(blocks.astype(int), (int)(self.__meshes[pop].__vxblocks))
      blockIndicesY = np.remainder(blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks), (int)(self.__meshes[pop].__vyblocks))
      blockIndicesZ = blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks*self.__meshes[pop].__vyblocks)
      blockCoordinatesX = blockIndicesX.astype(float) * self.__meshes[pop].__dvx * WID + self.__meshes[pop].__vxmin
      blockCoordinatesY = blockIndicesY.astype(float) * self.__meshes[pop].__dvy * WID + self.__meshes[pop].__vymin
      blockCoordinatesZ = blockIndicesZ.astype(float) * self.__meshes[pop].__dvz * WID + self.__meshes[pop].__vzmin
      # Return the coordinates:
      return np.array([blockCoordinatesX.astype(float),
                       blockCoordinatesY.astype(float),
                       blockCoordinatesZ.astype(float)]).transpose()

   def get_velocity_blocks( self, blockCoordinates, pop="proton" ):
      ''' Returns the block ids of the given block coordinates in a numpy array form

          :param blockcoordinates:         list of block coordinates e.g. np.array([np.array([2,1,3]), np.array([5,6,6]), ..])
          :returns: a numpy array containing the block ids e.g. np.array([4,2,56,44,2, ..])

          .. seealso:: :func:`get_velocity_block_coordinates`
      '''
      WID=self.get_WID()
      mins = np.array([self.__meshes[pop].__vxmin, self.__meshes[pop].__vymin, self.__meshes[pop].__vzmin]).astype(float)
      dvs = np.array([WID*self.__meshes[pop].__dvx, WID*self.__meshes[pop].__dvy, WID*self.__meshes[pop].__dvz]).astype(float)
      multiplier = np.array([1, self.__meshes[pop].__vxblocks, self.__meshes[pop].__vxblocks * self.__meshes[pop].__vyblocks]).astype(float)
      velocity_block_ids = np.sum(np.floor(((blockCoordinates.astype(float) - mins) / dvs)) * multiplier, axis=-1)
      return velocity_block_ids

   def construct_velocity_cells( self, blocks ):
      ''' Returns velocity cells in given blocks

          :param blocks:         list of block ids
          :returns: a numpy array containing the velocity cell ids e.g. np.array([4,2,56,44,522, ..])
      '''
      WID=self.get_WID()
      WID3=WID*WID*WID
      return np.ravel(np.outer(np.array(blocks), np.ones(WID3)) + np.arange(WID3))

   def construct_velocity_cell_coordinates( self, blocks ):
      ''' Returns velocity cell coordinates in given blocks

          :param blocks:         list of block ids
          :returns: a numpy array containing the velocity cell ids e.g. np.array([4,2,56,44,522, ..])
      '''
      # Construct velocity cell coordinates from velocity cells and return them
      return self.get_velocity_cell_coordinates( self.construct_velocity_cells(blocks) )


   def construct_velocity_cell_nodes( self, blocks, pop="proton" ):
      ''' Returns velocity cell nodes in given blocks

          :param blocks:         list of block ids
          :returns: a numpy array containing velocity cell nodes and the keys for velocity cells

          .. note:: This is used for constructing velocity space inside the mayavi module

          .. seealso:: :mod:`grid`
      '''
      blocks = np.array(blocks)
      WID=self.get_WID()
      WID2=WID*WID
      WID3=WID2*WID
      # Get block coordinates:
      blockIndicesX = np.remainder(blocks.astype(int), (int)(self.__meshes[pop].__vxblocks)).astype(np.uint16)
      blockIndicesY = np.remainder(blocks.astype(int)//(int)(self.__meshes[pop].__vxblocks), (int)(self.__meshes[pop].__vyblocks)).astype(np.uint16)
      blockIndicesZ = (blocks.astype(np.uint64)//(int)(self.__meshes[pop].__vxblocks*self.__meshes[pop].__vyblocks)).astype(np.uint16)

      cellsPerDirection = WID
      cellsPerBlock = WID3

      # Get velocity cell min coordinates (per velocity block)
      vcellids = np.arange(cellsPerBlock).astype(np.uint32)
      cellIndicesX = np.remainder(vcellids.astype(int), (int)(cellsPerDirection)).astype(np.uint16)
      cellIndicesY = np.remainder((vcellids.astype(int)//(int)(cellsPerDirection)).astype(int), (int)(cellsPerDirection)).astype(np.uint16)
      cellIndicesZ = (vcellids.astype(int)//(int)(cellsPerDirection*cellsPerDirection)).astype(np.uint16)

      # Construct velocity cell node indices for every velocity cell per velocity block

      nodesPerCell = 8

      # NOTE: The ordering of the numpy array won't make sense to anyone who hasn't read VTK documentation. For further info check VTK_VOXEL. The numpy array is constructed according to VTK voxel's nodes
      cellNodeIndicesX = np.ravel(np.outer(cellIndicesX, np.ones(nodesPerCell)) + np.array([0, 1, 0, 1, 0, 1, 0, 1])).astype(np.uint16)
      cellNodeIndicesY = np.ravel(np.outer(cellIndicesY, np.ones(nodesPerCell)) + np.array([0, 0, 1, 1, 0, 0, 1, 1])).astype(np.uint16)
      cellNodeIndicesZ = np.ravel(np.outer(cellIndicesZ, np.ones(nodesPerCell)) + np.array([0, 0, 0, 0, 1, 1, 1, 1])).astype(np.uint16)

      nodeIndices_local = []
      nodesPerDirection = 5

      for i in range(nodesPerDirection):
         for j in range(nodesPerDirection):
            for k in range(nodesPerDirection):
               nodeIndices_local.append(np.array([i,j,k]))
      nodeIndices_local = np.array(nodeIndices_local).astype(np.uint16)

      nodesPerBlock = (int)(nodesPerDirection * nodesPerDirection * nodesPerDirection)


      def calculate_node_indices( self, blockIndicesX, blockIndicesY, blockIndicesZ, nodeIndices_local, nodesPerBlock, cellsPerDirection ):
         nodeIndicesX = np.ravel(np.outer(blockIndicesX, np.ones(nodesPerBlock).astype(np.uint16)) * cellsPerDirection + nodeIndices_local[:,0])
         nodeIndicesY = np.ravel(np.outer(blockIndicesY, np.ones(nodesPerBlock).astype(np.uint16)) * cellsPerDirection + nodeIndices_local[:,1])
         nodeIndicesZ = np.ravel(np.outer(blockIndicesZ, np.ones(nodesPerBlock).astype(np.uint16)) * cellsPerDirection + nodeIndices_local[:,2])
   
         nodeIndices = np.transpose(np.array([nodeIndicesX, nodeIndicesY, nodeIndicesZ], copy=False))

         # Transform indices into unique keys
         nodeKeys = np.sum(nodeIndices * np.array([1, cellsPerDirection*self.__meshes[pop].__vxblocks+1, (cellsPerDirection*self.__meshes[pop].__vxblocks+1)*(cellsPerDirection*self.__meshes[pop].__vyblocks+1)]), axis=1)
         # Sort the keys and delete duplicates
         return np.unique(nodeKeys)
      #nodeIndices = calculate_node_indices( blockIndicesX, blockIndicesY, blockIndicesZ, nodeIndices_local, nodesPerBlock, cellsPerDirection )

      # Put the node indices into keys:
      nodeKeys = np.array([], dtype=np.uint64)
      N = 10
      for i in range(N):
         fromIndex = i*(len(blockIndicesX)//N)
         if i != N-1:
            toIndex = (i+1)*(len(blockIndicesX)//N)
         else:
            toIndex = len(blockIndicesX)
         nodeKeys = np.append(nodeKeys, calculate_node_indices( self, blockIndicesX[fromIndex:toIndex], blockIndicesY[fromIndex:toIndex], blockIndicesZ[fromIndex:toIndex], nodeIndices_local, nodesPerBlock, cellsPerDirection ) )


      # Delete duplicate nodes and sort the list:
      nodeKeys = np.unique(nodeKeys) #We now have all of the nodes in a list!




      def calc_global_cell_keys( self, blockIndicesX, blockIndicesY, blockIndicesZ, cellNodeIndicesX, cellNodeIndicesY, cellNodeIndicesZ, cellsPerBlock, nodesPerCell, cellsPerDirection, nodeKeys ):
         # reate node  indices for the cells
         globalCellIndicesX = np.ravel(np.outer(blockIndicesX, np.ones(cellsPerBlock * nodesPerCell).astype(np.uint16)) * cellsPerDirection + cellNodeIndicesX)
         globalCellIndicesY = np.ravel(np.outer(blockIndicesY, np.ones(cellsPerBlock * nodesPerCell).astype(np.uint16)) * cellsPerDirection + cellNodeIndicesY)
         globalCellIndicesZ = np.ravel(np.outer(blockIndicesZ, np.ones(cellsPerBlock * nodesPerCell).astype(np.uint16)) * cellsPerDirection + cellNodeIndicesZ)
   
         globalCellIndices = np.array([globalCellIndicesX, globalCellIndicesY, globalCellIndicesZ], copy=False)
         globalCellIndices = np.transpose(globalCellIndices)
         # Transform cell indices into unique keys
         globalCellIndices = np.sum(globalCellIndices * np.array([1, cellsPerDirection*self.__meshes[pop].__vxblocks+1, (cellsPerDirection*self.__meshes[pop].__vxblocks+1)*(cellsPerDirection*self.__meshes[pop].__vyblocks+1)]), axis=1)
         # Return cell nodes' indexes in the nodeKeys list
         return np.searchsorted(nodeKeys, globalCellIndices)


      # Create cellKeys
      cellKeys = np.zeros(len(blockIndicesX)*cellsPerBlock*nodesPerCell, dtype=np.uint32)
      N = 10
      # Append keys in cuts to save memory
      for i in range(N):
         fromIndex = i*(len(blockIndicesX)//N)
         if i != N-1:
            toIndex = (i+1)*(len(blockIndicesX)//N)
         else:
            toIndex = len(blockIndicesX)
         # Append cell keys
         cellKeys[fromIndex*cellsPerBlock*nodesPerCell:toIndex*cellsPerBlock*nodesPerCell] = calc_global_cell_keys( self, blockIndicesX[fromIndex:toIndex], blockIndicesY[fromIndex:toIndex], blockIndicesZ[fromIndex:toIndex], cellNodeIndicesX, cellNodeIndicesY, cellNodeIndicesZ, cellsPerBlock, nodesPerCell, cellsPerDirection, nodeKeys )

      cellKeys = np.reshape(cellKeys, (len(blocks)*64,8))

      # We now have all the cell keys and avgs values! (avgs is in the same order as cell keys)
      # Now transform node indices back into real indices
      nodeCoordinatesX = np.remainder(nodeKeys, (int)(cellsPerDirection*self.__meshes[pop].__vxblocks+1)).astype(np.float32) * self.__meshes[pop].__dvx + self.__meshes[pop].__vxmin
      nodeCoordinatesY = np.remainder(nodeKeys//(int)(cellsPerDirection*self.__meshes[pop].__vxblocks+1), cellsPerDirection*self.__meshes[pop].__vyblocks+1).astype(np.float32) * self.__meshes[pop].__dvy + self.__meshes[pop].__vymin
      nodeCoordinatesZ = ( nodeKeys // (int)((cellsPerDirection*self.__meshes[pop].__vxblocks+1) * (cellsPerDirection*self.__meshes[pop].__vyblocks+1)) ).astype(np.float32) * self.__meshes[pop].__dvz + self.__meshes[pop].__vzmin
      
      # Nodekeyss is no longer needed
      del nodeKeys

      nodes = np.array([nodeCoordinatesX, nodeCoordinatesY, nodeCoordinatesZ], copy=False)
      # Take a transpose
      nodes = np.transpose(nodes)

      return [nodes, cellKeys]





   def read_parameter(self, name):
      ''' Read a parameter from the vlsv file

      :param name:   Name of the parameter
      :returns: The parameter value

      .. seealso:: :func:`read_variable` :func:`read_variable_info`
      '''
      
      # Special handling for time
      if name=="time":
         if self.check_parameter(name="t"):
            return self.read(name="t", tag="PARAMETER")
      if name=="t":
         if self.check_parameter(name="time"):
            return self.read(name="time", tag="PARAMETER")

      return self.read(name=name, tag="PARAMETER")


   def read_velocity_cells(self, cellid, pop="proton"):
      ''' Read velocity cells from a spatial cell
      
      :param cellid: Cell ID of the cell whose velocity cells the function will read
      :returns: Map of velocity cell ids (unique for every velocity cell) and corresponding value

      #Example:

      example_cellid = 1111

      velocity_cell_map = vlsvReader.read_velocity_cells(example_cellid)
      velocity_cell_ids = velocity_cell_map.keys()
      velocity_cell_values = velocity_cell_map.values()

      random_index = 4 # Just some index
      random_velocity_cell_id = velocity_cell_ids[random_index]

      print ("Velocity cell value at velocity cell id " + str(random_velocity_cell_id) + ": " + str(velocity_cell_map[random_velocity_cell_id]))

      # Getting the corresponding coordinates might be more useful than having the velocity cell id so:
      velocity_cell_coordinates = vlsvReader.get_velocity_cell_coordinates(velocity_cell_ids) # Get velocity cell coordinates corresponding to each velocity cell id

      random_velocity_cell_coordinates = velocity_cell_ids[random_index]
      print("Velocity cell value at velocity cell id " + str(random_velocity_cell_id) + "and coordinates " + str(random_velocity_cell_coordinates) + ": " + str(velocity_cell_map[random_velocity_cell_id]))

      .. seealso:: :func:`read_blocks`
      '''
      
      if self.use_dict_for_blocks: # old deprecated version, uses dict for blocks data
         if not pop in self.__fileindex_for_cellid_blocks:
            self.__set_cell_offset_and_blocks(pop) 
         # Check that cells has vspace
         if not cellid in self.__fileindex_for_cellid_blocks[pop]:
            warnings.warn("Cell(s) does not have velocity distribution")
            return {}
         # Navigate to the correct position:
         offset = self.__fileindex_for_cellid_blocks[pop][cellid][0]
         num_of_blocks = self.__fileindex_for_cellid_blocks[pop][cellid][1]

      else:  # Uses arrays (much faster to initialize)
         if not pop in self.__cells_with_blocks:
            self.__set_cell_offset_and_blocks_nodict(pop) 
         # Check that cells has vspace
         try:
            cells_with_blocks_index = self.__order_for_cellid_blocks[pop][cellid]
         except:
            warnings.warn("Cell(s) does not have velocity distribution")
            return {}
         # Navigate to the correct position:
         offset = self.__blocks_per_cell_offsets[pop][cells_with_blocks_index]
         num_of_blocks = self.__blocks_per_cell[pop][cells_with_blocks_index]

      if self.__fptr.closed:
         fptr = open(self.file_name,"rb")
      else:
         fptr = self.__fptr

      # Read in avgs and velocity cell ids:
      for child in self.__xml_root:
         # Read in avgs
         if "name" in child.attrib and (child.attrib["name"] == pop) and (child.tag == "BLOCKVARIABLE"):
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            #array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]

            # Navigate to the correct position
            offset_avgs = int(offset * vector_size * element_size + ast.literal_eval(child.text))
            fptr.seek(offset_avgs)

            if datatype == "float" and element_size == 4:
               data_avgs = np.fromfile(fptr, dtype = np.float32, count = vector_size*num_of_blocks) 
            if datatype == "float" and element_size == 8:
               data_avgs = np.fromfile(fptr, dtype = np.float64, count = vector_size*num_of_blocks) 
            data_avgs = data_avgs.reshape(num_of_blocks, vector_size)
         # Read in block coordinates:
         if ("name" in child.attrib) and (child.attrib["name"] == pop) and (child.tag == "BLOCKIDS"):
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            #array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]

            offset_block_ids = int(offset * vector_size * element_size + ast.literal_eval(child.text))
            fptr.seek(offset_block_ids)

            if datatype == "uint" and element_size == 4:
               data_block_ids = np.fromfile(fptr, dtype = np.uint32, count = vector_size*num_of_blocks)
            elif datatype == "uint" and element_size == 8:
               data_block_ids = np.fromfile(fptr, dtype = np.uint64, count = vector_size*num_of_blocks)
            else:
               raise TypeError("Error! Bad data type in blocks! datatype found was "+datatype)

         if (pop=="avgs") and (child.tag == "BLOCKIDS"): # Old avgs files did not have the name set for BLOCKIDS
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            #array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]

            offset_block_ids = int(offset * vector_size * element_size + ast.literal_eval(child.text))
            fptr.seek(offset_block_ids)

            if datatype == "uint" and element_size == 4:
               data_block_ids = np.fromfile(fptr, dtype = np.uint32, count = vector_size*num_of_blocks)
            elif datatype == "uint" and element_size == 8:
               data_block_ids = np.fromfile(fptr, dtype = np.uint64, count = vector_size*num_of_blocks)
            else:
               raise TypeError("Error! Bad data type in blocks! datatype found was "+datatype)

            data_block_ids = data_block_ids.reshape(num_of_blocks, vector_size)

      fptr.close()

      # Check to make sure the sizes match (just some extra debugging)
      if len(data_avgs) != len(data_block_ids):
         raise ValueError("BAD DATA SIZES")
      # Make a dictionary (hash map) out of velocity cell ids and avgs:
      velocity_cells = {}
      array_size = len(data_avgs)

      # Construct velocity cells:
      WID=self.get_WID()
      WID2=WID*WID
      WID3=WID2*WID
      velocity_cell_ids = []
      for kv in range(WID):
         for jv in range(WID):
            for iv in range(WID):
               velocity_cell_ids.append(kv*WID2 + jv*WID + iv)

      for i in range(array_size):
         velocity_block_id = data_block_ids[i]
         avgIndex = 0
         avgs = data_avgs[i]

         for j in velocity_cell_ids + WID3*velocity_block_id:
            velocity_cells[(int)(j)] = avgs[avgIndex]
            avgIndex = avgIndex + 1
      return velocity_cells

   def get_spatial_mesh_size(self):
      ''' Read spatial mesh size
      
      :returns: Size of mesh in number of blocks, array with three elements
      '''
      return np.array([self.__xcells, self.__ycells, self.__zcells])

   def get_spatial_block_size(self):
      ''' Read spatial mesh block size
      
      :returns: Size of block in number of cells, array with three elements
      '''
      return np.array([self.__xblock_size, self.__yblock_size, self.__zblock_size])

   def get_spatial_mesh_extent(self):
      ''' Read spatial mesh extent
      
      :returns: Maximum and minimum coordinates of the mesh, [xmin, ymin, zmin, xmax, ymax, zmax]
      '''
      return np.array([self.__xmin, self.__ymin, self.__zmin, self.__xmax, self.__ymax, self.__zmax])

   def get_fsgrid_mesh_size(self):
      ''' Read fsgrid mesh size
      
      :returns: Size of mesh in number of cells, array with three elements
      '''
      # Get fsgrid domain size (this can differ from vlasov grid size if refined)
      try:
         bbox = self.read(tag="MESH_BBOX", mesh="fsgrid")
         return np.array(bbox[0:3])
      except:
         bbox = self.read(tag="MESH_BBOX", mesh="SpatialGrid")
         return np.array(bbox[0:3]) * 2**self.get_max_refinement_level()

   def get_fsgrid_mesh_extent(self):
      ''' Read fsgrid mesh extent
      
      :returns: Maximum and minimum coordinates of the mesh, [xmin, ymin, zmin, xmax, ymax, zmax]
      '''
      return np.array([self.__xmin, self.__ymin, self.__zmin, self.__xmax, self.__ymax, self.__zmax])

   def get_fsgrid_cell_size(self):
      ''' Read fsgrid cell size
      
      :returns: Maximum and minimum coordinates of the mesh, [dx, dy, dz]
      '''
      size = self.get_fsgrid_mesh_size()
      ext = self.get_fsgrid_mesh_extent()
      ext = ext[3:6]-ext[0:3]
      return ext/size

   def get_fsgrid_indices(self, coords):
      ''' Convert spatial coordinates coords to an index array [xi, yi, zi] for fsgrid

      :returns 3-tuple of integers [xi, yi, zi] corresponding to fsgrid cell containing coords (low-inclusive)
      Example:
      ii = f.get_fsgrid_mesh_extent(coords)
      fsvar_at_coords = fsvar_array.item(ii)
      '''
      lower = self.get_fsgrid_mesh_extent()[0:3]
      dx = self.get_fsgrid_cell_size()
      r0 = coords-lower
      ri = np.floor(r0/dx).astype(int)
      sz = self.get_fsgrid_mesh_size()
      if (ri < 0).any() or (ri>sz-1).any():
         logging.info("get_fsgrid_indices: Resulting index out of bounds, returning None")
         return None
      return tuple(ri)

   def get_fsgrid_slice_indices(self, lower, upper, eps=1e-3):
      ''' Get indices for a subarray of an fsgrid variable, in the cuboid from lower to upper.
      This is meant for mapping a set of fsgrid cells to a given SpatialGrid cell.
      Shifts the corners (lower, upper) by dx_fsgrid*eps inward, if direct low-inclusive behaviour
      is required, set kword eps = 0.


      :returns two 3-tuples of integers.
      Example:
      ii = f.get_fsgrid_mesh_extent(coords)
      fsvar_at_coords = fsvar_array.item(ii)
      '''
      dx = self.get_fsgrid_cell_size()
      eps = dx*eps
      loweri = self.get_fsgrid_indices(lower+eps)
      upperi = self.get_fsgrid_indices(upper-eps)
      return loweri, upperi
      

   def get_velocity_mesh_size(self, pop="proton"):
      ''' Read velocity mesh size
      
      :returns: Size of mesh in number of blocks, array with three elements
      '''
      return np.array([self.__meshes[pop].__vxblocks, self.__meshes[pop].__vyblocks, self.__meshes[pop].__vzblocks])

   def get_velocity_block_size(self, pop="proton"):
      ''' Read velocity mesh block size
      
      :returns: Size of block in number of cells, array with three elements
      '''
      return np.array([self.__meshes[pop].__vxblock_size, self.__meshes[pop].__vyblock_size, self.__meshes[pop].__vzblock_size])

   def get_velocity_mesh_extent(self, pop="proton"):
      ''' Read velocity mesh extent
      
      :returns: Maximum and minimum coordinates of the mesh, [vxmin, vymin, vzmin, vxmax, vymax, vzmax]
      '''
      return np.array([self.__meshes[pop].__vxmin, self.__meshes[pop].__vymin, self.__meshes[pop].__vzmin, self.__meshes[pop].__vxmax, self.__meshes[pop].__vymax, self.__meshes[pop].__vzmax])

   def get_velocity_mesh_dv(self, pop="proton"):
      ''' Read velocity mesh cell size
      
      :returns: Velocity mesh cell size, array with three elements [dvx, dvy, dvz]
      '''
      return np.array([self.__meshes[pop].__dvx, self.__meshes[pop].__dvy, self.__meshes[pop].__dvz])

   def get_ionosphere_mesh_size(self):
      ''' Read size of the ionosphere mesh, if there is one.

      :returns: Size of the mesh in number of nodes and elements, array with two elements
      '''
      try:
         domainsizes = self.read(tag="MESH_DOMAIN_SIZES", mesh="ionosphere")
         return [domainsizes[0], domainsizes[2]]
      except:
         logging.info("Error: Failed to read ionosphere mesh size. Are you reading from a file without ionosphere?")
         return [0,0]

   def get_ionosphere_node_coords(self):
      ''' Read ionosphere node coordinates (in cartesian GSM coordinate system).

      :returns: [x,y,z] array of ionosphere node coordinates (in meters)
      '''
      try:
         coords = np.array(self.read(tag="MESH_NODE_CRDS", mesh="ionosphere")).reshape([-1,3])
         return coords
      except:
         logging.info("Error: Failed to read ionosphere mesh coordinates. Are you reading from a file without ionosphere?")
         return []

   def get_ionosphere_latlon_coords(self):
      ''' Read ionosphere node coordinates (in magnetic longitude / latitude)

      :returns: [lat,lon] array of ionosphere node coordinates
      '''
      coords = self.get_ionosphere_node_coords()
      latlon = np.zeros([coords.shape[0], 2])
      latlon[:,0] = np.arccos(coords[:,2]/6471e3)   # Note, ionosphere height is R_E + 100km
      latlon[:,1] = np.arctan2(coords[:,1],coords[:,0])
      return latlon

   def get_ionosphere_element_corners(self):
      ''' Read ionosphere mesh element corners

      :returns: [c1,c2,c3] array of ionosphere mesh node indices (starting from 0)
      '''
      try:
         meshdata = np.array(self.read(tag="MESH", name="ionosphere")).reshape([-1,5])
         # Elements in meshdata are:
         # - vlsv::celltype::TRIANGLE ("this is a triangle")
         # - 3                        ("it has three corners")
         # - Corner index 1
         # - Corner index 2
         # - Corner index 3
         return meshdata[:,2:5]
      except:
         logging.info("Error: Failed to read ionosphere mesh elements. Are you reading from a file without ionosphere?")
         return []

   def get_ionosphere_mesh_area(self):
      ''' Read areas of ionosphere elements (triangular mesh)

      :returns: 1D array, areas of the triangular elements [m^2]
      '''
      n = self.get_ionosphere_node_coords()       # nodes: shape (n_nodes, 3) vertices
      c = self.get_ionosphere_element_corners()   # corners of elements: indices integers 0-(n_nodes-1), shape (n_elements, 3)
      p = n[c,:]                               # shape(n_elements, 3, 3)   1st index is the element, 2nd index is triangle corners, 3rd index is x-y-z position
      r1 = p[:,1,:] - p[:,0,:]
      r2 = p[:,2,:] - p[:,0,:]
      areas = np.linalg.norm(np.cross(r1, r2), axis = 1) / 2.
      # checked: sum of triangle areas is near the expected area 4*pi*R^2 for a sphere:
      # ( np.sum(areas) - (np.pi * 4 ) * (R_EARTH + 100000.)**2 ) / np.sum(areas) ~ 1
      return areas

   def get_ionosphere_element_coords(self):
      ''' Read coordinates of ionosphere elements (triangle barycenters)

      :returns: [x, y, z] array of ionosphere element barycenter coordinates (in meters).
      '''
      n = self.get_ionosphere_node_coords()          # Nodes, shape (n_nodes, 3)
      ec = self.get_ionosphere_element_corners()     # Element corners, shape (n_elements, 3)
      ig_r = np.zeros(np.array(ec).shape)
      for i in range(ig_r.shape[0]):
         ig_r[i,:] = (n[ec[i,0], :] + n[ec[i,1], :] + n[ec[i,2], :]) / 3  #barycenter, aka centroid
      return ig_r

   def read_ionosphere_node_variable_at_elements(self, varname):
      ''' Interpolate an ionospheric node variavle to the element barycenters

      :param varname: string, specifying variable (or data reducer) defined at ionosphere nodes
      :returns: specified variable interpolated to the elements' barycenters

        note: linear barycentric interpolation is just the sum of 3 corner values divided by 3!

        TODO: check behavior for var.ndim>1
      '''
      var = self.read_variable(varname)
      c = self.get_ionosphere_element_corners()              # (n_elements, 3) node indices
      return np.nansum(var[c], axis = -1) / 3.

   def read_blocks(self, cellid, pop="proton"):
      ''' Read raw block data from the open file and return the data along with block ids
      
      :param cellid: Cell ID of the cell whose velocity blocks are read
      :returns: A numpy array with block ids and data eg [array([2, 5, 6, 234, 21]), array([1.0e-8, 2.1e-8, 2.1e-8, 0, 4.0e-8])]

      .. seealso:: :func:`read_velocity_cells`
      '''
      # Uses new format
      return self.__read_blocks(cellid,pop)

   def get_precipitation_centre_energy(self, pop="proton"):
      ''' Read precipitation energy bins

      :returns: Array of centre energies
      '''
      return self.__meshes[pop].__precipitation_centre_energy

   def optimize_open_file(self):
      '''Opens the vlsv file for reading
         Files are opened and closed automatically upon reading and in the case of reading multiple times it will help to keep the file open with this command

         .. code-block: python

            #Example usage:
            variables = []
            vlsvReader.optimize_open_file()
            for i in range(1000):
               variables.append(vlsvReader.read_variable("rho", cellids=i))
            vlsvReader.optimize_close_file()

         .. note:: This should only be used for optimization purposes.
      '''
      self.__fptr = open(self.file_name,"rb")


   def optimize_close_file(self):
      '''Closes the vlsv file
         Files are opened and closed automatically upon reading and in the case of reading multiple times it will help to keep the file open with this command

         .. code-block: python

            # Example usage:
            variables = []
            vlsvReader.optimize_open_file()
            for i in range(1000):
               variables.append(vlsvReader.read_variable("rho", cellids=i))
            vlsvReader.optimize_close_file()

         .. note:: This should only be used for optimization purposes.
      '''
      if self.__fptr.closed:
         return
      else:
         self.__fptr.close()
         return

   def optimize_clear_fileindex_for_cellid_blocks(self):
      ''' Clears a private variable containing number of blocks and offsets for particular cell ids

         .. code-block: python

             # Example usage:
             vlsvReaders = []
             # Open a list of vlsv files
             for i in range(1000):
                vlsvReaders.append( VlsvReader("test" + str(i) + ".vlsv") )
             # Go through vlsv readers and print info:
             for vlsvReader in vlsvReaders:
                # Print something from the file on the screen
                print( vlsvReader.read_blocks( cellid= 5021 )) # Stores info into a private variable
                # Upon reading from vlsvReader a private variable that contains info on cells that have blocks has been saved -- now clear it to save memory
                vlsvReader.optimize_clear_fileindex_for_cellid_blocks()

         .. note:: This should only be used for optimization purposes.
      '''
      self.__fileindex_for_cellid_blocks = {}
      self.__cells_with_blocks = {}
      self.__blocks_per_cell = {}
      self.__blocks_per_cell_offsets = {}
      self.__order_for_cellid_blocks = {}

   def optimize_clear_fileindex_for_cellid(self):
      ''' Clears a private variable containing cell ids and their locations

         .. code-block: python

             # Example usage:
             vlsvReaders = []
             # Open a list of vlsv files
             for i in range(1000):
                vlsvReaders.append( VlsvReader("test" + str(i) + ".vlsv") )
             # Go through vlsv readers and logging.info info:
             for vlsvReader in vlsvReaders:
                # Print something from the file on the screen
                logging.info vlsvReader.read_variable("B", cellids=2) # Stores info into a private variable
                # Upon reading from vlsvReader a private variable that contains info on cells that have blocks has been saved -- now clear it to save memory
                vlsvReader.optimize_clear_fileindex_for_cellid()

         .. note:: This should only be used for optimization purposes.
      '''
      self.__fileindex_for_cellid = {}


