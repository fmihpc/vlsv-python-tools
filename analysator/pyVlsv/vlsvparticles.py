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

class VlsvParticles(object):
   ''' Class for reading VLSV files of particle pusher output
   ''' 
   file_name=""
   def __init__(self, file_name):
      ''' Initializes the vlsv file (opens the file, reads the file footer and reads in some parameters)

          :param file_name:     Name of the vlsv file
      '''
      # Make sure the path is set in file name: 
      file_name = os.path.abspath(file_name)

      self.file_name = file_name
      self.__fptr = open(self.file_name,"rb")
      self.__xml_root = ET.fromstring("<VLSV></VLSV>")
      self.__fileindex_for_cellid={}
      self.__fileindex_for_cellid_blocks={}
      self.__read_xml_footer()
      # Check if the file is using new or old vlsv format
      # Read parameters (Note: Reading the spatial cell locations and
      # storing them will anyway take the most time and memory):

      self.__fptr.close()


   def __read_xml_footer(self):
      ''' Reads in the XML footer of the VLSV file and store all the content
      ''' 
      max_xml_size = 1000000
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
      xml_data = fptr.read(max_xml_size)
      # Read the xml as string
      (xml_string,) = struct.unpack("%ds" % len(xml_data), xml_data)
      # Input the xml data into xml_root
      self.__xml_root = ET.fromstring(xml_string)
      if self.__fptr.closed:
         fptr.close()

   def list(self):
      ''' Print out a description of the content of the file. Useful
         for interactive usage
      '''
      logging.info("tag = MESH")
      for child in self.__xml_root:
         if child.tag == "MESH" and "name" in child.attrib:
            logging.info("   " + str(child.attrib["name"]))

   def read_particles_all(self):
      ''' Read particle pusher data from the open vlsv file.
          This routine returns all particles, including inactive ones.
      
      '''

      if self.__fptr.closed:
         fptr = open(self.file_name,"rb")
      else:
         fptr = self.__fptr

      proton_position=None
      proton_velocity=None
      for child in self.__xml_root:
          if child.attrib["name"] == 'proton_position':
              vector_size = ast.literal_eval(child.attrib["vectorsize"])
              array_size = ast.literal_eval(child.attrib["arraysize"])
              element_size = ast.literal_eval(child.attrib["datasize"])
              datatype = child.attrib["datatype"]

              offset = ast.literal_eval(child.text)
              fptr.seek(offset)

              if datatype == "float" and element_size == 4:
                  data = np.fromfile(fptr, dtype = np.float32, count=vector_size*array_size)
              if datatype == "float" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.float64, count=vector_size*array_size)
              if datatype == "int" and element_size == 4:
                  data = np.fromfile(fptr, dtype=np.int32, count=vector_size*array_size)
              if datatype == "int" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.int64, count=vector_size*array_size)
              if datatype == "uint" and element_size == 4:
                  data = np.fromfile(fptr, dtype=np.uint32, count=vector_size*array_size)
              if datatype == "uint" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.uint64, count=vector_size*array_size)

              data=data.reshape(array_size, vector_size)         
              proton_position = data

          if child.attrib["name"] == 'proton_velocity':
              vector_size = ast.literal_eval(child.attrib["vectorsize"])
              array_size = ast.literal_eval(child.attrib["arraysize"])
              element_size = ast.literal_eval(child.attrib["datasize"])
              datatype = child.attrib["datatype"]

              offset = ast.literal_eval(child.text)
              fptr.seek(offset)

              if datatype == "float" and element_size == 4:
                  data = np.fromfile(fptr, dtype = np.float32, count=vector_size*array_size)
              if datatype == "float" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.float64, count=vector_size*array_size)
              if datatype == "int" and element_size == 4:
                  data = np.fromfile(fptr, dtype=np.int32, count=vector_size*array_size)
              if datatype == "int" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.int64, count=vector_size*array_size)
              if datatype == "uint" and element_size == 4:
                  data = np.fromfile(fptr, dtype=np.uint32, count=vector_size*array_size)
              if datatype == "uint" and element_size == 8:
                  data = np.fromfile(fptr, dtype=np.uint64, count=vector_size*array_size)

              data=data.reshape(array_size, vector_size)         
              proton_velocity = data

      if self.__fptr.closed:
         fptr.close()

      return proton_position, proton_velocity

   def read_particles(self):
      ''' Read particle pusher data from the open vlsv file. 
          This routine returns only active particles.
      '''

      proton_position, proton_velocity = self.read_particles_all()
      isvalid = np.isfinite(proton_position[:,0])
      proton_position = proton_position[isvalid,:]
      proton_velocity = proton_velocity[isvalid,:]

      return proton_position, proton_velocity

