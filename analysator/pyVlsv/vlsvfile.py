# 
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

''' The vlsvfile module has all the classes related to reading and writing vlsv files


   .. code-block:: python

      # Example:
      import analysator as pt
      pt.vlsvfile.
      #press [tab] -> get the functions

'''

import logging
from vlsvreader import VlsvReader
from vlsvreader import fsDecompositionFromGlobalIds,fsReadGlobalIdsPerRank,fsGlobalIdToGlobalIndex
from vlsvwriter import VlsvWriter
from vlasiatorreader import VlasiatorReader

from vlsvparticles import VlsvParticles
import reduction
import reduction_sidecar
