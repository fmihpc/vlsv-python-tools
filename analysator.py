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

import filemanagement
import socket, re, os, tempfile, atexit, shutil

# Input current folder's path
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)))
# Input folder paths
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "miscellaneous")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyCalculations")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyCellDataReduction")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyMayaVi")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyPlots")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyVisit")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyVlsv")


# Make sure matplotlib has a unique temp directory
mpldir = tempfile.mkdtemp()
atexit.register(shutil.rmtree, mpldir)
os.environ['MPLCONFIGDIR']=mpldir

# Check if user is on taito.csc.fi without loading the mayavi2 module
import numpy as np
import matplotlib
if matplotlib.__version__=="0.99.1.1" and np.__version__=="1.4.1":
   print('Warning, according to loaded numpy and matplotlib versions, user appears to be')
   print('either using csc.taito.fi without loading the mayavi2 module, or by invoking')
   print('the system python interpeter by calling "./scriptname.py" instead of "python ./scriptname.py"')

# Run TeX typesetting through the full TeX engine instead of python's own mathtext. Allows
# for changing fonts, bold math symbols etc, but may cause trouble on some systems.
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['text.dvipnghack'] = 'True' # This hack might fix it on some systems


# Import modules
try:
   import calculations
except ImportError as e:
   print("Note: Did not import calculations module: ", e)

try:
   import vlsvfile
except ImportError as e:
   print("Note: Did not import vlsvfile module: ", e)

import os
import matplotlib.pyplot as plt

if os.getenv('PTNONINTERACTIVE') != None:
   # Non-interactive plotting mode
   try:
      plt.switch_backend('Agg')
   except:
      print("Note: Unable to switch to Agg backend")
else:
   # Interactive plotting mode
   plt.ion()
   try:
      import grid
   except ImportError as e:
      print("Note: Did not import grid module: ", e)
   try:
      plt.switch_backend('TkAgg')
   except:
      print("Note: Unable to switch to TkAgg backend")

try:
   import plot
except ImportError as e:
   print("Note: Did not import plot module: ", e)

try:
   import miscellaneous
except ImportError as e:
   print("Note: Did not import miscellaneous: ", e)

