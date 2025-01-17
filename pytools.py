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
import warnings
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=os.environ.get('ANALYSATOR_LOG_LEVEL', 'INFO').upper())
warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("once", category=PendingDeprecationWarning)
warnings.filterwarnings("once", category=FutureWarning)

# Input current folder's path
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)))
# Input folder paths
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "miscellaneous")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyCalculations")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyCellDataReduction")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyPlots")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyVisit")
filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyVlsv")
if os.getenv('PTMAYAVI2') != None:
   filemanagement.sys.path.insert(0, filemanagement.os.path.dirname(filemanagement.os.path.abspath(__file__)) + "/" + "pyMayaVi")


# Make sure matplotlib has a unique temp directory
mpldir = tempfile.mkdtemp()
atexit.register(shutil.rmtree, mpldir)
os.environ['MPLCONFIGDIR']=mpldir

# Check if user is on taito.csc.fi without loading the mayavi2 module
import numpy as np
import matplotlib
if matplotlib.__version__=="0.99.1.1" and np.__version__=="1.4.1":
   logging.info('Warning, according to loaded numpy and matplotlib versions, user appears to be')
   logging.info('either using csc.taito.fi without loading the mayavi2 module, or by invoking')
   logging.info('the system python interpeter by calling "./scriptname.py" instead of "python ./scriptname.py"')

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Run TeX typesetting through the full TeX engine instead of python's own mathtext. Allows
# for changing fonts, bold math symbols etc, but may cause trouble on some systems.
if not os.getenv('PTNOLATEX'):
   matplotlib.rc('text', usetex=True)
   matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
   # matplotlib.rcParams['mathtext.fontset'] = 'stix'
   # matplotlib.rcParams['font.family'] = 'STIXGeneral'
   # Matplotlib suppressed logging messages came out after enabling logging.INFO: font.family must be one of (serif, sans-serif, cursive, monospace) when text.usetex is True. serif will be used by default.
   matplotlib.rcParams['font.family'] = 'serif'
   logging.info("Using LaTeX formatting")
   # matplotlib.rcParams['text.dvipnghack'] = 'True' # This hack might fix it on some systems

# Set backends
if matplotlib.get_backend()[:9] == 'module://':
   logging.info("Using backend "+matplotlib.get_backend())
   backend_interactive = matplotlib.get_backend()
   backend_noninteractive = matplotlib.get_backend()
elif not os.getenv('PTBACKEND'):
   backend_interactive = 'TkAgg'
   backend_noninteractive = 'Agg'
else:
   backend_interactive = os.getenv('PTBACKEND')
   backend_noninteractive = os.getenv('PTBACKEND')

# Import modules
try:
   import calculations
except ImportError as e:
   logging.info("Note: Did not import calculations module: " + str(e))

try:
   import vlsvfile
except ImportError as e:
   logging.info("Note: Did not import vlsvfile module: " + str(e))

import os
import matplotlib.pyplot as plt

if os.getenv('PTNONINTERACTIVE') != None:
   # Non-interactive plotting mode
   try:
      plt.switch_backend(backend_noninteractive)
   except:
      logging.info("Note: Unable to switch to "+backend_noninteractive+" backend")
else:
   # Interactive plotting mode
   plt.ion()
   try:
      plt.switch_backend(backend_interactive)
   except:
      logging.info("Note: Unable to switch to "+backend_interactive+" backend")

   #Only attempt loading MayaVi2 if requested
   if os.getenv('PTMAYAVI2') != None:
      try:
         import grid
      except ImportError as e:
         logging.info("Note: Did not import (outdated MayaVi2) grid module: " + str(e))

try:
   import plot
except ImportError as e:
   logging.info("Note: Did not import plot module: " + str(e))

try:
   import miscellaneous
except ImportError as e:
   logging.info("Note: Did not import miscellaneous: " + str(e))

