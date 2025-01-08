
import socket, re, os, tempfile, atexit, shutil, sys
import warnings
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=os.environ.get('ANALYSATOR_LOG_LEVEL', 'INFO').upper())

# Input current folder's path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Input folder paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "miscellaneous")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pyCalculations")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pyPlots")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pyVlsv")
if os.getenv('PTMAYAVI2') != None:
   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/" + "pyMayaVi")


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



# from analysator.pytools import *
# import warnings
# warnings.filterwarnings("once", category=DeprecationWarning)
# warnings.filterwarnings("once", category=PendingDeprecationWarning)
# warnings.filterwarnings("once", category=FutureWarning)


# # from pytools import vslvfile

# from os import path as __path
# root = __path.dirname(__file__)
# with open(__path.join(root,'pytools.py'),'r') as f:
#     source = f.read()
#     exec(source)
#     f.close()
#     del f
