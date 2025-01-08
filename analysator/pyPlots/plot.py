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

''' The plot module has all the functions related to plotting variables.

   .. code-block:: python

      # Example:
      import analysator as pt
      pt.pt.
      #press [tab] -> get the functions

'''

from plot_variables import plot_variables, plot_multiple_variables


import logging
import matplotlib.pyplot as plt
import matplotlib
import colormaps as cmaps

import plot_helpers
from plot_colormap import plot_colormap
from plot_vdf import plot_vdf
from plot_vdfdiff import plot_vdfdiff
from plot_vdf_profiles import plot_vdf_profiles
from plot_colormap3dslice import plot_colormap3dslice
from plot_threeslice import plot_threeslice
from plot_ionosphere import plot_ionosphere

try:
    from plot_isosurface import plot_isosurface, plot_neutral_sheet
except:
    logging.info("plot_isosurface not imported. To access it, use Python version >3.8 and install scikit-image.")

from packaging.version import Version

import numpy as np, os

if Version(matplotlib.__version__) < Version("3.3.0"):
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.register_cmap(name='viridis_r', cmap=cmaps.viridis_r)
    plt.register_cmap(name='plasma', cmap=cmaps.plasma)
    plt.register_cmap(name='plasma_r', cmap=cmaps.plasma_r)
    plt.register_cmap(name='inferno', cmap=cmaps.inferno)
    plt.register_cmap(name='inferno_r', cmap=cmaps.inferno_r)
    plt.register_cmap(name='magma', cmap=cmaps.magma)
    plt.register_cmap(name='magma_r', cmap=cmaps.magma_r)

# Register custom colourmaps
if Version(matplotlib.__version__) < Version("3.5.0"):
    plt.register_cmap(name='parula', cmap=cmaps.parula)
    plt.register_cmap(name='parula_r', cmap=cmaps.parula_r)
    plt.register_cmap(name='hot_desaturated', cmap=cmaps.hot_desaturated_colormap)
    plt.register_cmap(name='hot_desaturated_r', cmap=cmaps.hot_desaturated_colormap_r) # Listed colormap requires making reversed version at earlier step
    plt.register_cmap(name='pale_desaturated', cmap=cmaps.pale_desaturated_colormap)
    plt.register_cmap(name='pale_desaturated_r', cmap=cmaps.pale_desaturated_colormap_r) # Listed colormap requires making reversed version at earlier step
    plt.register_cmap(name='warhol', cmap=cmaps.warhol_colormap)
else:
    matplotlib.colormaps.register(name='parula', cmap=cmaps.parula)
    matplotlib.colormaps.register(name='parula_r', cmap=cmaps.parula_r)
    matplotlib.colormaps.register(name='hot_desaturated', cmap=cmaps.hot_desaturated_colormap)
    matplotlib.colormaps.register(name='hot_desaturated_r', cmap=cmaps.hot_desaturated_colormap_r) # Listed colormap requires making reversed version at earlier step
    matplotlib.colormaps.register(name='pale_desaturated', cmap=cmaps.pale_desaturated_colormap)
    matplotlib.colormaps.register(name='pale_desaturated_r', cmap=cmaps.pale_desaturated_colormap_r) # Listed colormap requires making reversed version at earlier step
    matplotlib.colormaps.register(name='warhol', cmap=cmaps.warhol_colormap)


decimalprecision_ax = 0
decimalprecision_cblin = 0
cb_linear = False

# Output matplotlib version
logging.info("Using matplotlib version "+matplotlib.__version__)

# Default output directory for plots
defaultoutputdir=os.path.expandvars('$HOME/Plots/')
if os.getenv('PTOUTPUTDIR'):
    defaultoutputdir=os.getenv('PTOUTPUTDIR')

# axisfmt replaces minus sign with en-dash to fix bug with latex descender value return
# nb: axis ticks are never plotted with scientific format
def axisfmt(x, pos):
    f = r'{:.'+decimalprecision_ax+r'f}'
    if not os.getenv('PTNOLATEX'):
        a = f.format(abs(x))
        if np.sign(x)<0: a = r'\mbox{\textbf{--}}'+a
        return r'$'+a+'$'
    else:
        return f.format(x)

# cbfmtsci replaces minus sign with en-dash to fix bug with latex descender value return
# Scientific format for colour bar ticks
def cbfmtsci(x, pos):
    if (cb_linear is True):
        # for linear, use more precision
        a, b = ('{:.'+str(int(decimalprecision_cblin))+'e}').format(x).split('e')
        precisionvalue = int(decimalprecision_cblin)
        if int(b) < 0:
            precisionvalue += int(b)
        if abs(precisionvalue)<1:
            precisionvalue=1
        f = '{:.' + str(abs(precisionvalue)) + 'f}'
        number = f.format(abs(float(a)))+r'{\times}'+'10^{{{}}}'.format(int(b))
    else:
        a, b = '{:.1e}'.format(x).split('e')
        number = '{:.1f}'.format(abs(float(a)))+r'{\times}'+'10^{{{}}}'.format(int(b))
    signchar=r''
    # Multiple braces for b take care of negative values in exponent
    # brackets around \times remove extra whitespace
    if not os.getenv('PTNOLATEX'):
        # replaces minus sign with en-dash to fix big with latex descender value return
        if np.sign(x)<0: signchar=r'\mbox{\textbf{--}}'
    else:
        if np.sign(x)<0: signchar=r'-'
    # Final special treatment for zero value
    if x==0:
        number = r'0.0{\times}10^{{{0}}}'
    return r'$'+signchar+number+'$'

# cbfmt replaces minus sign with en-dash to fix bug with latex descender value return, used for colorbar
# nb: regular floating i.e. non-scientific format for colorbar ticks
def cbfmt(x, pos):
    # Special treatment for zero value
    if x==0:
        return r'$0.0$'
    # Set required decimal precision
    a, b = '{:.1e}'.format(x).split('e')
    # e.g. 9.0e-1 means we need precision 1
    if (cb_linear is True):
        # for linear, use more precision
        precision = str(int(decimalprecision_cblin))
        #if int(b)<1: precision = str(1+abs(int(b)))
    else:
        precision = '0'
        if int(b)<1: precision = str(abs(int(b)))
    f = r'{:.'+precision+'f}'
    if not os.getenv('PTNOLATEX'):
        a = f.format(abs(x))
        if np.sign(x)<0: a = r'\mbox{\textbf{--}}'+a
        return r'$'+a+'$'
    else:
        return f.format(x)


# Helper routines for latex output handling
def bfstring(string):
    if not os.getenv('PTNOLATEX'):
        if len(string)==0:
            return ''
        else:
            return r'\mathbf{'+string+'}'
    # LaTeX output off
    return string

def rmstring(string):
    if len(string)==0:
        return ''
    else:
        return r'\mathrm{'+string+'}'

def mathmode(string):
    if len(string)==0:
        return ''
    else:
        # First remove any internal possible dollar signs, then wrap whole string into math block
        result = string.replace('$','')
        if os.getenv('PTNOLATEX'):
            # Get rid of latex spaces
            result = result.replace(r'\,','~').replace(r'\qquad','~~~~~~')
        return r"$"+result+"$"

def textbfstring(string):
    if not os.getenv('PTNOLATEX'):
        if len(string)==0:
            return ''
        else:
            return r'\textbf{'+string+'}'
    # LaTex output off
    return string
