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

import matplotlib
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.patches import Circle, Wedge
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Register custom colourmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='viridis_r', cmap=matplotlib.colors.ListedColormap(cmaps.viridis.colors[::-1]))
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
plt.register_cmap(name='plasma_r', cmap=matplotlib.colors.ListedColormap(cmaps.plasma.colors[::-1]))
plt.register_cmap(name='inferno', cmap=cmaps.inferno)
plt.register_cmap(name='inferno_r', cmap=matplotlib.colors.ListedColormap(cmaps.inferno.colors[::-1]))
plt.register_cmap(name='magma', cmap=cmaps.magma)
plt.register_cmap(name='magma_r', cmap=matplotlib.colors.ListedColormap(cmaps.magma.colors[::-1]))
plt.register_cmap(name='parula', cmap=cmaps.parula)
plt.register_cmap(name='parula_r', cmap=matplotlib.colors.ListedColormap(cmaps.parula.colors[::-1]))
# plt.register_cmap(name='cork',cmap=cork_map)
# plt.register_cmap(name='davos_r',cmap=davos_r_map)
plt.register_cmap(name='hot_desaturated', cmap=cmaps.hot_desaturated_colormap)
plt.register_cmap(name='hot_desaturated_r', cmap=cmaps.hot_desaturated_colormap_r) # Listed colormap requires making reversed version at earlier step
plt.register_cmap(name='pale_desaturated', cmap=cmaps.pale_desaturated_colormap)
plt.register_cmap(name='pale_desaturated_r', cmap=cmaps.pale_desaturated_colormap_r) # Listed colormap requires making reversed version at earlier step

plt.register_cmap(name='warhol', cmap=cmaps.warhol_colormap)

# Different style scientific format for colour bar ticks
def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    # this should bring all colorbar ticks to the same horizontal position, but for
    # some reason it doesn't work. (signchar=r'\enspace')
    signchar=r'' 
    # replaces minus sign with en-dash to fix big with latex descender value return
    if np.sign(x)<0: signchar=r'\mbox{\textbf{--}}'
    # Multiple braces for b take care of negative values in exponent
    # brackets around \times remove extra whitespace
    return r'$'+signchar+'{}'.format(abs(float(a)))+r'{\times}'+'10^{{{}}}$'.format(int(b))

# axisfmt replaces minus sign with en-dash to fix big with latex descender value return
def axisfmt(x, pos):
    # Find out required decimal precision
    a, b = '{:.1e}'.format(np.amax(abs(np.array(plot_colormap.boxcoords)))).split('e')
    precision = '0'
    if int(b)<1: precision = str(abs(-1-int(b)))
    f = r'{:.'+precision+r'f}'
    a = f.format(abs(x))
    if np.sign(x)<0: a = r'\mbox{\textbf{--}}'+a
    return r'$'+a+'$'

# cbfmt replaces minus sign with en-dash to fix big with latex descender value return, used for colorbar
def cbfmt(x, pos):
    # Find out required decimal precision
    a, b = '{:.1e}'.format(x).split('e')
    precision = '0'
    if (plot_colormap.lin is None):
        if int(b)<1: precision = str(abs(int(b)))
    else:
        # for linear, use more precision
        if int(b)<1: precision = str(abs(-1+int(b)))

    f = r'{:.'+precision+r'f}'
    a = f.format(abs(x))
    if np.sign(x)<0: a = r'\mbox{\textbf{--}}'+a
    return r'$'+a+'$'

def plot_colormap(filename=None,
                  vlsvobj=None,
                  filedir=None, step=None,
                  outputdir=None, outputfile=None,
                  nooverwrite=None,
                  var=None, op=None, operator=None,
                  title=None, cbtitle=None, draw=None, usesci=True,
                  symlog=None,
                  boxm=[],boxre=[],colormap=None,
                  run=None, nocb=None, internalcb=None,
                  wmark=None,wmarkb=None,
                  axisunit=None, thick=1.0,scale=1.0,
                  tickinterval=None,
                  noborder=None, noxlabels=None, noylabels=None,
                  vmin=None, vmax=None, lin=None,
                  external=None, expression=None, 
                  vscale=1.0,
                  pass_vars=None, pass_times=None, pass_full=None,
                  fluxfile=None, fluxdir=None,
                  fluxthick=1.0, fluxlines=1,
                  fsaved=None,
                  Earth=None,
                  highres=None,
                  vectors=None, vectordensity=100, vectorcolormap='gray', vectorsize=1.0,
                  streamlines=None, streamlinedensity=1, streamlinecolor='white',streamlinethick=1.0,
                  axes=None, cbaxes=None,
                  ):

    ''' Plots a coloured plot with axes and a colour bar.

    :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj:     Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename
    :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/)
                        If directory does not exist, it will be created. If the string does not end in a
                        forward slash, the final parti will be used as a perfix for the files.
    :kword outputfile:  Singular output file name

    :kword nooverwrite: Set to only perform actions if the target output file does not yet exist                    

    :kword var:         variable to plot, e.g. rho, RhoBackstream, beta, Temperature, MA, Mms, va, vms,
                        E, B, v, V or others. Accepts any variable known by analysator.
                        Per-population variables are simply given as "proton/rho" etc
    :kword operator:    Operator to apply to variable: None, x, y, or z. Vector variables return either
                        the queried component, or otherwise the magnitude. 
    :kword op:          duplicate of operator
           
    :kword boxm:        zoom box extents [x0,x1,y0,y1] in metres (default and truncate to: whole simulation box)
    :kword boxre:       zoom box extents [x0,x1,y0,y1] in Earth radii (default and truncate to: whole simulation box)
    :kword colormap:    colour scale for plot, use e.g. hot_desaturated, jet, viridis, plasma, inferno,
                        magma, parula, nipy_spectral, RdBu, bwr
    :kword run:         run identifier, used for constructing output filename
    :kword title:       string to use as plot title instead of time.
                        Special case: Set to "msec" to plot time with millisecond accuracy or "musec"
                        for microsecond accuracy. "sec" is integer second accuracy.
    :kword cbtitle:     string to use as colorbar title instead of map name
    :kword axisunit:    Plot axes using 10^{axisunit} m (default: Earth radius R_E)
    :kword tickinterval: Interval at which to have ticks on axes (not colorbar)

    :kwird usesci:      Use scientific notation for colorbar ticks? (default: True)
    :kword vmin,vmax:   min and max values for colour scale and colour bar. If no values are given,
                        min and max values for whole plot (non-zero rho regions only) are used.
    :kword lin:         Flag for using linear colour scaling instead of log. If an integer, defines number
                        of colorbar ticks.
    :kword symlog:      Use logarithmic scaling, but linear when abs(value) is below the value given to symlog.
                        Allows symmetric quasi-logarithmic plots of e.g. transverse field components.
                        A given of 0 translates to a threshold of max(abs(vmin),abs(vmax)) * 1.e-2, but this can
                        result in the innermost tick marks overlapping. In this case, using a larger value for 
                        symlog is suggested.
    :kword wmark:       If set to non-zero, will plot a Vlasiator watermark in the top left corner. If set to a text
                        string, tries to use that as the location, e.g. "NW","NE","SW","SW"
    :kword wmarkb:      As for wmark, but uses an all-black Vlasiator logo.
    :kword Earth:       If set, draws an earth at (0,0)
    :kword highres:     Creates the image in high resolution, scaled up by this value (suitable for print). 

    :kword draw:        Set to anything but None in order to draw image on-screen instead of saving to file (requires x-windowing)

    :kword noborder:    Plot figure edge-to-edge without borders (default off)
    :kword noxlabels:   Suppress x-axis labels and title
    :kword noylabels:   Suppress y-axis labels and title
    :kword scale:       Scale text size (default=1.0)
    :kword thick:       line and axis thickness, default=1.0
    :kword nocb:        Set to suppress drawing of colourbar
    :kword internalcb:  Set to draw colorbar inside plot instead of outside. If set to a text
                        string, tries to use that as the location, e.g. "NW","NE","SW","SW"

    :kword external:    Optional function to use for external plotting of e.g. contours. The function
                        receives the following arguments: ax, XmeshXY,YmeshXY, pass_maps
                        If the function accepts a fifth variable, if set to true, it is expected to 
                        return a list of required variables for constructing the pass_maps dictionary.
    :kword expression:  Optional function which calculates a custom expression to plot. The function
                        receives the same dictionary of numpy arrays as external, as an argument pass_maps,
                        the contents of which are maps of variables. Each is either of size [ysize,xsize]
                        or for multi-dimensional variables (vectors, tensors) it's [ysize,xsize,dim].
                        If the function accepts a second variable, if set to true, it is expected to 
                        return a list of required variables for pass_maps.

    Important note: the dictionaries of arrays passed to external and expression are of shape [ysize,xzize], so
    for some analysis transposing them is necessary. For pre-existing functions to use and to base new functions
    on, see the plot_helpers.py file.

    :kword vscale:      Scale all values with this before plotting. Useful for going from e.g. m^-3 to cm^-3
                        or from tesla to nanotesla. Guesses correct units for colourbar for some known
                        variables.

    :kword pass_vars:   Optional list of map names to pass to the external/expression functions 
                        as a dictionary of numpy arrays. Each is either of size [ysize,xsize] or 
                        for multi-dimensional variables (vectors, tensors) it's [ysize,xsize,dim].
    :kword pass_times:  Integer, how many timesteps in each direction should be passed to external/expression
                        functions in pass_vars (e.g. pass_times=1 passes the values of three timesteps). If
                        pass_times has two values, the first is the extent before, the second after.
                        (e.g. pass_times=[2,1] passes the values of two preceding and one following timesteps
                        for a total of four timesteps)
                        This causes pass_vars to become a list of timesteps, with each timestep containing
                        a dictionary of numpy arrays as for regular pass_vars. An additional dictionary entry is
                        added as 'dstep' which gives the timestep offset from the master frame.
                        Does not work if working from a vlsv-object.
    :kword pass_full:   Set to anything but None in order to pass the full arrays instead of a zoomed-in section

    :kword fluxfile:    Filename to plot fluxfunction from
    :kword fluxdir:     Directory in which fluxfunction files can be found
    :kword fluxthick:   Scale fluxfunction line thickness
    :kword fluxlines:   Relative density of fluxfunction contours
    :kword fsaved:      Overplot locations of fSaved. If keyword is set to a string, that will be the colour used.

    :kword vectors:     Set to a vector variable to overplot (unit length vectors, color displays variable magnitude)
    :kword vectordensity: Aim for how many vectors to show in plot window (default 100)
    :kword vectorcolormap: Colormap to use for overplotted vectors (default: gray)
    :kword vectorsize:  Scaling of vector sizes

    :kword streamlines: Set to a vector variable to overplot as streamlines
    :kword streamlinedensity: Set streamline density (default 1)
    :kword streamlinecolor: Set streamline color (default white)
    :kword streamlinethick: Set streamline thickness

    :kword axes:        Provide the routine a set of axes to draw within instead of generating a new image.
                        It is recommended to either also provide cbaxes or activate nocb, unless one wants a colorbar
                        to be automatically added next to the panel (but this may affect the overall layout)
                        Note that the aspect ratio of the colormap is made equal in any case, hence the axes
                        proportions may change if the box and axes size are not designed to match by the user
    :kword cbaxes:      Provide the routine a set of axes for the colourbar.

    :returns:           Outputs an image to a file or to the screen.

    .. code-block:: python

    # Example usage:
    plot_colormap(filename=fileLocation, var="MA", run="BCQ",
                  colormap='nipy_spectral',step=j, outputdir=outputLocation,
                  lin=True, wmark=1, vmin=2.7, vmax=10, 
                  external=cavitoncontours, pass_vars=['rho','B','beta'])
    # Where cavitoncontours is an external function which receives the arguments
    #  ax, XmeshXY,YmeshXY, pass_maps
    # where pass_maps is a dictionary of maps for the requested variables.

    # example (simple) use of expressions:
    def exprMA_cust(exprmaps, requestvariables=False):
        if requestvariables==True:
           return ['va']
        custombulkspeed=750000. # m/s
        va = exprmaps['va'][:,:]
        MA = custombulkspeed/va
        return MA
    plot_colormap(filename=fileLocation, vmin=1 vmax=40, expression=exprMA_cust,lin=True)

    '''

    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    watermarkimageblack=os.path.join(os.path.dirname(__file__), 'logo_black.png')
    # watermarkimage=os.path.expandvars('$HOME/appl_taito/analysator/pyPlot/logo_color.png')

    # Input file or object
    if filename!=None:
        f=pt.vlsvfile.VlsvReader(filename)
    elif ((filedir!=None) and (step!=None)):
        filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
        f=pt.vlsvfile.VlsvReader(filename)
    elif vlsvobj!=None:
        f=vlsvobj
    else:
        print("Error, needs a .vlsv file name, python object, or directory and step")
        return

    # Flux function files
    if fluxdir!=None:
        if step != None:
            fluxfile = fluxdir+'flux.'+str(step).rjust(7,'0')+'.bin'
            if not os.path.exists(fluxfile):
                fluxfile = fluxdir+'bulk.'+str(step).rjust(7,'0')+'.bin'
        else:
            if filename!=None:
                # Parse step from filename
                fluxfile = fluxdir+'flux.'+filename[-12:-5]+'.bin'
                if not os.path.exists(fluxfile):
                    fluxfile = fluxdir+'bulk.'+filename[-12:-5]+'.bin'
            else:
                print("Requested flux lines via directory but working from vlsv object, cannot find step.")

    if fluxfile!=None:
        if not os.path.exists(fluxfile):
            print("Error locating flux function file!")
            fluxfile=None
                
    # Scientific notation for colorbar ticks?
    if usesci is not True:
        usesci=False
    
    if operator==None:
        if op!=None:
            operator=op

    if colormap==None:
        # Default values
        colormap="hot_desaturated"
        if operator=='x' or operator=='y' or operator=='z':
            colormap="bwr"
    cmapuse=matplotlib.cm.get_cmap(name=colormap)

    fontsize=8*scale # Most text
    fontsize2=10*scale # Time title
    fontsize3=8*scale # Colour bar ticks and title

    # Plot title with time
    timeval=f.read_parameter("time")

    # Plot title with time
    if title==None or title=="msec" or title=="musec":        
        if timeval == None:    
            plot_title = ''
        else:
            timeformat='{:4.1f}'
            if title=="sec": timeformat='{:4.0f}'
            if title=="msec": timeformat='{:4.3f}'
            if title=="musec": timeformat='{:4.6f}'
            plot_title = "t="+timeformat.format(timeval)+' s'
    else:
        plot_title = title

    # step, used for file name
    if step!=None:
        stepstr = '_'+str(step).rjust(7,'0')
    else:
        if filename!=None:
            stepstr = '_'+filename[-12:-5]
        else:
            stepstr = ''

    # If run name isn't given, just put "plot" in the output file name
    if run==None:
        run='plot'
        if filename!=None:
            # If working within CSC filesystem, make a guess:
            if filename[0:16]=="/proj/vlasov/2D/":
                run = filename[16:19]

    # Verify validity of operator
    operatorstr=''
    if operator!=None:
        # .isdigit checks if the operator is an integer (for taking an element from a vector)
        if type(operator) is int:
            operator = str(operator)
        if operator!='x' and operator!='y' and operator!='z' and operator!='magnitude' and not operator.isdigit():
            print("Unknown operator "+operator)
            operator=None
        if operator=='x' or operator=='y' or operator=='z':
            # For components, always use linear scale, unless symlog is set
            operatorstr='_'+operator
            if symlog==None and lin is None:
                lin=True
        # index a vector
        if operator.isdigit():
            operator = str(operator)
            operatorstr='_{'+operator+'}'

    # Output file name
    if expression!=None:
        varstr=expression.__name__.replace("/","_")
    else:        
        if var==None:
            # If no expression or variable given, defaults to rho
            var='rho'
            if f.check_variable("proton/vg_rho"): # multipop v5
                var = 'proton/vg_rho'
            elif f.check_variable("proton/rho"): # multipop
                var = 'proton/rho'
            elif f.check_variable("moments"): # restart
                if len(f.read_variable("moments",cellids=1))==4:
                    var = 'restart_rho'
                else: # multipop restart
                    var = 'restart_rhom'
        varstr=var.replace("/","_")

    # File output checks
    if draw==None and axes==None:
        if outputfile==None: # Generate filename
            if outputdir==None: # default initial path
                outputdir=os.path.expandvars('$HOME/Plots/')
            # Sub-directories can still be defined in the "run" variable
            outputfile = outputdir+run+"_map_"+varstr+operatorstr+stepstr+".png"
        else: 
            if outputdir!=None:
                outputfile = outputdir+outputfile

        # Re-check to find actual target sub-directory
        outputprefixind = outputfile.rfind('/')
        if outputprefixind >= 0:            
            outputdir = outputfile[:outputprefixind+1]

        # Ensure output directory exists
        if not os.path.exists(outputdir):
            try:
                os.makedirs(outputdir)
            except:
                pass

        if not os.access(outputdir, os.W_OK):
            print("No write access for directory "+outputdir+"! Exiting.")
            return

        # Check if target file already exists and overwriting is disabled
        if (nooverwrite!=None and os.path.exists(outputfile)):            
            if os.stat(outputfile).st_size > 0: # Also check that file is not empty
                print("Found existing file "+outputfile+". Skipping.")
                return
            else:
                print("Found existing file "+outputfile+" of size zero. Re-rendering.")


    Re = 6.371e+6 # Earth radius in m
    #read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize
    cellids = f.read_variable("CellID")
    pt.plot.plot_helpers.CELLSIZE = cellsize
    
    # Check if ecliptic or polar run
    if ysize==1:
        simext=[xmin,xmax,zmin,zmax]
        sizes=[xsize,zsize]
        pt.plot.plot_helpers.PLANE = 'XZ'
    if zsize==1:
        simext=[xmin,xmax,ymin,ymax]
        sizes=[xsize,ysize]
        pt.plot.plot_helpers.PLANE = 'XY'

    # Select window to draw
    if len(boxm)==4:
        boxcoords=list(boxm)
    elif len(boxre)==4:
        boxcoords=[i*Re for i in boxre]
    else:
        boxcoords=list(simext)

    # If box extents were provided manually, truncate to simulation extents
    boxcoords[0] = max(boxcoords[0],simext[0])
    boxcoords[1] = min(boxcoords[1],simext[1])
    boxcoords[2] = max(boxcoords[2],simext[2])
    boxcoords[3] = min(boxcoords[3],simext[3])

    # Axes and units (default R_E)
    if axisunit!=None: # Use m or km or other
        if np.isclose(axisunit,0):
            axisunitstr = r'm'
        elif np.isclose(axisunit,3):
            axisunitstr = r'km'
        else:
            axisunitstr = r'$10^{'+str(int(axisunit))+'}$ m'
        axisunit = np.power(10,int(axisunit))
    else:
        axisunitstr = r'$\mathrm{R}_{\mathrm{E}}$'
        axisunit = Re
        
    # Scale data extent and plot box
    simext=[i/axisunit for i in simext]
    boxcoords=[i/axisunit for i in boxcoords]    
    plot_colormap.boxcoords = boxcoords # Make boxcoords available for formatter function

    ##########
    # Read data and calculate required variables
    ##########
    if expression==None:        
        # Read data from file
        if operator==None:
            operator="pass"
        datamap_info = f.read_variable_info(var, operator=operator)

        cb_title_use = datamap_info.latex
        # if cb_title_use == "": 
        #     cb_title_use = r""+var.replace("_","\_")
        datamap_unit = datamap_info.latexunits

        # If vscale is in use
        if not np.isclose(vscale,1.):
            datamap_unit=datamap_info.latexunits+r"${\times}$"+fmt(vscale,None)
        # Allow specialist units for known vscale and unit combinations
        if datamap_info.units=="s" and np.isclose(vscale,1.e6):
            datamap_unit = r"$\mu$s"
        if datamap_info.units=="s" and np.isclose(vscale,1.e3):
            datamap_unit = "ms"
        if datamap_info.units=="T" and np.isclose(vscale,1.e9):
            datamap_unit = "nT"
        if datamap_info.units=="K" and np.isclose(vscale,1.e-6):
            datamap_unit = "MK"
        if datamap_info.units=="Pa" and np.isclose(vscale,1.e9):
            datamap_unit = "nPa"
        if datamap_info.units=="1/m3" and np.isclose(vscale,1.e-6):
            datamap_unit = r"$\mathrm{cm}^{-3}$"
        if datamap_info.units=="m/s" and np.isclose(vscale,1.e-3):
            datamap_unit = r"$\mathrm{km}\,\mathrm{s}^{-1}$"
        if datamap_info.units=="V/m" and np.isclose(vscale,1.e3):
            datamap_unit = r"$\mathrm{mV}\,\mathrm{m}^{-1}$"            
        if datamap_info.units=="eV/cm3" and np.isclose(vscale,1.e-3):
            datamap_unit = r"$\mathrm{keV}\,\mathrm{cm}^{-3}$"            
        
        # Add unit to colorbar title
        if datamap_unit!="":
            cb_title_use = cb_title_use + " ["+datamap_unit+"]"

        datamap = datamap_info.data

        # Verify data shape
        if np.ndim(datamap)==0:
            print("Error, read only single value from vlsv file!",datamap.shape)
            return -1
        # fsgrid reader returns array in correct shape but needs to be transposed
        if var.startswith('fg_'):
            datamap = np.swapaxes(datamap, 0,1)
        else:            
            # For vlasov grid reader, reorder and reshape.
            if np.ndim(datamap)==1:
                datamap = datamap[cellids.argsort()].reshape([sizes[1],sizes[0]])
            elif np.ndim(datamap)==2: # vector variable
                datamap = datamap[cellids.argsort()].reshape([sizes[1],sizes[0],datamap.shape[1]])
            elif np.ndim(datamap)==3:  # tensor variable
                datamap = datamap[cellids.argsort()].reshape([sizes[1],sizes[0],datamap.shape[1],datamap.shape[2]])
            else:
                print("Error in reshaping datamap!") 
    else:
        # Expression set, use generated or provided colorbar title
        cb_title_use = expression.__name__.replace("_","\_") +'$'+operatorstr+'$' 

    # Allow title override
    if cbtitle!=None:
        # Here allow underscores for manual math mode
        cb_title_use = cbtitle       

    # Generates the mesh to map the data to.
    [XmeshXY,YmeshXY] = scipy.meshgrid(np.linspace(simext[0],simext[1],num=sizes[0]+1),np.linspace(simext[2],simext[3],num=sizes[1]+1))

    # The grid generated by meshgrid has all four corners for each cell.
    # We mask using only the centre values.
    # Calculate offsets for cell-centre coordinates
    XmeshCentres = XmeshXY[:-1,:-1] + 0.5*(XmeshXY[0,1]-XmeshXY[0,0])
    YmeshCentres = YmeshXY[:-1,:-1] + 0.5*(YmeshXY[1,0]-YmeshXY[0,0])    
    maskgrid = np.ma.array(XmeshCentres)    
    if pass_full is None:
        # If zoomed-in using a defined box, and not specifically asking to pass all values:        
        # Generate mask for only visible section (with small buffer for e.g. gradient calculations)
        maskboundarybuffer = 2.*cellsize/axisunit
        maskgrid = np.ma.masked_where(XmeshCentres<(boxcoords[0]-maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(XmeshCentres>(boxcoords[1]+maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(YmeshCentres<(boxcoords[2]-maskboundarybuffer), maskgrid)
        maskgrid = np.ma.masked_where(YmeshCentres>(boxcoords[3]+maskboundarybuffer), maskgrid)

    if np.ma.is_masked(maskgrid):
        # Save lists for masking
        MaskX = np.where(~np.all(maskgrid.mask, axis=1))[0] # [0] takes the first element of a tuple
        MaskY = np.where(~np.all(maskgrid.mask, axis=0))[0]
        XmeshPass = XmeshXY[MaskX[0]:MaskX[-1]+2,:]
        XmeshPass = XmeshPass[:,MaskY[0]:MaskY[-1]+2]
        YmeshPass = YmeshXY[MaskX[0]:MaskX[-1]+2,:]
        YmeshPass = YmeshPass[:,MaskY[0]:MaskY[-1]+2]
        XmeshCentres = XmeshCentres[MaskX[0]:MaskX[-1]+1,:]
        XmeshCentres = XmeshCentres[:,MaskY[0]:MaskY[-1]+1]
        YmeshCentres = YmeshCentres[MaskX[0]:MaskX[-1]+1,:]
        YmeshCentres = YmeshCentres[:,MaskY[0]:MaskY[-1]+1]
    else:
        XmeshPass = np.ma.array(XmeshXY)
        YmeshPass = np.ma.array(YmeshXY)

    # Attempt to call external and expression functions to see if they have required
    # variable information (If they accept the requestvars keyword, they should
    # return a list of variable names as strings)
    if pass_vars is None:        
        pass_vars=[] # Initialise list unless already provided
    if expression!=None: # Check the expression
        try:
            reqvariables = expression(None,True)
            for i in reqvariables:
                if not (i in pass_vars): pass_vars.append(i)
        except:
            pass
    if external!=None: # Check the external
        try:
            reqvariables = external(None,None,None,None,True)
            for i in reqvariables:
                if not (i in pass_vars): pass_vars.append(i)
        except:
            pass
    # If expression or external routine need variables, read them from the file.
    if pass_vars!=None:        
        if pass_times==None:
            # Note: pass_maps is now a dictionary
            pass_maps = {}
            # Gather the required variable maps for a single time step
            for mapval in pass_vars:
                # a check_variable(mapval) doesn't work as it doesn't know about
                # data reducers. Try/catch?
                if mapval.startswith('fg_'):
                    pass_map = f.read_fsgrid_variable(mapval)
                    pass_map = np.swapaxes(pass_map, 0,1)
                else:
                    pass_map = f.read_variable(mapval)
                if np.ndim(pass_map)==0:
                    print("Error, read only single value from vlsv file!",pass_map.shape)
                    return -1
                # fsgrid reader returns array in correct shape. 
                # For vlasov grid reader, reorder and reshape.
                if not mapval.startswith('fg_'):
                    if np.ndim(pass_map)==1:
                        pass_map = pass_map[cellids.argsort()].reshape([sizes[1],sizes[0]])
                    elif np.ndim(pass_map)==2: # vector variable
                        pass_map = pass_map[cellids.argsort()].reshape([sizes[1],sizes[0],pass_map.shape[1]])
                    elif np.ndim(pass_map)==3:  # tensor variable
                        pass_map = pass_map[cellids.argsort()].reshape([sizes[1],sizes[0],pass_map.shape[1],pass_map.shape[2]])
                    else:
                        print("Error in reshaping pass_map!") 
                if np.ma.is_masked(maskgrid):
                    if np.ndim(pass_map)==2:
                        pass_map = pass_map[MaskX[0]:MaskX[-1]+1,:]
                        pass_map = pass_map[:,MaskY[0]:MaskY[-1]+1]
                    elif np.ndim(pass_map)==3: # vector variable
                        pass_map = pass_map[MaskX[0]:MaskX[-1]+1,:,:]
                        pass_map = pass_map[:,MaskY[0]:MaskY[-1]+1,:]
                    elif np.ndim(pass_map)==4:  # tensor variable
                        pass_map = pass_map[MaskX[0]:MaskX[-1]+1,:,:,:]
                        pass_map = pass_map[:,MaskY[0]:MaskY[-1]+1,:,:]
                    else:
                        print("Error in masking pass_maps!") 
                pass_maps[mapval] = pass_map # add to the dictionary
        else:
            # Or gather over a number of time steps
            # Note: pass_maps is now a list of dictionaries
            pass_maps = []
            if step!=None and filename!=None:
                currstep = step
            else:
                if filename!=None: # parse from filename
                    currstep = int(filename[-12:-5])
                else:
                    print("Error, cannot determine current step for time extent extraction!")
                    return
            # define relative time step selection
            if np.ndim(pass_times)==0:
                dsteps = np.arange(-abs(int(pass_times)),abs(int(pass_times))+1)
            elif np.ndim(pass_times)==1 and len(pass_times)==2:
                dsteps = np.arange(-abs(int(pass_times[0])),abs(int(pass_times[1]))+1)
            else:
                print("Invalid value given to pass_times")
                return
            # Loop over requested times
            for ds in dsteps:
                # Construct using known filename.
                filenamestep = filename[:-12]+str(currstep+ds).rjust(7,'0')+'.vlsv'
                print(filenamestep)
                fstep=pt.vlsvfile.VlsvReader(filenamestep)
                step_cellids = fstep.read_variable("CellID")
                # Append new dictionary as new timestep
                pass_maps.append({})
                # Add relative step identifier to dictionary
                pass_maps[-1]['dstep'] = ds
                # Gather the required variable maps
                for mapval in pass_vars:
                    if mapval.startswith('fg_'):
                        pass_map = fstep.read_fsgrid_variable(mapval)
                        pass_map = np.swapaxes(pass_map, 0,1)
                    else:
                        pass_map = fstep.read_variable(mapval)
                    if np.ndim(pass_map)==0:
                        print("Error, read only single value from vlsv file!",pass_map.shape)
                        return -1
                    # fsgrid reader returns array in correct shape. 
                    # For vlasov grid reader, reorder and reshape.
                    if not mapval.startswith('fg_'):
                        if np.ndim(pass_map)==1:
                            pass_map = pass_map[step_cellids.argsort()].reshape([sizes[1],sizes[0]])
                        elif np.ndim(pass_map)==2: # vector variable
                            pass_map = pass_map[step_cellids.argsort()].reshape([sizes[1],sizes[0],pass_map.shape[1]])
                        elif np.ndim(pass_map)==3:  # tensor variable
                            pass_map = pass_map[step_cellids.argsort()].reshape([sizes[1],sizes[0],pass_map.shape[1],pass_map.shape[2]])
                        else:
                            print("Error in reshaping pass_map!") 
                    if np.ma.is_masked(maskgrid):
                        if np.ndim(pass_map)==2:
                            pass_map = pass_map[MaskX[0]:MaskX[-1]+1,:]
                            pass_map = pass_map[:,MaskY[0]:MaskY[-1]+1]
                        elif np.ndim(pass_map)==3: # vector variable
                            pass_map = pass_map[MaskX[0]:MaskX[-1]+1,:,:]
                            pass_map = pass_map[:,MaskY[0]:MaskY[-1]+1,:]
                        elif np.ndim(pass_map)==4:  # tensor variable
                            pass_map = pass_map[MaskX[0]:MaskX[-1]+1,:,:,:]
                            pass_map = pass_map[:,MaskY[0]:MaskY[-1]+1,:,:]
                        else:
                            print("Error in masking pass_maps!") 
                    pass_maps[-1][mapval] = pass_map # add to the dictionary

    # Optional user-defined expression used for color panel instead of a single pre-existing var
    if expression!=None:
        # Here pass_maps is already the cropped-via-mask data array
        datamap = expression(pass_maps)
        # Handle operators
        if ((operator is not None) and (operator!='pass') and (operator!='magnitude')):
            if operator=='x': operator = '0'
            if operator=='y': operator = '1'
            if operator=='z': operator = '2'
            if not operator.isdigit():
                print("Error parsing operator for custom expression!")
                return
            elif np.ndim(datamap)==3:
                datamap = datamap[:,:,int(operator)]
                
    # Now, if map is a vector or tensor, reduce it down
    if np.ndim(datamap)==3: # vector
        if datamap.shape[2]!=3:
            # This may also catch 3D simulation fsgrid variables
            print("Error, expected array of 3-element vectors, found array of shape ",datamap.shape)
            return -1
        # take magnitude of three-element vectors
        datamap = np.linalg.norm(datamap, axis=-1)
    if np.ndim(datamap)==4: # tensor
        if datamap.shape[2]!=3 or datamap.shape[3]!=3:
            # This may also catch 3D simulation fsgrid variables
            print("Error, expected array of 3x3 tensors, found array of shape ",datamap.shape)
            return -1
        # take trace
        datamap = datamap[:,:,0,0]+datamap[:,:,1,1]+datamap[:,:,2,2]
    if np.ndim(datamap)>=5: # Too many dimensions
        print("Error, too many dimensions in datamap, found array of shape ",datamap.shape)
        return -1
    if np.ndim(datamap)!=2:
        # Array dimensions not as expected
        print("Error reading variable "+var+"! Found array of shape ",datamap.shape,". Exiting.")
        return -1
        
    # Scale final generated datamap if requested
    datamap = datamap * vscale

    # Find rhom map for use in masking out ionosphere
    if f.check_variable("vg_rhom"):
        rhomap = f.read_variable("vg_rhom")
    elif f.check_variable("proton/vg_rho"):
        rhomap = f.read_variable("proton/vg_rho")
    elif f.check_variable("proton/rho"):
        rhomap = f.read_variable("proton/rho")
    elif f.check_variable("moments"):
        rhomap = f.read_variable("restart_rhom")
    else:
        rhomap = f.read_variable("rhom")
    rhomap = rhomap[cellids.argsort()].reshape([sizes[1],sizes[0]])
        
    # Crop both rhomap and datamap to view region
    if np.ma.is_masked(maskgrid):
        # Strip away columns and rows which are outside the plot region
        rhomap = rhomap[MaskX[0]:MaskX[-1]+1,:]
        rhomap = rhomap[:,MaskY[0]:MaskY[-1]+1]
        # Also for the datamap, unless it was already provided by an expression
        if expression==None:
            datamap = datamap[MaskX[0]:MaskX[-1]+1,:]
            datamap = datamap[:,MaskY[0]:MaskY[-1]+1]

    # Mask region outside ionosphere. Note that for some boundary layer cells, 
    # a density is calculated, but e.g. pressure is not, and these cells aren't
    # excluded by this method. Also mask away regions where datamap is invalid
    rhomap = np.ma.masked_less_equal(np.ma.masked_invalid(rhomap), 0)
    rhomap = np.ma.masked_where(~np.isfinite(datamap), rhomap)
    XYmask = rhomap.mask
    if np.ma.is_masked(XYmask):
        if XYmask.mask.all():
            # if everything was masked in rhomap, allow plotting
            XYmask[:,:] = False
        else:
            # Mask datamap
            datamap = np.ma.array(datamap, mask=XYmask)

    # If automatic range finding is required, find min and max of array
    # Performs range-finding on a masked array to work even if array contains invalid values
    if vmin!=None:
        vminuse=vmin
    else: 
        vminuse=np.ma.amin(datamap)
    if vmax!=None:
        vmaxuse=vmax
    else:
        vmaxuse=np.ma.amax(datamap)

    # If both values are zero, we have an empty array
    if vmaxuse==vminuse==0:
        print("Error, requested array is zero everywhere. Exiting.")
        return 0

    # If vminuse and vmaxuse are extracted from data, different signs, and close to each other, adjust to be symmetric
    # e.g. to plot transverse field components. Always done for symlog.
    if vmin==None and vmax==None:
        if np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2) or symlog!=None:
            absval = max(abs(vminuse),abs(vmaxuse))
            vminuse = -absval
            vmaxuse = absval

    # Ensure that lower bound is valid for logarithmic plots
    if (vminuse <= 0) and (lin is None) and (symlog is None):
        # Drop negative and zero values
        vminuse = np.ma.amin(np.ma.masked_less_equal(datamap,0))

    # Make vmaxuse and vminuse available for formatter functions
    plot_colormap.vminuse = vminuse
    plot_colormap.vmaxuse = vmaxuse
    plot_colormap.lin = lin

    # If symlog scaling is set:
    plot_colormap.linthresh = None
    if symlog!=None:
        if symlog>0:
            plot_colormap.linthresh = symlog 
        else:
            plot_colormap.linthresh = max(abs(vminuse),abs(vmaxuse))*1.e-2

    # Lin or log colour scaling, defaults to log
    if lin is None:
        # Special SymLogNorm case
        if symlog!=None:
            #norm = SymLogNorm(linthresh=plot_colormap.linthresh, linscale = 0.3, vmin=vminuse, vmax=vmaxuse, ncolors=cmapuse.N, clip=True)
            norm = SymLogNorm(linthresh=plot_colormap.linthresh, linscale = 0.3, vmin=vminuse, vmax=vmaxuse, clip=True)
            maxlog=int(np.ceil(np.log10(vmaxuse)))
            minlog=int(np.ceil(np.log10(-vminuse)))
            logthresh=int(np.floor(np.log10(plot_colormap.linthresh)))
            logstep=1
            ticks=([-(10**x) for x in range(logthresh, minlog+1, logstep)][::-1]
                    +[0.0]
                    +[(10**x) for x in range(logthresh, maxlog+1, logstep)] )
        else:
            # Logarithmic plot
            norm = LogNorm(vmin=vminuse,vmax=vmaxuse)
            ticks = LogLocator(base=10,subs=list(range(10))) # where to show labels
    else:
        # Linear
        linticks = 7
        if isinstance(lin, int):
            linticks = abs(lin)
            if linticks==1: # old default was to set lin=1 for seven linear ticks
                linticks = 7
                
        levels = MaxNLocator(nbins=255).tick_values(vminuse,vmaxuse)
        norm = BoundaryNorm(levels, ncolors=cmapuse.N, clip=True)
        ticks = np.linspace(vminuse,vmaxuse,num=linticks)

    # Select plotting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if axes==None: # If axes are provided, leave backend as-is.
        if draw!=None:
            if str(matplotlib.get_backend()) is not 'TkAgg':
                plt.switch_backend('TkAgg')
        else:
            if str(matplotlib.get_backend()) is not 'Agg':
                plt.switch_backend('Agg')  

    # Select image shape to match plotted area
    boxlenx = boxcoords[1]-boxcoords[0]
    boxleny = boxcoords[3]-boxcoords[2]
    # Round the values so that image sizes won't wobble when there's e.g. a moving box and numerical inaccuracies.
    # This is only done if the box size is suitable for the unit in use.
    if ((boxlenx > 10) and (boxleny > 10)):
        boxlenx = float( 0.05 * int(boxlenx*20*1.024) ) 
        boxleny = float( 0.05 * int(boxleny*20*1.024) ) 
    ratio = np.sqrt(boxleny/boxlenx)
    # default for square figure is figsize=[4.0,3.15] (with some accounting for axes etc)
    figsize = [4.0,3.15*ratio]
    # Special case for edge-to-edge figures
    if len(plot_title)==0 and (nocb!=None or internalcb!=None) and noborder!=None and noxlabels!=None and noylabels!=None:
        ratio = (boxcoords[3]-boxcoords[2])/(boxcoords[1]-boxcoords[0])
        figsize = [3.0,3.0*ratio]

    # If requested high res image
    if highres is not None:
        highresscale = 2
        if ((type(highres) is float) or (type(highres) is int)):
            highresscale = float(highres)
            if np.isclose(highresscale, 1.0):
                highresscale = 2
        figsize= [x * highresscale for x in figsize]
        fontsize=fontsize*highresscale
        fontsize2=fontsize2*highresscale
        fontsize3=fontsize3*highresscale
        scale=scale*highresscale
        thick=thick*highresscale
        fluxthick=fluxthick*highresscale
        streamlinethick=streamlinethick*highresscale
        vectorsize=vectorsize*highresscale

    if axes==None:
        # Create 300 dpi image of suitable size
        fig = plt.figure(figsize=figsize,dpi=300)
        ax1 = plt.gca() # get current axes
    else:
        ax1=axes
        fig = plt.gcf() # get current figure

    # Plot the actual mesh
    fig1 = ax1.pcolormesh(XmeshPass,YmeshPass,datamap, cmap=colormap,norm=norm)

    # Title and plot limits
    if len(plot_title)!=0:
        plot_title = r"\textbf{"+plot_title+"}"
        ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')

    ax1.set_xlim([boxcoords[0],boxcoords[1]])
    ax1.set_ylim([boxcoords[2],boxcoords[3]])
    ax1.set_aspect('equal')

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(thick)
    ax1.xaxis.set_tick_params(width=thick,length=3)
    ax1.yaxis.set_tick_params(width=thick,length=3)
    #ax1.xaxis.set_tick_params(which='minor',width=3,length=5)
    #ax1.yaxis.set_tick_params(which='minor',width=3,length=5)

    if noxlabels==None:
        ax1.set_xlabel(r'\textbf{X ['+axisunitstr+']}',fontsize=fontsize,weight='black')
        for item in ax1.get_xticklabels():
            item.set_fontsize(fontsize)
            item.set_fontweight('black')
        ax1.xaxis.offsetText.set_fontsize(fontsize)# set axis exponent offset font sizes
    if noylabels==None:
        if ysize==1: #Polar
            ax1.set_ylabel(r'\textbf{Z ['+axisunitstr+']}',fontsize=fontsize,weight='black')
        else: #Ecliptic
            ax1.set_ylabel(r'\textbf{Y ['+axisunitstr+']}',fontsize=fontsize,weight='black')
        for item in ax1.get_yticklabels():
            item.set_fontsize(fontsize)
            item.set_fontweight('black')
        ax1.yaxis.offsetText.set_fontsize(fontsize)# set axis exponent offset font sizes

    # Limit ticks, slightly according to ratio
    # ax1.xaxis.set_major_locator(plt.MaxNLocator(int(7/np.sqrt(ratio))))
    # ax1.yaxis.set_major_locator(plt.MaxNLocator(int(7*np.sqrt(ratio))))

    # add flux function contours
    if fluxfile != None:
        # Read binary flux function data from prepared files
        flux_function = np.fromfile(fluxfile,dtype='double').reshape(sizes[1],sizes[0])

        # Find inflow position values
        cid = f.get_cellid( [xmax-2*cellsize, 0,0] )
        ff_b = f.read_variable("B", cellids=cid)       
        if f.check_variable("moments"): # restart file
            ff_v = f.read_variable("restart_V", cellids=cid)            
        else:
            ff_v = f.read_variable("V", cellids=cid)            
        # Account for movement
        outofplane = [0,-1,0] # For polar runs
        if zsize==1: outofplane = [0,0,1] # For ecliptic runs
        flux_function = flux_function - timeval * np.inner(np.cross(ff_v,ff_b), outofplane)

        # Mask region (e.g. ionosphere)
        if np.ma.is_masked(maskgrid):
            flux_function = flux_function[MaskX[0]:MaskX[-1]+1,:]
            flux_function = flux_function[:,MaskY[0]:MaskY[-1]+1]
        if np.ma.is_masked(XYmask):
            flux_function = np.ma.array(flux_function, mask=XYmask)
        # The flux level contours must be fixed instead of scaled based on min/max values in order
        # to properly account for flux freeze-in and advection with plasma
        flux_levels = np.linspace(-10,10,fluxlines*60)        
        fluxcont = ax1.contour(XmeshCentres,YmeshCentres,flux_function,flux_levels,colors='k',linestyles='solid',linewidths=0.5*fluxthick,zorder=2)

    # add fSaved identifiers
    if fsaved != None:
        if type(fsaved) is str:
            fScolour = fsaved
        else:
            fScolour = 'black'
        fsavedvariable=None
        if f.check_variable("fSaved"):
            fsavedvariable="fSaved"
        if f.check_variable("vg_f_saved"):
            fsavedvariable="vg_f_saved"
        if fsavedvariable is not None:
            fSmap = f.read_variable(fsavedvariable)
            fSmap = fSmap[cellids.argsort()].reshape([sizes[1],sizes[0]])
            if np.ma.is_masked(maskgrid):
                fSmap = fSmap[MaskX[0]:MaskX[-1]+1,:]
                fSmap = fSmap[:,MaskY[0]:MaskY[-1]+1]
            if np.ma.is_masked(XYmask):
                fSmap = np.ma.array(fSmap, mask=XYmask)            
            fScont = ax1.contour(XmeshCentres,YmeshCentres,fSmap,[0.5],colors=fScolour, 
                                 linestyles='solid',linewidths=0.5,zorder=2)


    if Earth is not None:
        Earth = Circle((0, 0), 1.0, color='k')
        Earth2 = Wedge((0,0), 0.9, -90, 90, fc='white', ec=None,lw=0.0)
        ax1.add_artist(Earth)
        ax1.add_artist(Earth2)

    # add vectors on top
    if vectors != None:
        if vectors.startswith('fg_'):
            vectmap = f.read_fsgrid_variable(vectors)
            vectmap = np.swapaxes(vectmap, 0,1)
        else:
            vectmap = f.read_variable(vectors)
            vectmap = vectmap[cellids.argsort()].reshape([sizes[1],sizes[0],3])

        if np.ma.is_masked(maskgrid):
            vectmap = vectmap[MaskX[0]:MaskX[-1]+1,:,:]
            vectmap = vectmap[:,MaskY[0]:MaskY[-1]+1,:]
        if np.ma.is_masked(XYmask):
            vectmap = np.ma.array(vectmap)
            for i in range(3):
                vectmap[:,:,i].mask = XYmask

        # Find vector lengths and define color
        lengths = np.linalg.norm(vectmap, axis=-1)
        colors = np.ma.log10(np.ma.divide(lengths,np.ma.mean(lengths)))
        
        # Try to estimate vectstep so there's about 100 vectors in the image area
        visibleboxcells = (axisunit**2)*(boxcoords[1]-boxcoords[0])*(boxcoords[3]-boxcoords[2])/(cellsize**2)
        vectstep = int(np.sqrt(visibleboxcells/vectordensity))
        vectstep = max(1,vectstep)        
        
        # inplane unit length vectors
        if zsize==1:
            vectmap[:,:,2] = np.ma.zeros(vectmap[:,:,2].shape)
        elif ysize==1:
            vectmap[:,:,1] = np.ma.zeros(vectmap[:,:,1].shape)
        vectmap = np.ma.divide(vectmap, np.linalg.norm(vectmap, axis=-1)[:,:,np.newaxis])
        
        X = XmeshCentres[::vectstep,::vectstep]
        Y = YmeshCentres[::vectstep,::vectstep]
        U = vectmap[::vectstep,::vectstep,0]            
        if zsize==1:
            V = vectmap[::vectstep,::vectstep,1]
        elif ysize==1:
            V = vectmap[::vectstep,::vectstep,2]
        C = colors[::vectstep,::vectstep] 
        # quiver uses scale in the inverse fashion
        ax1.quiver(X,Y,U,V,C, cmap=vectorcolormap, units='dots', scale=0.05/vectorsize, headlength=4, headwidth=4,
                   headaxislength=2, scale_units='dots', pivot='middle')

    if streamlines!=None:
        if streamlines.startswith('fg_'):
            slinemap = f.read_fsgrid_variable(streamlines)
            slinemap = np.swapaxes(slinemap, 0,1)
        else:
            slinemap = f.read_variable(streamlines)
            slinemap = slinemap[cellids.argsort()].reshape([sizes[1],sizes[0],3])
        if np.ma.is_masked(maskgrid):
            slinemap = slinemap[MaskX[0]:MaskX[-1]+1,:,:]
            slinemap = slinemap[:,MaskY[0]:MaskY[-1]+1,:]
        if np.ma.is_masked(XYmask):
            slinemap = np.ma.array(slinemap)
            for i in range(3):
                slinemap[:,:,i].mask = XYmask

        U = slinemap[:,:,0]
        if zsize==1:
            V = slinemap[:,:,1]
        elif ysize==1:
            V = slinemap[:,:,2]
        ax1.streamplot(XmeshCentres,YmeshCentres,U,V,linewidth=0.5*streamlinethick, density=streamlinedensity, color=streamlinecolor, arrowsize=streamlinethick*0.5)

    # Optional external additional plotting routine overlayed on color plot
    # Uses the same pass_maps variable as expressions
    if external!=None:
        #extresult=external(ax1, XmeshXY,YmeshXY, pass_maps)
        if axes==None:
            extresult=external(ax1, XmeshCentres,YmeshCentres, pass_maps)
        else:
            extresult=external(axes, XmeshCentres,YmeshCentres, pass_maps)

    if nocb==None:
        if cbaxes is not None: 
            # Colorbar axes are provided
            cax = cbaxes
            cbdir="right"; horalign="left"
        elif internalcb is not None:
            # Colorbar within plot area
            cbloc=1; cbdir="left"; horalign="right"
            if type(internalcb) is str:
                if internalcb=="NE":
                    cbloc=1; cbdir="left"; horalign="right"
                if internalcb=="NW":
                    cbloc=2; cbdir="right"; horalign="left"
                if internalcb=="SW": 
                    cbloc=3; cbdir="right"; horalign="left"
                if internalcb=="SE": 
                    cbloc=4; cbdir="left";  horalign="right"
            # borderpad default value is 0.5, need to increase it to make room for colorbar title
            cax = inset_axes(ax1, width="5%", height="35%", loc=cbloc, borderpad=1.0,
                             bbox_transform=ax1.transAxes, bbox_to_anchor=(0.15,0,0.85,0.92))
        else:
            # Split existing axes to make room for colorbar
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbdir="right"; horalign="left"

        # Colourbar title
        if len(cb_title_use)!=0:
            #plt.text(1.0, 1.01, cb_title_use, fontsize=fontsize3,weight='black', transform=ax1.transAxes, horizontalalignment='center')
            cb_title_use = r"\textbf{"+cb_title_use+"}"        

        # First draw colorbar
        if usesci is True:
            cb = plt.colorbar(fig1,ticks=ticks,format=mtick.FuncFormatter(fmt),cax=cax, drawedges=False)
        else:
            #cb = plt.colorbar(fig1,ticks=ticks,cax=cax, drawedges=False, format=mtick.FormatStrFormatter('%4.2f'))
            cb = plt.colorbar(fig1,ticks=ticks,cax=cax, drawedges=False, format=mtick.FuncFormatter(cbfmt))
        cb.outline.set_linewidth(thick)
        cb.ax.yaxis.set_ticks_position(cbdir)

        if cbaxes is None:
            cb.ax.tick_params(labelsize=fontsize3)#,width=1.5,length=3)
            cb_title = cax.set_title(cb_title_use,fontsize=fontsize3,fontweight='bold', horizontalalignment=horalign)
            cb_title.set_position((0.,1.+0.025*scale)) # avoids having colourbar title too low when fontsize is increased
        else:
            cb.ax.tick_params(labelsize=fontsize)
            cb_title = cax.set_title(cb_title_use,fontsize=fontsize,fontweight='bold', horizontalalignment=horalign)

        # Perform intermediate draw if necessary to gain access to ticks
        if (symlog!=None and np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2)) or (lin==None and symlog==None):
            fig.canvas.draw() # draw to get tick positions

        # Adjust placement of innermost ticks for symlog if it indeed is (quasi)symmetric
        if symlog!=None and np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2):
            cbt=cb.ax.yaxis.get_ticklabels()
            (cbtx,cbty) = cbt[len(cbt)//2-1].get_position() # just below zero
            if abs(0.5-cbty)/scale < 0.1:
                cbt[len(cbt)//2-1].set_va("top")
            (cbtx,cbty) = cbt[len(cbt)//2+1].get_position() # just above zero
            if abs(0.5-cbty)/scale < 0.1:
                cbt[len(cbt)//2+1].set_va("bottom")
            if len(cbt)>=7: # If we have at least seven ticks, may want to adjust next ones as well
                (cbtx,cbty) = cbt[len(cbt)//2-2].get_position() # second below zero
                if abs(0.5-cbty)/scale < 0.15:
                    cbt[len(cbt)//2-2].set_va("top")
                (cbtx,cbty) = cbt[len(cbt)//2+2].get_position() # second above zero
                if abs(0.5-cbty)/scale < 0.15:
                    cbt[len(cbt)//2+2].set_va("bottom")

        # if too many subticks in logarithmic colorbar:
        if lin==None and symlog==None:
            nlabels = len(cb.ax.yaxis.get_ticklabels()) / ratio
            # Force less ticks for internal colorbars
            if internalcb!=None: nlabels = nlabels * 1.5
            valids = ['1','2','3','4','5','6','7','8','9']
            if nlabels > 10:
                valids = ['1','2','3','4','5','6','8']
            if nlabels > 19:
                valids = ['1','2','5']
            if nlabels > 28:
                valids = ['1']
            # for label in cb.ax.yaxis.get_ticklabels()[::labelincrement]:
            for label in cb.ax.yaxis.get_ticklabels():
                if usesci is True:
                    # labels will be in format $x.0\times10^{y}$
                    firstdigit = label.get_text().replace('$','')[0]
                else:
                    firstdigit = (label.get_text().replace('$','').replace('.','')).lstrip('0')[0]
                
                if not firstdigit in valids: label.set_visible(False)

    # Add Vlasiator watermark
    if (wmark is not None or wmarkb is not None) and axes is None:
        if wmark!=None:
            wm = plt.imread(get_sample_data(watermarkimage))
        else:
            wmark=wmarkb # for checking for placement
            wm = plt.imread(get_sample_data(watermarkimageblack))
        if type(wmark) is str:
            anchor = wmark
        else:
            anchor="NW"
        # Allowed region and anchor used in tandem for desired effect
        if anchor=="NW" or anchor=="W" or anchor=="SW":
            rect = [0.01, 0.01, 0.3, 0.98]
        elif anchor=="NE" or anchor=="E" or anchor=="SE":
            rect = [0.69, 0.01, 0.3, 0.98]
        elif anchor=="N" or anchor=="C" or anchor=="S":
            rect = [0.35, 0.01, 0.3, 0.98]
        newax = fig.add_axes(rect, anchor=anchor, zorder=1)
        newax.imshow(wm)
        newax.axis('off')

    # Find maximum possible lengths of axis tick labels
    # Only counts digits
    ticklens = [ len(re.sub(r'\D',"",axisfmt(bc,None))) for bc in boxcoords]
    tickmaxlens = [np.amax(ticklens[0:1]),np.amax(ticklens[2:3])]

    # Adjust axis tick labels
    for axisi, axis in enumerate([ax1.xaxis, ax1.yaxis]):
        if tickinterval!=None:
            axis.set_major_locator(mtick.MultipleLocator(tickinterval))
        # Custom tick formatter
        axis.set_major_formatter(mtick.FuncFormatter(axisfmt))
        ticklabs = axis.get_ticklabels()
        # Set boldface.
        for t in ticklabs:
            t.set_fontweight("black")
            # If label has >3 numbers, tilt it
            if tickmaxlens[axisi]>3: 
                t.set_rotation(30)
                t.set_verticalalignment('top')
                t.set_horizontalalignment('right')

    # Or turn x-axis labels off
    if noxlabels!=None:
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False) 
    # Or turn y-axis labels off
    if noylabels!=None:
        for label in ax1.yaxis.get_ticklabels():
            label.set_visible(False)


    # Adjust layout. Uses tight_layout() but in fact this ensures 
    # that long titles and tick labels are still within the plot area.
    if axes is not None:
        savefig_pad=0.01
        bbox_inches='tight'
    elif noborder==None:
        plt.tight_layout()
        savefig_pad=0.05 # The default is 0.1
        bbox_inches=None
    else:
        plt.tight_layout(pad=0.01)
        savefig_pad=0.01
        bbox_inches='tight'
        
    # Save output or draw on-screen
    if draw==None and axes==None:
        try:
            plt.savefig(outputfile,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
        except:
            print("Error with attempting to save figure due to matplotlib LaTeX integration.")
        print(outputfile+"\n")
    elif axes==None:
        # Draw on-screen
        plt.draw()
        plt.show()
