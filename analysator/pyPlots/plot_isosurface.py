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
import matplotlib
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage import measure
import scipy
import os, sys
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from packaging.version import Version
import ids3d
import re

def plot_isosurface(filename=None,
                    vlsvobj=None,
                    filedir=None, step=None,
                    outputdir=None, nooverwrite=None,
                    #
                    surf_var=None, surf_op=None, surf_level=None,
                    color_var=None, color_op=None,
                    surf_step=1,
                    #
                    title=None, cbtitle=None,
                    draw=None, usesci=None,
                    symmetric=False,
                    highres=None,
                    boxm=[],boxre=[],
                    colormap=None,
                    run=None,wmark=None, nocb=False,
                    unit=None, thick=1.0,scale=1.0,
                    vscale=1.0,
                    vmin=None, vmax=None, lin=None, symlog=None,
                    angle = [30.,90.], transparent=True
                    ):

    ''' Plots a coloured isosurface plot with axes and a colour bar.

    :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj:     Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename
    :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/ or override with PTOUTPUTDIR)
                        If directory does not exist, it will be created. If the string does not end in a
                        forward slash, the final parti will be used as a perfix for the files.
    :kword nooverwrite: Set to only perform actions if the target output file does not yet exist                    
     
    :kword surf_var:    Variable to read for defining surface
    :kword surf_op:     Operator to use for variable to read surface
    :kword surf_level:  Level at which to define surface
    :kword surf_step:   Vertex stepping for surface generation: larger value returns coarser surface

    :kword color_var:   Variable to read for coloring surface
    :kword color_op:    Operator to use for variable to color surface ('x', 'y', 'z'; if None and color_var is a vector, the magnitude is taken)
           
    :kword boxm:        zoom box extents [x0,x1,y0,y1] in metres (default and truncate to: whole simulation box)
    :kword boxre:       zoom box extents [x0,x1,y0,y1] in Earth radii (default and truncate to: whole simulation box)
    :kword colormap:    colour scale for plot, use e.g. hot_desaturated, jet, viridis, plasma, inferno,
                        magma, parula, nipy_spectral, RdBu, bwr
    :kword run:         run identifier, used for constructing output filename
    :kword title:       string to use as plot title instead of time
    :kword cbtitle:     string to use as colorbar title instead of map name
    :kword unit:        Plot axes using 10^{unit} m (default: Earth radius R_E)

    :kwird usesci:      Use scientific notation for colorbar ticks? (default: 1)
    :kword vscale:      Scale all values with this before plotting. Useful for going from e.g. m^-3 to cm^-3
                        or from tesla to nanotesla. Guesses correct units for colourbar for some known
                        variables. Set to None to seek for a default scaling.
    :kword vmin,vmax:   min and max values for colour scale and colour bar. If no values are given,
                        min and max values for whole plot (non-zero rho regions only) are used.
    :kword symmetric:   Set the absolute value of vmin and vmax to the greater of the two
    :kword lin:         Flag for using linear colour scaling instead of log
    :kword symlog:      Use logarithmic scaling, but linear when abs(value) is below the value given to symlog.
                        Allows symmetric quasi-logarithmic plots of e.g. transverse field components.
                        A given of 0 translates to a threshold of max(abs(vmin),abs(vmax)) * 1.e-2.
    :kword wmark:       If set to non-zero, will plot a Vlasiator watermark in the top left corner.
    :kword highres:     Creates the image in high resolution, scaled up by this value (suitable for logging.info). 
    :kword draw:        Set to nonzero in order to draw image on-screen instead of saving to file (requires x-windowing)
    :kword transparent: Set to False in order to make the surface opaque
    :kword scale:       Scale text size (default=1.0)
    :kword thick:       line and axis thickness, default=1.0
    :kword nocb:        Set to suppress drawing of colourbar
    :kword angle:       Viewing elevation and azimuthal angles in degrees

    :returns:           Outputs an image to a file or to the screen.

    .. code-block:: python

    # Example usage:

    '''
    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    # watermarkimage=os.path.expandvars('$HOME/appl_taito/analysator/pyPlot/logo_color.png')

    outputprefix = ''
    if outputdir==None:
        outputdir=pt.plot.defaultoutputdir
    outputprefixind = outputdir.rfind('/')
    if outputprefixind >= 0:
        outputprefix = outputdir[outputprefixind+1:]
        outputdir = outputdir[:outputprefixind+1]
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Input file or object
    if filename!=None:
        f=pt.vlsvfile.VlsvReader(filename)
    elif vlsvobj!=None:
        f=vlsvobj
    elif ((filedir!=None) and (step!=None)):
        filename = glob.glob(filedir+'bulk*'+str(step).rjust(7,'0')+'.vlsv')[0]
        #filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
        f=pt.vlsvfile.VlsvReader(filename)
    else:
        logging.info("Error, needs a .vlsv file name, python object, or directory and step")
        return
                
    # Scientific notation for colorbar ticks?
    if usesci==None:
        usesci=1
    
    if colormap==None:
        # Default values
        colormap="hot_desaturated"
        if color_op!=None:
            colormap="bwr"
    if Version(matplotlib.__version__) < Version("3.5.0"):
        cmapuse=matplotlib.cm.get_cmap(name=colormap)
    else:
        cmapuse=matplotlib.colormaps.get_cmap(colormap)

    fontsize=8*scale # Most text
    fontsize2=10*scale # Time title
    fontsize3=7*scale # Colour bar ticks

    # Plot title with time
    timeval=None
    timeval=f.read_parameter("time")
    if timeval==None:
        timeval=f.read_parameter("t")
    if timeval==None:
        logging.info("Unknown time format encountered")

    # Plot title with time
    if title==None:        
        if timeval == None:    
            logging.info("Unknown time format encountered")
            plot_title = ''
        else:
            #plot_title = "t="+str(int(timeval))+' s'
            plot_title = "t="+'{:4.2f}'.format(timeval)+' s'
    else:
        plot_title = title

    # step, used for file name
    if step is not None:
        stepstr = '_'+str(step).rjust(7,'0')
    else:
        if filename:
            stepstr = '_'+filename[-12:-5]
        else:
            stepstr = ''

    # If run name isn't given, just put "plot" in the output file name
    if run==None:
        run='plot'

    # Verify validity of operator
    surf_opstr=''
    color_opstr=''
    if color_op!=None:
        if color_op!='x' and color_op!='y' and color_op!='z':
            logging.info("Unknown operator "+color_op+", defaulting to None/magnitude for a vector.")
            color_op=None            
        else:
            # For components, always use linear scale, unless symlog is set
            color_opstr='_'+color_op
            if symlog==None:
                lin=7
    # Verify validity of operator
    if surf_op!=None:
        if surf_op!='x' and surf_op!='y' and surf_op!='z':
            logging.info("Unknown operator "+surf_op)
            surf_op=None            
        else:
            surf_opstr='_'+surf_op

    # Output file name
    surf_varstr=surf_var.replace("/","_")
    if color_var!=None:
        color_varstr=color_var.replace("/","_")
    else:
        color_varstr="solid"
    savefigname = outputdir+outputprefix+run+"_isosurface_"+surf_varstr+surf_opstr+"-"+color_varstr+color_opstr+stepstr+".png"

    # Check if target file already exists and overwriting is disabled
    if (nooverwrite!=None and os.path.exists(savefigname)):
        # Also check that file is not empty
        if os.stat(savefigname).st_size > 0:
            return
        else:
            logging.info("Found existing file "+savefigname+" of size zero. Re-rendering.")


    Re = 6.371e+6 # Earth radius in m
    # read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)    
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize
    cellids = f.read_variable("CellID")

    # Read the FSgrid mesh
    try:
        [xsizefg, ysizefg, zsizefg] = f.get_fsgrid_mesh_size()
        xsizefg = int(xsizefg)
        ysizefg = int(ysizefg)
        zsizefg = int(zsizefg)
        [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = f.get_fsgrid_mesh_extent()
        cellsizefg = (xmaxfg-xminfg)/xsizefg
        pt.plot.plot_helpers.CELLSIZE = cellsizefg
    except:
        if xsize!=1 and ysize!=1 and zsize!=1:
            logging.info("Did not find fsgrid data, but found 3D DCCRG mesh. Attempting to adapt.")
            [xsizefg, ysizefg, zsizefg] = [xsize * 2**f.get_max_refinement_level(), ysize * 2**f.get_max_refinement_level(), zsize * 2**f.get_max_refinement_level()]
            [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = [xmin, ymin, zmin, xmax, ymax, zmax]
            cellsizefg = cellsize
            pt.plot.plot_helpers.CELLSIZE = cellsize
        else:
            logging.info("Found 2D DCCRG mesh without FSgrid data. Exiting.")
            return -1

    # sort the cellid and the datamap list
    indexids = cellids.argsort()
    cellids = cellids[indexids]

    # find the highest refiment level
    reflevel = ids3d.refinement_level(xsize, ysize, zsize, cellids[-1])
    for i in range(5): # Check if Vlasov grid doesn't reach maximum (fsgrid) refinement
        if xsize*(2**(reflevel + i)) == xsizefg:
            reflevel += i
            break


    if (xsize==1) or (ysize==1) or (zsize==1):
        logging.info("Error: isosurface plotting requires 3D spatial domain!")
        return

    simext=[xmin,xmax,ymin,ymax,zmin,zmax]
    sizes=[xsize,ysize,zsize]

    # Select window to draw
    if len(boxm)==6:
        boxcoords=boxm
    elif len(boxre)==6:
        boxcoords=[i*Re for i in boxre]
    else:
        boxcoords=simext

    # If box extents were provided manually, truncate to simulation extents
    boxcoords[0] = max(boxcoords[0],simext[0])
    boxcoords[1] = min(boxcoords[1],simext[1])
    boxcoords[2] = max(boxcoords[2],simext[2])
    boxcoords[3] = min(boxcoords[3],simext[3])
    boxcoords[4] = max(boxcoords[4],simext[4])
    boxcoords[5] = min(boxcoords[5],simext[5])

    # Axes and units (default R_E)
    if unit!=None: # Use m or km or other
        if unit==0:
            unitstr = pt.plot.rmstring('m')
        if unit==3:
            unitstr = pt.plot.rmstring('km')
        else:
            unitstr = r'10^{'+str(int(unit))+'} '+pt.plot.rmstring('m')
        unit = np.power(10,int(unit))
    else:
        unitstr = pt.plot.rmstring('R')+'_'+pt.plot.rmstring('E')
        unit = Re
    unitstr = pt.plot.mathmode(unitstr)
        
    # Scale data extent and plot box
    simext_org = simext
    simext=[i/unit for i in simext]
    boxcoords=[i/unit for i in boxcoords]

    if color_var != None:
        if color_op==None:
            color_op="pass"
        datamap_info = f.read_variable_info(color_var, operator=color_op)

        cb_title_use = datamap_info.latex
        # Check if vscale results in standard unit
        vscale, _, datamap_unit_latex = datamap_info.get_scaled_units(vscale=vscale)

        # Add unit to colorbar title
        if datamap_unit_latex:
            cb_title_use = cb_title_use + r"\,["+datamap_unit_latex+"]"
    else: # color_var==None
        cb_title_use = ""
        nocb=1

    if cbtitle is not None:
        # Here allow underscores for manual math mode
        cb_title_use = cbtitle      


    if surf_op is None:
            surf_op="pass"
    surf_datamap_info = f.read_variable_info(surf_var, operator=surf_op)

    surf_datamap = surf_datamap_info.data

    # Verify data shape
    if np.ndim(surf_datamap)==0:
        logging.info("Error, read only single surface variable value from vlsv file! surf_datamap.shape being " + str(surf_datamap.shape))
        return -1
    

    #Define the box limits
    low = np.array([boxcoords[0], boxcoords[2], boxcoords[4]])*unit
    up = np.array([boxcoords[1], boxcoords[3], boxcoords[5]])*unit

    # Choose CellIDs inside the box
    ids, idx = ids3d.ids3d_box(cellids, low, up, reflevel, xsize, ysize, zsize, [xmin, ymin, zmin, xmax, ymax, zmax])

    surf_datamap = surf_datamap[indexids]
    surf_datamap = surf_datamap[idx]

    # Reshape surface data
    if np.ndim(surf_datamap)==1:
        surf_data_in_box = ids3d.idmesh3d2(ids, surf_datamap, reflevel, xsize, ysize, zsize,  None)
    elif np.ndim(surf_datamap)==2:
        surf_data_in_box = ids3d.idmesh3d2(ids, surf_datamap, reflevel, xsize, ysize, zsize, surf_datamap.shape[1])
    elif np.ndim(surf_datamap)==3:
        surf_data_in_box = ids3d.idmesh3d2(ids, surf_datamap, reflevel, xsize, ysize, zsize, (surf_datamap.shape[1],surf_datamap.shape[2]))
    else:
        logging.info("Dimension error in constructing 2D AMR slice!")
        return -1

    # Remove empty rows, columns and tubes
    empty_0 = np.where(~np.all(surf_data_in_box==0, axis=0))
    empty_1 = np.where(~np.all(surf_data_in_box==0, axis=1))

    MaskX = empty_1[0]
    MaskY = empty_0[0]
    MaskZ = empty_1[1]

    cropped_surf_data = surf_data_in_box[min(MaskX):max(MaskX)+1,:,:]
    cropped_surf_data = cropped_surf_data[:,min(MaskY):max(MaskY)+1,:]
    cropped_surf_data = cropped_surf_data[:,:,min(MaskZ):max(MaskZ)+1]


    # Keep removing the face of the rectangle with the largest number of invalid values until the surface has only proper values
    counter = 0
    while True:
        zero_counts = np.zeros(6)
        zero_counts[0] = np.count_nonzero(cropped_surf_data[0, :, :]  == 0) # Face 0 (front)
        zero_counts[1] = np.count_nonzero(cropped_surf_data[-1, :, :] == 0)  # Face 1 (back)
        zero_counts[2] = np.count_nonzero(cropped_surf_data[:, 0, :]  == 0) # Face 2 (left)
        zero_counts[3] = np.count_nonzero(cropped_surf_data[:, -1, :] == 0)  # Face 3 (right)
        zero_counts[4] = np.count_nonzero(cropped_surf_data[:, :, 0]  == 0) # Face 4 (top)
        zero_counts[5] = np.count_nonzero(cropped_surf_data[:, :, -1] == 0)  # Face 5 (bottom)

        face_to_be_removed = np.argmax(zero_counts)

        if face_to_be_removed==0:
            cropped_surf_data = cropped_surf_data[1:,:,:]
        elif face_to_be_removed==1:
            cropped_surf_data = cropped_surf_data[:-1,:,:]
        elif face_to_be_removed==2:
            cropped_surf_data = cropped_surf_data[:,1:,:]
        elif face_to_be_removed==3:
            cropped_surf_data = cropped_surf_data[:,:-1,:]
        elif face_to_be_removed==4:
            cropped_surf_data = cropped_surf_data[:,:,1:]
        elif face_to_be_removed==5:
            cropped_surf_data = cropped_surf_data[:,:,:-1]

        counter+=1
        if counter > 50:    # Failsafe
            logging.info("Error with boundaries! Exiting.")
            return -1
        
        if np.all(zero_counts==0):
            break 
    

    if surf_level==None:
        surf_level = 0.5*(np.amin(cropped_surf_data)+np.amax(cropped_surf_data))
    logging.info("Minimum found surface value "+str(np.amin(cropped_surf_data))+" surface level "+str(surf_level)+" max "+str(np.amax(cropped_surf_data)))

    # Select plotting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if draw is not None:
        if str(matplotlib.get_backend()) is not pt.backend_interactive: #'TkAgg': 
            plt.switch_backend(pt.backend_interactive)
    else:
        if str(matplotlib.get_backend()) is not pt.backend_noninteractive: #'Agg':
            plt.switch_backend(pt.backend_noninteractive)  


    spacing = ((xmaxfg-xminfg)/(xsizefg*unit), (ymaxfg-yminfg)/(ysizefg*unit), (zmaxfg-zminfg)/(zsizefg*unit))

    verts, faces, normals, arrays = measure.marching_cubes(cropped_surf_data, level=surf_level,
                                                                   spacing=spacing,
                                                                   gradient_direction='descent',
                                                                   step_size=surf_step,
                                                                   allow_degenerate=True, method='lewiner', mask=cropped_surf_data!=0)
    


    #offset with respect to simulation domain corner
    verts[:,0] = verts[:,0] + low[0]/unit
    verts[:,1] = verts[:,1] + low[1]/unit
    verts[:,2] = verts[:,2] + low[2]/unit

    # A box that contains the isosurface
    contour_box_low = np.array([min(verts[:,0])-1,min(verts[:,1])-1,min(verts[:,2])-1])*unit
    contour_box_up = np.array([max(verts[:,0])+1,max(verts[:,1])+1,max(verts[:,2])+1])*unit



    # Next find color variable values at vertices
    if color_var != None:
        nverts = len(verts[:,0])
        logging.info("Extracting color values for "+str(nverts)+" vertices and "+str(len(faces[:,0]))+" faces.")
        all_coords = np.empty((nverts, 3))
        for i in np.arange(nverts):            
            # # due to mesh generation, some coordinates may be outside simulation domain
            # WARNING this means it might be doing wrong things in the periodic dimension of 2.9D runs.
            coords = verts[i,:]*unit 
            coords[0] = max(coords[0],simext_org[0]+0.1*cellsize)
            coords[0] = min(coords[0],simext_org[1]-cellsize)
            coords[1] = max(coords[1],simext_org[2]+0.1*cellsize)
            coords[1] = min(coords[1],simext_org[3]-cellsize)
            coords[2] = max(coords[2],simext_org[4]+0.1*cellsize)
            coords[2] = min(coords[2],simext_org[5]-cellsize)
            all_coords[i] = coords


        # Choose CellIDs inside the isosurface box
        color_ids, color_idx = ids3d.ids3d_box(cellids, contour_box_low, contour_box_up, reflevel, xsize, ysize, zsize, [xmin, ymin, zmin, xmax, ymax, zmax])

        # Read the variables to be plotted inside the box
        if color_op=="pass":
            vg_colors = f.read_variable(color_var, color_ids)
            # If value was vector value, take magnitude
            if np.ndim(vg_colors) != 1:
                vg_colors = np.linalg.norm(np.asarray(vg_colors),axis=-1)
        else:
            vg_colors = f.read_variable(color_var, color_ids, operator=color_op)            
        if np.ndim(vg_colors)!=1:
            raise ValueError("Error reading color variable "+color_var+"! Exiting.")


        vg_coords = f.read_variable("vg_coordinates", color_ids)

        # Interpolate the values to the surface
        interpolation = scipy.interpolate.RBFInterpolator(vg_coords, vg_colors, neighbors = 27)

        color_data = interpolation(all_coords)*vscale

        # Make sure color data is 1-dimensional (e.g. magnitude of E instead of 3 components)
        if np.ndim(color_data)!=1:
            color_data=np.linalg.norm(color_data, axis=-1)

    if color_var==None:
        # dummy norm
        logging.info("No surface color given, using dummy setup")
        norm = BoundaryNorm([0,1], ncolors=cmapuse.N, clip=True)
        vminuse=0
        vmaxuse=1
    else:
        # If automatic range finding is required, find min and max of array
        # Performs range-finding on a masked array to work even if array contains invalid values
        color_data = np.ma.masked_invalid(color_data)
        if vmin!=None:
            vminuse=vmin
        else: 
            vminuse=np.ma.amin(color_data)
        if vmax!=None:
            vmaxuse=vmax
        else:
            vmaxuse=np.ma.amax(color_data)       

        # If both values are zero, we have an empty array
        if vmaxuse==vminuse==0:
            logging.info("Error, requested array is zero everywhere. Exiting.")
            return 0            

        # If vminuse and vmaxuse are extracted from data, different signs, and close to each other, adjust to be symmetric
        # e.g. to plot transverse field components. Always done for symlog.
        if vmin is None and vmax is None:
            if symmetric or np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2) or symlog is not None:
                absval = max(abs(vminuse),abs(vmaxuse))
                vminuse = -absval
                vmaxuse = absval

        # Check that lower bound is valid for logarithmic plots
        if (vminuse <= 0) and (lin==None) and (symlog==None):
            # Drop negative and zero values
            vminuse = np.ma.amin(np.ma.masked_less_equal(color_data,0))

        # Special case of very small vminuse values
        if ((vmin is None) or (vmax is None)) and (vminuse > 0) and (vminuse < vmaxuse*1.e-5):
            vminuse = vmaxuse*1e-5
            if lin is not None:
                vminuse = 0

        # If symlog scaling is set:
        if symlog!=None:
            if symlog>0:
                linthresh = symlog 
            else:
                linthresh = max(abs(vminuse),abs(vmaxuse))*1.e-2

        # Lin or log colour scaling, defaults to log
        if lin is None:
            # Special SymLogNorm case
            if symlog is not None:
                if Version(matplotlib.__version__) < Version("3.2.0"):
                    norm = SymLogNorm(linthresh=linthresh, linscale = 1.0, vmin=vminuse, vmax=vmaxuse, clip=True)
                    logging.info("WARNING: colormap SymLogNorm uses base-e but ticks are calculated with base-10.")
                    #TODO: copy over matplotlib 3.3.0 implementation of SymLogNorm into analysator
                else:
                    norm = SymLogNorm(base=10, linthresh=linthresh, linscale = 1.0, vmin=vminuse, vmax=vmaxuse, clip=True)
                maxlog=int(np.ceil(np.log10(vmaxuse)))
                minlog=int(np.ceil(np.log10(-vminuse)))
                logthresh=int(np.floor(np.log10(linthresh)))
                logstep=1
                ticks=([-(10**x) for x in range(logthresh, minlog+1, logstep)][::-1]
                        +[0.0]
                        +[(10**x) for x in range(logthresh, maxlog+1, logstep)] )
            else:
                # Logarithmic plot
                norm = LogNorm(vmin=vminuse,vmax=vmaxuse)
                ticks = LogLocator(base=10,subs=range(10)) # where to show labels
        else:
            # Linear
            levels = MaxNLocator(nbins=255).tick_values(vminuse,vmaxuse)
            norm = BoundaryNorm(levels, ncolors=cmapuse.N, clip=True)
            ticks = np.linspace(vminuse,vmaxuse,num=lin)            

        logging.info("Selected color range: "+str(vminuse)+" to "+str(vmaxuse))

    # Create 300 dpi image of suitable size
    fig = plt.figure(figsize=[4.5,4.5],dpi=300)
    #ax1 = fig.gca(projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')

    # If requested high res image
    if highres:
        highresscale = 2
        if ((type(highres) is float) or (type(highres) is int)):
            highresscale = float(highres)
            if np.isclose(highresscale, 1.0):
                highresscale = 2
        # figsize= [x * highresscale for x in figsize] # figsize not defined, neither used in this function
        fontsize=fontsize*highresscale
        fontsize2=fontsize2*highresscale
        fontsize3=fontsize3*highresscale
        scale=scale*highresscale
        thick=thick*highresscale
        # streamlinethick=streamlinethick*highresscale # streamlinethick not defined, neither used in this function
        # vectorsize=vectorsize*highresscale # vectorsize not defined, neither used in this function

    # Generate virtual bounding box to get equal aspect
    maxrange = np.array([boxcoords[1]-boxcoords[0], boxcoords[3]-boxcoords[2], boxcoords[5]-boxcoords[4]]).max() / 2.0
    midvals = np.array([boxcoords[1]+boxcoords[0], boxcoords[3]+boxcoords[2], boxcoords[5]+boxcoords[4]]) / 2.0


    # Three options:
    # 2.9D ecliptic, 2.9D polar, or 3D
    if ysize < 0.2*xsize: # 2.9D polar, perform rotation
        generatedsurface = ax1.plot_trisurf(verts[:,2], verts[:,0], verts[:,1], triangles=faces,
                                            cmap=cmapuse, norm=norm, vmin=vminuse, vmax=vmaxuse, 
                                            lw=0, shade=False, edgecolors=None, antialiased=transparent)
        ax1.set_xlabel("z ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_ylabel("x ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_zlabel("y ["+unitstr+"]", fontsize=fontsize3)


        # Set camera angle
        ax1.view_init(elev=angle[0], azim=angle[1])
        # Set virtual bounding box
        ax1.set_xlim([midvals[2]-maxrange, midvals[2]+maxrange])
        ax1.set_ylim([midvals[0]-maxrange, midvals[0]+maxrange])
        ax1.set_zlim([midvals[1]-maxrange, midvals[1]+maxrange])
        ax1.tick_params(labelsize=fontsize3,width=thick,length=3*thick)
        
    else: # 3D or 2.9D ecliptic, leave as is
        generatedsurface = ax1.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces,
                                            cmap=cmapuse, norm=norm, vmin=vminuse, vmax=vmaxuse, 
                                            lw=0.2, shade=False, edgecolors=None, antialiased=transparent)
        ax1.set_xlabel("x ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_ylabel("y ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_zlabel("z ["+unitstr+"]", fontsize=fontsize3)
        # Set camera angle
        ax1.view_init(elev=angle[0], azim=angle[1])
        # Set virtual bounding box
        ax1.set_xlim([midvals[0]-maxrange, midvals[0]+maxrange])
        ax1.set_ylim([midvals[1]-maxrange, midvals[1]+maxrange])
        ax1.set_zlim([midvals[2]-maxrange, midvals[2]+maxrange])
        ax1.tick_params(labelsize=fontsize3,width=thick,length=3*thick)


    # Setting per-triangle colours for plot_trisurf needs to be done
    # as a separate set_array call.
    if color_var != None:
        # Find face-averaged colors
        # (simply setting the array to color_data failed for some reason)
        colors = np.mean(color_data[faces], axis=1)
        generatedsurface.set_array(colors)
        
    # Title and plot limits
    if len(plot_title)!=0:
        ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')


    # ax1.set_aspect('equal') #<<<--- this does not work for 3D plots!

    # for axis in ['top','bottom','left','right']:
    #     ax1.spines[axis].set_linewidth(thick)
    # ax1.xaxis.set_tick_params(width=thick,length=3)
    # ax1.yaxis.set_tick_params(width=thick,length=3)
    # #ax1.xaxis.set_tick_params(which='minor',width=3,length=5)
    # #ax1.yaxis.set_tick_params(which='minor',width=3,length=5)    


    if not nocb:
        # First draw colorbar
        if usesci==0:        
            cb = fig.colorbar(generatedsurface,ticks=ticks, format=mtick.FuncFormatter(pt.plot.cbfmt), drawedges=False, fraction=0.023, pad=0.02)
        else:
            cb = fig.colorbar(generatedsurface,ticks=ticks,format=mtick.FuncFormatter(pt.plot.cbfmtsci),drawedges=False, fraction=0.046, pad=0.04)

        if len(cb_title_use)!=0:
            cb_title_use = pt.plot.mathmode(pt.plot.bfstring(cb_title_use))

        if lin is None:
            pt.plot.cb_linear = False
        else:
            pt.plot.cb_linear = True

        # Ensure minor tick marks are off
        if lin is not None:
            cb.minorticks_off()

        cb.ax.tick_params(labelsize=fontsize3,width=thick,length=3*thick)
        cb.outline.set_linewidth(thick)
        cb.ax.set_title(cb_title_use)
        cb.ax.title.set_horizontalalignment('center')

        # Adjust precision for colorbar ticks
        thesetickvalues = cb.locator()
        if len(thesetickvalues)<2:
            precision_b=1
        else:
            mintickinterval = abs(thesetickvalues[-1]-thesetickvalues[0])
            # find smallest interval
            for ticki in range(len(thesetickvalues)-1):
                mintickinterval = min(mintickinterval,abs(thesetickvalues[ticki+1]-thesetickvalues[ticki]))
            precision_a, precision_b = '{:.1e}'.format(mintickinterval).split('e')
            # e.g. 9.0e-1 means we need precision 1
            # e.g. 1.33e-1 means we need precision 3?
        pt.plot.decimalprecision_cblin = 1
        if int(precision_b)<1: pt.plot.decimalprecision_cblin = str(1+abs(-int(precision_b)))
        cb.update_ticks()

        if symlog is not None and np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2):
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
        if lin is None and symlog is None:
            nlabels = len(cb.ax.yaxis.get_ticklabels()) #/ ratio
            valids = ['1','2','3','4','5','6','7','8','9']
            if nlabels > 10:
                valids = ['1','2','3','4','5','6','8']
            if nlabels > 19:
                valids = ['1','2','5']
            if nlabels > 28:
                valids = ['1']
            # for label in cb.ax.yaxis.get_ticklabels()[::labelincrement]:
            for labi,label in enumerate(cb.ax.yaxis.get_ticklabels()):
                labeltext = label.get_text().replace('$','').replace('{','').replace('}','').replace(r'\mbox{\textbf{--}}','').replace('-','').replace('.','').lstrip('0')
                if not labeltext:
                    continue
                firstdigit = labeltext[0]
                if not firstdigit in valids: 
                    label.set_visible(False)



    # Add Vlasiator watermark
    if wmark!=None:        
        wm = plt.imread(get_sample_data(watermarkimage))
        newax = fig.add_axes([0.01, 0.90, 0.3, 0.08], anchor='NW', zorder=-1)
        newax.imshow(wm)
        newax.axis('off')

    plt.tight_layout()
    savefig_pad=0.1 # The default is 0.1
    bbox_inches=None


    # Save output or draw on-screen
    if draw==None:
        # Note: generated title can cause strange PNG header problems
        # in rare cases. This problem is under investigation, but is related to the exact generated
        # title string. This try-catch attempts to simplify the time string until output succedes.
        try:
            plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
            savechange=0
        except:
            savechange=1
            plot_title = "t="+'{:4.1f}'.format(timeval)+' s '
            ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')                
            try:
                plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
            except:
                plot_title = "t="+str(int(timeval))+' s   '
                ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')                
                try:
                    plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
                except:
                    plot_title = ""
                    ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')                
                    try:
                        plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
                    except:
                        logging.info("Error:" + str(sys.exc_info()))
                        logging.info("Error with attempting to save figure, sometimes due to matplotlib LaTeX integration.")
                        logging.info("Usually removing the title should work, but this time even that failed.")                        
                        savechange = -1
        if savechange>0:
            logging.info("Due to rendering error, replaced image title with "+plot_title)
        if savechange>=0:
            logging.info(savefigname+"\n")
    else:
        plt.draw()
        plt.show()





def plot_neutral_sheet(filename=None,
                  vlsvobj=None,
                  filedir=None, step=None, run=None,
                  outputdir=None, outputfile=None,
                  nooverwrite=False,
                  var=None, op=None, operator=None,
                  title=None, cbtitle=None, draw=None, usesci=True,
                  symlog=None,
                  boxm=None,boxre=None,colormap=None,
                  nocb=False, internalcb=False,
                  wmark=False,wmarkb=False,
                  axisunit=None, thick=1.0,scale=1.0,
                  tickinterval=0, # Fairly certain this is a valid null value
                  noborder=False, noxlabels=False, noylabels=False,
                  vmin=None, vmax=None, lin=None,
                  external=None, expression=None,
                  diff=None,
                  vscale=1.0,
                  absolute=False,
                  symmetric=False,
                  pass_vars=None, pass_times=None, pass_full=False,
                  nomask=None,
                  Earth=None,
                  highres=None,
                  vectors=None, vectordensity=100, vectorcolormap='gray', vectorsize=1.0,
                  streamlines=None, streamlinedensity=1, streamlinecolor='white', streamlinethick=1.0,
                  axes=None, cbaxes=None,
                  useimshow=False, imshowinterp='none', folding_alpha=0.2, z_extent=[-5,5], sheetlayer='above'
                  ):
    
    ''' Plots a coloured plot along the neutral sheet with axes and a colour bar.

        :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
        :kword vlsvobj:     Optionally provide a python vlsvfile object instead
        :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
        :kword step:        output step index, used for constructing output (and possibly input) filename
        :kword run:         run identifier, used for constructing output filename
        :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/ or override with PTOUTPUTDIR)
                            If directory does not exist, it will be created. If the string does not end in a
                            forward slash, the final part will be used as a prefix for the files.
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
        :kword title:       string to use as plot title instead of time.
                            Special case: Set to "msec" to plot time with millisecond accuracy or "musec"
                            for microsecond accuracy. "sec" is integer second accuracy.
        :kword cbtitle:     string to use as colorbar title instead of map name
        :kword axisunit:    Plot axes using 10^{axisunit} m (default: Earth radius R_E)
        :kword tickinterval: Interval at which to have ticks on axes (not colorbar)

        :kwird usesci:      Use scientific notation for colorbar ticks? (default: True)
        :kword vmin,vmax:   min and max values for colour scale and colour bar. If no values are given,
                            min and max values for whole plot (non-zero rho regions only) are used.
        :kword symmetric:   Set the absolute value of vmin and vmax to the greater of the two
        :kword lin:         Flag for using linear colour scaling instead of log
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


        :kword draw:        Set to nonzero in order to draw image on-screen instead of saving to file (requires x-windowing)

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
                            variables. Set to None to seek for a default scaling.
        :kword absolute:    Plot the absolute of the evaluated variable

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

        :kword diff:        Instead of a regular plot, plot the difference between the selected plot type for
                            the regular source file and the file given by this keyword. This overides external
                            and expression keywords, as well as related pass_vars, pass_times, and pass_full.

        :kword z_extent:    Search bracket for the neutral sheet in axisunit units
        :kword folding_alpha: If non-zero, plots transparent dots over the regions where the sheet has multiple separate z-values. A value of 1.0 is opaque, a value of 0.0 is transparent.
        :kword sheetlayer:  If set to 'above', plots the topmost layer of the neutral sheet in case of folding. If set to anything else, 
                            the downmost layer is plotted.
        :kword nomask:      Do not mask plotting based on proton density

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
        :kword normal:      Direction of the normal of the 2D cut through ('x', 'y', or 'z' or a vector)
        :kword cutpoint:    Coordinate (in normal direction) through which the cut must pass [m]
        :kword cutpointre:  Coordinate (in normal direction) through which the cut must pass [rE]
        :kword useimshow:   Use imshow for raster background instead (default: False)
        :kword imshowinterp: Use this matplotlib interpolation for imshow (default: 'none')


        :returns:           Outputs an image to a file or to the screen.

        .. code-block:: python

            # Example usage:
            plot_colormap(filename=fileLocation, var="MA", run="BCQ",
                        colormap='nipy_spectral',step=j, outputdir=outputLocation,
                        lin=1, wmark=1, vmin=2.7, vmax=10, 
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
            plot_colormap(filename=fileLocation, vmin=1 vmax=40, expression=exprMA_cust,lin=1)

    '''

    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    watermarkimageblack=os.path.join(os.path.dirname(__file__), 'logo_black.png')

    # Switch None-keywords to empty lists (this way subsequent calls get correct empty default values
    if boxm is None:
        boxm=[],
    if boxre is None:
        boxre=[]
    if pass_vars is None:
        pass_vars=[]

    # Change certain falsy values:
    if not lin and lin != 0:
        lin = None
    if lin is True:
        lin = 7
    if not symlog and symlog != 0:
        symlog = None
    if symlog is True:
        symlog = 0
    if (filedir == ''):
        filedir = './'
    if (outputdir == ''):
        outputdir = './'

    # Input file or object
    if filename:
        f=pt.vlsvfile.VlsvReader(filename)
    elif (filedir and step is not None):
        filename = glob.glob(filedir+'bulk*'+str(step).rjust(7,'0')+'.vlsv')[0]
        #filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
        f=pt.vlsvfile.VlsvReader(filename)
    elif vlsvobj:
        f=vlsvobj
    else:
        logging.info("Error, needs a .vlsv file name, python object, or directory and step")
        return
    
    if operator is None:
        if op is not None:
            operator=op

    if not colormap:
        # Default values
        colormap="hot_desaturated"
        if operator is not None and operator in 'xyz':
            colormap="bwr"
    if Version(matplotlib.__version__) < Version("3.5.0"):
        cmapuse=matplotlib.cm.get_cmap(name=colormap)
    else:
        cmapuse=matplotlib.colormaps.get_cmap(colormap)
    
    fontsize=8*scale # Most text
    fontsize2=10*scale # Time title
    fontsize3=8*scale # Colour bar ticks and title
    # Small internal colorbar needs increased font size
    if internalcb: 
        fontsize3=fontsize3*2

    # Plot title with time
    timeval=f.read_parameter("time")

    # Plot title with time
    if title is None or title=="msec" or title=="musec":        
        if timeval is None:    
            plot_title = ''
        else:
            timeformat='{:4.1f}'
            if title=="sec": timeformat='{:4.0f}'
            if title=="msec": timeformat='{:4.3f}'
            if title=="musec": timeformat='{:4.6f}'
            plot_title = "t="+timeformat.format(timeval)+r'\,s'
    else:
        plot_title = title

    # step, used for file name
    if step is not None:
        stepstr = '_'+str(step).rjust(7,'0')
    else:
        if filename:
            stepstr = '_'+filename[-12:-5]
        else:
            stepstr = ''

    # If run name isn't given, just put "plot" in the output file name
    if run is None:
        run='plot'

    # Verify validity of operator
    operatorstr=''
    operatorfilestr=''
    if operator is not None:
        # .isdigit checks if the operator is an integer (for taking an element from a vector)
        if type(operator) is int:
            operator = str(operator)
        if not operator in 'xyz' and operator != 'magnitude' and not operator.isdigit():
            logging.info("Unknown operator "+str(operator))
            operator=None
            operatorstr=''
        if operator in 'xyz':
            # For components, always use linear scale, unless symlog is set
            operatorstr='_'+operator
            operatorfilestr='_'+operator
            if symlog is None:
                lin=7
        # index a vector
        if operator.isdigit():
            operator = str(operator)
            operatorstr='_{'+operator+'}'
            operatorfilestr='_'+operator
        # Note: operator magnitude gets operatorstr=''

    # Output file name
    if expression:
        varstr=expression.__name__.replace("/","_")
    else:        
        if not var:
            # If no expression or variable given, defaults to rhom
            var='vg_rhom'
            if f.check_variable("proton/vg_rho"): # multipop v5
                var = 'proton/vg_rho'
            elif f.check_variable("moments"): # restart
                var = 'vg_restart_rhom'
            elif f.check_variable("rho"): # old pre-v5 data (no fsgrid or AMR)
                var = 'rho'
        varstr=var.replace("/","_")

    # Activate diff mode?
    if diff:
        if (expression or external or pass_vars or pass_times or pass_full):
            logging.info("attempted to perform diff with one of the following active:")
            logging.info("expression or external or pass_vars or pass_times or pass_full. Exiting.")
            return -1
        expression=pt.plot.plot_helpers.expr_Diff
        pass_vars.append(var)
        varstr="DIFF_"+var.replace("/","_")
        pass_times=[1,0]

        # File output checks
    if not draw and not axes:
        if not outputfile: # Generate filename
            if not outputdir: # default initial path
                outputdir=pt.plot.defaultoutputdir
            # Sub-directories can still be defined in the "run" variable
            outputfile = outputdir+run+"_sheet_"+varstr+operatorfilestr+stepstr+".png"
        else: 
            if outputdir:
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
            logging.info(("No write access for directory "+outputdir+"! Exiting."))
            return

        # Check if target file already exists and overwriting is disabled
        if (nooverwrite and os.path.exists(outputfile)):            
            if os.stat(outputfile).st_size > 0: # Also check that file is not empty
                logging.info(("Found existing file "+outputfile+". Skipping."))
                return
            else:
                logging.info(("Found existing file "+outputfile+" of size zero. Re-rendering."))


    Re = 6.371e+6 # Earth radius in m
    # read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)    
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize
    cellids = f.read_variable("CellID")

    # Read the FSgrid mesh
    try:
        [xsizefg, ysizefg, zsizefg] = f.get_fsgrid_mesh_size()
        xsizefg = int(xsizefg)
        ysizefg = int(ysizefg)
        zsizefg = int(zsizefg)
        [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = f.get_fsgrid_mesh_extent()
        cellsizefg = (xmaxfg-xminfg)/xsizefg
        pt.plot.plot_helpers.CELLSIZE = cellsizefg
    except:
        if xsize!=1 and ysize!=1 and zsize!=1:
            logging.info("Did not find fsgrid data, but found 3D DCCRG mesh. Attempting to adapt.")
            [xsizefg, ysizefg, zsizefg] = [xsize * 2**f.get_max_refinement_level(), ysize * 2**f.get_max_refinement_level(), zsize * 2**f.get_max_refinement_level()]
            [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = [xmin, ymin, zmin, xmax, ymax, zmax]
            cellsizefg = cellsize
            pt.plot.plot_helpers.CELLSIZE = cellsize
        else:
            logging.info("Found 2D DCCRG mesh without FSgrid data. Exiting.")
            return -1

    # sort the cellid and the datamap list
    indexids = cellids.argsort()
    cellids = cellids[indexids]

    # find the highest refiment level
    reflevel = ids3d.refinement_level(xsize, ysize, zsize, cellids[-1])
    for i in range(5): # Check if Vlasov grid doesn't reach maximum (fsgrid) refinement
        if xsize*(2**(reflevel + i)) == xsizefg:
            reflevel += i
            break

    # Verify that FSgrid and spatial grid agree
    if ((xmin!=xminfg) or (xmax!=xmaxfg) or
        (ymin!=yminfg) or (ymax!=ymaxfg) or
        (zmin!=zminfg) or (zmax!=zmaxfg) or
        (xsize*(2**reflevel) !=xsizefg) or (ysize*(2**reflevel) !=ysizefg) or (zsize*(2**reflevel) !=zsizefg)):
        logging.info("FSgrid and vlasov grid disagreement!")
        return -1
    
    # Plotting grid in the XY plane
    axislabels = ['X','Y']
    simext=[xmin,xmax,ymin,ymax,zmin,zmax]
    sizes=[xsize,ysize]
    sliceoffset = abs(zmin)
    xyz = 2
    idlist, indexlist = ids3d.ids3d(cellids, sliceoffset, reflevel, xsize, ysize, zsize, zmin=zmin, zmax=zmax)


    # Axes and units (default R_E)
    if axisunit is not None: # Use m or km or other
        if np.isclose(axisunit,0):
            axisunitstr = pt.plot.rmstring('m')
        elif np.isclose(axisunit,3):
            axisunitstr = pt.plot.rmstring('km')
        else:
            axisunitstr = r'$10^{'+str(int(axisunit))+'} '+pt.plot.rmstring('m')
        axisunit = np.power(10,int(axisunit))
    else:
        axisunitstr = pt.plot.rmstring('R')+'_'+pt.plot.rmstring('E')
        axisunit = Re


    # Select window to draw
    if len(boxm)==4:
        boxcoords=list(boxm)+[i*axisunit for i in z_extent]
    elif len(boxre)==4:
        boxcoords=[i*Re for i in boxre]+[i*Re for i in z_extent]
    else:
        boxcoords=list(simext)

    # If box extents were provided manually, truncate to simulation extents
    boxcoords[0] = max(boxcoords[0],simext[0])
    boxcoords[1] = min(boxcoords[1],simext[1])
    boxcoords[2] = max(boxcoords[2],simext[2])
    boxcoords[3] = min(boxcoords[3],simext[3])
    boxcoords[4] = max(boxcoords[4],simext[4])
    boxcoords[5] = min(boxcoords[5],simext[5])

    # Scale data extent and plot box
    simext=[i/axisunit for i in simext]
    boxcoords=[i/axisunit for i in boxcoords]

    #################################################
    # Find rhom map for use in masking out ionosphere
    #################################################
    if f.check_variable("moments"):
        rhomap = f.read_variable("vg_restart_rhom")
    elif f.check_variable("proton/vg_rho"):
        rhomap = f.read_variable("proton/vg_rho")
    elif f.check_variable("vg_rhom"):
        rhomap = f.read_variable("vg_rhom")
    elif f.check_variable("rho"): #old non-AMR data, can still be 3D
        rhomap = f.read_variable("rho")
    else:
        logging.info("error!")
        quit
              
    rhomap = rhomap[indexids] # sort
    rhomap = rhomap[indexlist] # find required cells
    # Create the plotting grid
    rhomap = ids3d.idmesh3d(idlist, rhomap, reflevel, xsize, ysize, zsize, xyz, None)

    # scale the sizes to the heighest refinement level because
    # plotting is done at that level
    sizes[0] = int(sizes[0]*2**reflevel)
    sizes[1] = int(sizes[1]*2**reflevel)

    # Allow title override
    if cbtitle is not None:
        # Here allow underscores for manual math mode
        cb_title_use = cbtitle       

    # Generates the mesh to map the data to.
    [XmeshXY,YmeshXY] = np.meshgrid(np.linspace(simext[0],simext[1],num=sizes[0]+1),np.linspace(simext[2],simext[3],num=sizes[1]+1))

    # The grid generated by meshgrid has all four corners for each cell.
    # We mask using only the centre values.
    # Calculate offsets for cell-centre coordinates
    XmeshCentres = XmeshXY[:-1,:-1] + 0.5*(XmeshXY[0,1]-XmeshXY[0,0])
    YmeshCentres = YmeshXY[:-1,:-1] + 0.5*(YmeshXY[1,0]-YmeshXY[0,0])
    maskgrid = np.ma.array(XmeshCentres)
    if not pass_full:
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
        XmeshPass = XmeshXY[MaskX[0]:MaskX[-1]+1,:]
        XmeshPass = XmeshPass[:,MaskY[0]:MaskY[-1]+1]
        YmeshPass = YmeshXY[MaskX[0]:MaskX[-1]+1,:]
        YmeshPass = YmeshPass[:,MaskY[0]:MaskY[-1]+1]
        XmeshCentres = XmeshCentres[MaskX[0]:MaskX[-1]+1,:]
        XmeshCentres = XmeshCentres[:,MaskY[0]:MaskY[-1]+1]
        YmeshCentres = YmeshCentres[MaskX[0]:MaskX[-1]+1,:]
        YmeshCentres = YmeshCentres[:,MaskY[0]:MaskY[-1]+1]

    else:
        XmeshPass = np.ma.array(XmeshXY)
        YmeshPass = np.ma.array(YmeshXY)

    sheet_xyz, folds = sheet_coordinate_finder(f, boxcoords, axisunit, cellids, reflevel, indexids, XmeshPass, YmeshPass, sheetlayer)

    # Read and sort coordinates for data interpolation
    vg_coordinates = f.read_variable("vg_coordinates")
    vg_coordinates = vg_coordinates[indexids]


    ############################################
    # Read data and calculate required variables
    ############################################
    if not expression:        
        # Read data from file
        if operator is None:
            operator="pass"
        datamap_info = f.read_variable_info(var, operator=operator)

        cb_title_use = datamap_info.latex
        # Check if vscale results in standard unit
        vscale, _, datamap_unit_latex = datamap_info.get_scaled_units(vscale=vscale)

        # Add unit to colorbar title
        if datamap_unit_latex:
            cb_title_use = cb_title_use + r"\,["+datamap_unit_latex+"]"

        datamap = datamap_info.data

        # Verify data shape
        if np.ndim(datamap)==0:
            logging.info("Error, read only single value from vlsv file! datamap.shape being " + str(datamap.shape))
            return -1
        elif np.ndim(datamap)==3: # Vector variable
            datamap = np.linalg.norm(datamap, axis=-1)
        
        # vlasov grid, AMR
        datamap = datamap[indexids] # sort

        data_interpolation = scipy.interpolate.RBFInterpolator(vg_coordinates, datamap, neighbors = 27)

        datamap = data_interpolation(sheet_xyz).reshape(XmeshPass.shape)

    else:
        # Expression set, use generated or provided colorbar title
        cb_title_use = expression.__name__ + operatorstr

    #Attempt to call external and expression functions to see if they have required
    # variable information (If they accept the requestvars keyword, they should
    # return a list of variable names as strings)
    pass3d = False
    meshReflevel = reflevel
    reqvariables = None
    if expression: # Check the expression
        try:
            reqvariables = expression(None,True)
        except:
            pass
    if external: # Check the external
        try:
            reqvariables = external(None,None,None,None,True)
        except:
            pass
    if reqvariables:
        try:
            for i in reqvariables:
                if i == "3d":
                    pass3d = True
                elif i == "noupscale":
                    meshReflevel = 0
                elif not (i in pass_vars): 
                    pass_vars.append(i)
        except:
            pass

    # If expression or external routine need variables, read them from the file.
    if pass_vars:        
        if not pass_times:
            # Note: pass_maps is now a dictionary
            pass_maps = {}
            # Gather the required variable maps for a single time step
            for mapval in pass_vars:

                # vlasov grid, AMR
                pass_map = f.read_variable(mapval)
                pass_map = pass_map[indexids] # sort

                if np.ndim(pass_map)==1:    
                    var_interpolation = scipy.interpolate.RBFInterpolator(vg_coordinates, pass_map, neighbors = 27)
                    pass_map = var_interpolation(sheet_xyz).reshape(XmeshPass.shape)
                elif np.ndim(pass_map)==2: # vector variable
                    temporary_pass_map = []
                    for s in range(pass_map.shape[1]):
                        var_interpolation = scipy.interpolate.RBFInterpolator(vg_coordinates, pass_map[:,s], neighbors = 27)
                        temporary_pass_map.append(var_interpolation(sheet_xyz).reshape(XmeshPass.shape))
                    pass_map = np.dstack(tuple(temporary_pass_map))

                elif np.ndim(pass_map)==3:  # tensor variable
                    logging.info("Tensor support has not been implemented yet! Aborting")
                    return -1
                else:
                    logging.info("Error in reshaping pass_maps!")


                pass_maps[mapval] = pass_map # add to the dictionary
        else:
            # Or gather over a number of time steps
            # Note: pass_maps is now a list of dictionaries
            pass_maps = []
            if diff:
                logging.info("Comparing files "+filename+" and "+diff)
            elif step is not None and filename:
                currstep = step
            else:
                if filename: # parse from filename
                    currstep = int(filename[-12:-5])
                else:
                    logging.info("Error, cannot determine current step for time extent extraction!")
                    return
            # define relative time step selection
            if np.ndim(pass_times)==0:
                dsteps = np.arange(-abs(int(pass_times)),abs(int(pass_times))+1)
            elif np.ndim(pass_times)==1 and len(pass_times)==2:
                dsteps = np.arange(-abs(int(pass_times[0])),abs(int(pass_times[1]))+1)
            else:
                logging.info("Invalid value given to pass_times")
                return
            # Loop over requested times
            for ds in dsteps:
                if diff:
                    if ds==0:
                        filenamestep = filename
                    else:
                        filenamestep = diff
                else:
                    # Construct using known filename.
                    filenamestep = filename[:-12]+str(currstep+ds).rjust(7,'0')+'.vlsv'
                    logging.info(filenamestep)
                fstep=pt.vlsvfile.VlsvReader(filenamestep)
                step_cellids = fstep.read_variable("CellID")
                step_indexids = step_cellids.argsort()
                step_cellids = step_cellids[step_indexids]
                step_reflevel = ids3d.refinement_level(xsize, ysize, zsize, step_cellids[-1])
                for i in range(5): # Check if Vlasov grid doesn't reach maximum (fsgrid) refinement
                    if xsize*(2**(step_reflevel + i)) == xsizefg:
                        step_reflevel += i
                        break

                fsheet_xyz, ffolds = sheet_coordinate_finder(fstep, boxcoords, axisunit, step_cellids, step_reflevel, step_indexids, XmeshPass, YmeshPass)


                # Append new dictionary as new timestep
                pass_maps.append({})
                # Add relative step identifier to dictionary
                pass_maps[-1]['dstep'] = ds
                # Gather the required variable maps
                for mapval in pass_vars:

                    # vlasov grid, AMR
                    pass_map = fstep.read_variable(mapval)
                    pass_map = pass_map[step_indexids] # sort

                    if np.ndim(pass_map)==1:    
                        var_interpolation = scipy.interpolate.RBFInterpolator(vg_coordinates, pass_map, neighbors = 27)
                        pass_map = var_interpolation(fsheet_xyz).reshape(XmeshPass.shape)
                    elif np.ndim(pass_map)==2: # vector variable
                        temporary_pass_map = []
                        for s in range(pass_map.shape[1]):
                            var_interpolation = scipy.interpolate.RBFInterpolator(vg_coordinates, pass_map[:,s], neighbors = 27)
                            temporary_pass_map.append(var_interpolation(fsheet_xyz).reshape(XmeshPass.shape))
                        pass_map = np.dstack(tuple(temporary_pass_map))

                    elif np.ndim(pass_map)==3:  # tensor variable
                        logging.info("Tensor support has not been implemented yet! Aborting")
                        return -1
                    else:
                        logging.info("Error in reshaping pass_maps!")
                    pass_maps[-1][mapval] = pass_map # add to the dictionary

    # colorbar title for diffs:
    if diff:
        listofkeys = iter(pass_maps[0])
        while True:
            diffvar = next(listofkeys)
            if diffvar!="dstep": break
        cb_title_use = pt.plot.mathmode(pt.plot.bfstring(pt.plot.rmstring("DIFF0~"+diffvar.replace("_",r"\_"))))
    # Evaluate time difference
    if diff:
        tvf=pt.vlsvfile.VlsvReader(filename)
        t0 = tvf.read_parameter('time')
        tvf1=pt.vlsvfile.VlsvReader(diff)
        t1 = tvf1.read_parameter('time')
        if (not np.isclose(t1-t0, 0.0, rtol=1e-6)):
            plot_title = plot_title + "~dt=" + str(t1-t0)


    #Optional user-defined expression used for color panel instead of a single pre-existing var
    if expression:
        datamap = expression(pass_maps)

        # Handle operators

        if (operator and (operator != 'pass') and (operator != 'magnitude')):
            if operator=='x': 
                operator = '0'
            if operator=='y': 
                operator = '1'
            if operator=='z': 
                operator = '2'
            if not operator.isdigit():
                logging.info("Error parsing operator for custom expression!")
                return
            elif np.ndim(datamap)==3:
                datamap = datamap[:,:,int(operator)]
        
    # Now, if map is a vector or tensor, reduce it down
    if np.ndim(datamap)==3: # vector
        if datamap.shape[2]!=3:
            logging.info("Error, expected array of 3-element vectors, found array of shape " + str(datamap.shape))
            return -1
        # take magnitude of three-element vectors
        datamap = np.linalg.norm(datamap, axis=-1)
    if np.ndim(datamap)==4: # tensor
        if datamap.shape[2]!=3 or datamap.shape[3]!=3:
            # This may also catch 3D simulation fsgrid variables
            logging.info("Error, expected array of 3x3 tensors, found array of shape " + str(datamap.shape))
            return -1
        # take trace
        datamap = datamap[:,:,0,0]+datamap[:,:,1,1]+datamap[:,:,2,2]
    if np.ndim(datamap)>=5: # Too many dimensions
        logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap.shape))
        return -1
    if np.ndim(datamap)!=2: # Too many dimensions
        logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap.shape))
        return -1

    # Scale final generated datamap if requested
    datamap = datamap * vscale
    
    # Take absolute
    if (absolute):
        datamap = abs(datamap)

    # Crop both rhomap and datamap to view region
    if np.ma.is_masked(maskgrid):
        # Strip away columns and rows which are outside the plot region
        rhomap = rhomap[MaskX[0]:MaskX[-1]+1,:]
        rhomap = rhomap[:,MaskY[0]:MaskY[-1]+1]


    # Mask region outside ionosphere. Note that for some boundary layer cells, 
    # a density is calculated, but e.g. pressure is not, and these cells aren't
    # excluded by this method. Also mask away regions where datamap is invalid
    if not nomask:
        rhomap = np.ma.masked_less_equal(np.ma.masked_invalid(rhomap), 0)
        rhomap = np.ma.masked_where(~np.isfinite(datamap), rhomap)
        XYmask = rhomap.mask
        if XYmask.any():
            if XYmask.all():
                # if everything was masked in rhomap, allow plotting
                XYmask[:,:] = False
            else:
                # Mask datamap
                datamap = np.ma.array(datamap, mask=XYmask)

    #If automatic range finding is required, find min and max of array
    # Performs range-finding on a masked array to work even if array contains invalid values
    if vmin is not None:
        vminuse=vmin
    else: 
        vminuse=np.ma.amin(datamap)
    if vmax is not None:
        vmaxuse=vmax
    else:
        vmaxuse=np.ma.amax(datamap)

    # If both values are zero, we have an empty array
    if vmaxuse==vminuse==0:
        logging.info("Error, requested array is zero everywhere. Exiting.")
        return 0

    # If vminuse and vmaxuse are extracted from data, different signs, and close to each other, adjust to be symmetric
    # e.g. to plot transverse field components. Always done for symlog.
    if vmin is None and vmax is None:
        if symmetric or np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2) or symlog is not None:
            absval = max(abs(vminuse),abs(vmaxuse))
            vminuse = -absval
            vmaxuse = absval

    # Ensure that lower bound is valid for logarithmic plots
    if (vminuse <= 0) and (lin is None) and (symlog is None):
        # Drop negative and zero values
        vminuse = np.ma.amin(np.ma.masked_less_equal(datamap,0))

    # Special case of very small vminuse values
    if ((vmin is None) or (vmax is None)) and (vminuse > 0) and (vminuse < vmaxuse*1.e-5):
        vminuse = vmaxuse*1e-5
        if lin is not None:
            vminuse = 0
    
    # # If symlog scaling is set:
    if symlog is not None:
        if symlog > 0:
            linthresh = symlog 
        else:
            linthresh = max(abs(vminuse),abs(vmaxuse))*1.e-2

    # Lin or log colour scaling, defaults to log
    if lin is None:
        # Special SymLogNorm case
        if symlog is not None:
            if Version(matplotlib.__version__) < Version("3.2.0"):
                norm = SymLogNorm(linthresh=linthresh, linscale = 1.0, vmin=vminuse, vmax=vmaxuse, clip=True)
                logging.info("WARNING: colormap SymLogNorm uses base-e but ticks are calculated with base-10.")
                #TODO: copy over matplotlib 3.3.0 implementation of SymLogNorm into analysator
            else:
                norm = SymLogNorm(base=10, linthresh=linthresh, linscale = 1.0, vmin=vminuse, vmax=vmaxuse, clip=True)
            maxlog=int(np.ceil(np.log10(vmaxuse)))
            minlog=int(np.ceil(np.log10(-vminuse)))
            logthresh=int(np.floor(np.log10(linthresh)))
            logstep=1
            ticks=([-(10**x) for x in range(logthresh, minlog+1, logstep)][::-1]
                    +[0.0]
                    +[(10**x) for x in range(logthresh, maxlog+1, logstep)] )
        else:
            # Logarithmic plot
            norm = LogNorm(vmin=vminuse,vmax=vmaxuse)
            ticks = LogLocator(base=10,subs=range(10)) # where to show labels
    else:
        # Linear
        levels = MaxNLocator(nbins=255).tick_values(vminuse,vmaxuse)
        norm = BoundaryNorm(levels, ncolors=cmapuse.N, clip=True)
        ticks = np.linspace(vminuse,vmaxuse,num=lin)            

    # Select plotting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if not axes: # If axes are provided, leave backend as-is.
        if draw:
            if str(matplotlib.get_backend()) != pt.backend_interactive: #'TkAgg': 
                plt.switch_backend(pt.backend_interactive)
        else:
            if str(matplotlib.get_backend()) != pt.backend_noninteractive: #'Agg':
                plt.switch_backend(pt.backend_noninteractive)  

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
    if (len(plot_title)==0 and (nocb or internalcb) and noborder and noxlabels and noylabels):
        ratio = (boxcoords[3]-boxcoords[2])/(boxcoords[1]-boxcoords[0])
        figsize = [4.0,4.0*ratio]

    # If requested high res image
    if highres:
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
        streamlinethick=streamlinethick*highresscale
        vectorsize=vectorsize*highresscale

    if not axes:
        # Create 300 dpi image of suitable size
        fig = plt.figure(figsize=figsize,dpi=300)
        ax1 = plt.gca() # get current axes
    else:
        ax1=axes
        fig = plt.gcf() # get current figure

    # Plot the actual mesh
    if(not useimshow):
        fig1 = ax1.pcolormesh(XmeshPass,YmeshPass,datamap, cmap=colormap,norm=norm)
    else:
        fig1 = ax1.imshow(datamap,
                          cmap=colormap,
                          norm=norm,
                          interpolation=imshowinterp,
                          origin='lower',
                          extent=(np.min(XmeshPass), np.max(XmeshPass), np.min(YmeshPass), np.max(YmeshPass))
                         )

    # Title and plot limits
    if len(plot_title)!=0:
        # Add 3D slice position in title
        plot_title = plot_title
        plot_title = pt.plot.mathmode(pt.plot.bfstring(plot_title))        
        ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')

    ax1.set_xlim([boxcoords[0],boxcoords[1]])
    ax1.set_ylim([boxcoords[2],boxcoords[3]])
    ax1.set_aspect('equal')

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(thick)
    ax1.xaxis.set_tick_params(width=thick,length=3*thick)
    ax1.yaxis.set_tick_params(width=thick,length=3*thick)
    #ax1.xaxis.set_tick_params(which='minor',width=3,length=5)
    #ax1.yaxis.set_tick_params(which='minor',width=3,length=5)

    if not noxlabels:
        xlabelstr = pt.plot.mathmode(pt.plot.bfstring(axislabels[0]+r'\,['+axisunitstr+']'))
        ax1.set_xlabel(xlabelstr,fontsize=fontsize,weight='black')
        for item in ax1.get_xticklabels():
            item.set_fontsize(fontsize)
            item.set_fontweight('black')
        ax1.xaxis.offsetText.set_fontsize(fontsize)# set axis exponent offset font sizes
    if not noylabels:
        ylabelstr = pt.plot.mathmode(pt.plot.bfstring(axislabels[1]+r'\,['+axisunitstr+']'))
        ax1.set_ylabel(ylabelstr,fontsize=fontsize,weight='black')
        for item in ax1.get_yticklabels():
            item.set_fontsize(fontsize)
            item.set_fontweight('black')
        ax1.yaxis.offsetText.set_fontsize(fontsize)# set axis exponent offset font sizes


    if Earth:
        Earth = matplotlib.patches.Circle((0, 0), 1.0, color='k')
        Earth2 = matplotlib.patches.Wedge((0,0), 0.9, -90, 90, fc='white', ec=None,lw=0.0)
        ax1.add_artist(Earth)
        ax1.add_artist(Earth2)

    if folding_alpha > 0:
        folds_to_plot=np.array(folds)
        ax1.scatter(folds_to_plot[:,0]/axisunit, folds_to_plot[:,1]/axisunit, c='k', s=0.1, alpha=folding_alpha)


    if streamlines:
        logging.info("Streamline support not implemented yet! Aborting")
        return -1       
        


    # add vectors on top
    if vectors:

        sheet_cids = [f.get_cellid(sheet_coord) for sheet_coord in sheet_xyz]   #slow
        xlen, ylen = XmeshPass.shape
        
        # vlasov grid, AMR
        vectmap = f.read_variable(vectors, sheet_cids)
        vectmap = vectmap.reshape(xlen, ylen, 3)

        if XYmask.any():
            vectmap = np.ma.array(vectmap)
            for i in range(3):
                vectmap[:,:,i].mask = XYmask

    
        # Find vector lengths and define color
        lengths=np.linalg.norm(vectmap, axis=-1)
        # Mask out the smallest vectors (at e.g. inner boundary)
        lengths=np.ma.masked_less(lengths, 0.01*np.amax(lengths))
        colors = np.ma.log10(np.ma.divide(lengths,np.ma.mean(lengths)))

        # Try to estimate vectstep so there's about 100 vectors in the image area
        visibleboxcells = (axisunit**2)*(boxcoords[1]-boxcoords[0])*(boxcoords[3]-boxcoords[2])/((cellsize*(0.5**reflevel))**2)
        vectstep = int(np.sqrt(visibleboxcells/vectordensity))
        vectstep = max(1,vectstep)
        
        # inplane unit length vectors
        vectmap[:,:,2] = np.ma.zeros(vectmap[:,:,2].shape)
        vectmap = np.ma.divide(vectmap, np.linalg.norm(vectmap, axis=-1)[:,:,np.newaxis])
       
        X = XmeshCentres[::vectstep,::vectstep]
        Y = YmeshCentres[::vectstep,::vectstep]

        U = vectmap[::vectstep,::vectstep,0]            
        V = vectmap[::vectstep,::vectstep,1]
        C = colors[::vectstep,::vectstep] 
        # quiver uses scale in the inverse fashion
    
        ax1.quiver(X,Y,U,V,C, cmap=vectorcolormap, units='dots', scale=0.05/vectorsize, headlength=4, headwidth=4,
                   headaxislength=2, scale_units='dots', pivot='middle')
    

    # Optional external additional plotting routine overlayed on color plot
    # Uses the same pass_maps variable as expressions
    if external:
        #extresult=external(ax1, XmeshXY,YmeshXY, pass_maps)
        if not axes:
            extresult=external(ax1, XmeshCentres,YmeshCentres, pass_maps)
        else:
            extresult=external(axes, XmeshCentres,YmeshCentres, pass_maps)


    if not nocb:
        if cbaxes: 
            # Colorbar axes are provided
            cax = cbaxes
            cbdir="right"; horalign="left"
        elif internalcb:
            # Colorbar within plot area
            cbloc=1; cbdir="left"; horalign="right"
            if type(internalcb) is str:
                if internalcb=="NW":
                    cbloc=2; cbdir="right"; horalign="left"
                if internalcb=="SW": 
                    cbloc=3; cbdir="right"; horalign="left"
                if internalcb=="SE": 
                    cbloc=4; cbdir="left";  horalign="right"
            # borderpad default value is 0.5, need to increase it to make room for colorbar title
            cax = inset_locator.inset_axes(ax1, width="5%", height="35%", loc=cbloc, borderpad=1.5,
                             bbox_transform=ax1.transAxes, bbox_to_anchor=(0,0,1,1))
        else:
            # Split existing axes to make room for colorbar
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbdir="right"; horalign="left"

        # Colourbar title
        if len(cb_title_use)!=0:
            cb_title_use = pt.plot.mathmode(pt.plot.bfstring(cb_title_use))

        # Set flag which affects colorbar decimal precision
        if lin is None:
            pt.plot.cb_linear = False
        else:
            pt.plot.cb_linear = True

        # First draw colorbar
        if usesci:
            cb = plt.colorbar(fig1, ticks=ticks, format=mtick.FuncFormatter(pt.plot.cbfmtsci), cax=cax, drawedges=False)
        else:
            #cb = plt.colorbar(fig1, ticks=ticks, format=mtick.FormatStrFormatter('%4.2f'), cax=cax, drawedges=False)
            cb = plt.colorbar(fig1, ticks=ticks, format=mtick.FuncFormatter(pt.plot.cbfmt), cax=cax, drawedges=False)
        cb.outline.set_linewidth(thick)
        cb.ax.yaxis.set_ticks_position(cbdir)
        # Ensure minor tick marks are off
        if lin is not None:
            cb.minorticks_off()

        if not cbaxes:
            cb.ax.tick_params(labelsize=fontsize3,width=thick,length=3*thick)
            cb_title = cax.set_title(cb_title_use,fontsize=fontsize3,fontweight='bold', horizontalalignment=horalign)
            cb_title.set_position((0.,1.+0.025*scale)) # avoids having colourbar title too low when fontsize is increased
        else:
            cb.ax.tick_params(labelsize=fontsize,width=thick,length=3*thick)
            cb_title = cax.set_title(cb_title_use,fontsize=fontsize,fontweight='bold', horizontalalignment=horalign)

        # Perform intermediate draw if necessary to gain access to ticks
        if (symlog is not None and np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2)) or (not lin and symlog is None):
            fig.canvas.draw() # draw to get tick positions

        # Adjust placement of innermost ticks for symlog if it indeed is (quasi)symmetric
        if symlog is not None and np.isclose(vminuse/vmaxuse, -1.0, rtol=0.2):
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

        # Adjust precision for colorbar ticks
        thesetickvalues = cb.locator()
        if len(thesetickvalues)<2:
            precision_b=1
        else:
            mintickinterval = abs(thesetickvalues[-1]-thesetickvalues[0])
            # find smallest interval
            for ticki in range(len(thesetickvalues)-1):
                mintickinterval = min(mintickinterval,abs(thesetickvalues[ticki+1]-thesetickvalues[ticki]))
            precision_a, precision_b = '{:.1e}'.format(mintickinterval).split('e')
            # e.g. 9.0e-1 means we need precision 1
            # e.g. 1.33e-1 means we need precision 3?
        pt.plot.decimalprecision_cblin = 1
        if int(precision_b)<1: pt.plot.decimalprecision_cblin = str(1+abs(-int(precision_b)))
        cb.update_ticks()

        # if too many subticks in logarithmic colorbar:
        if not lin and symlog is None:
            nlabels = len(cb.ax.yaxis.get_ticklabels()) / ratio
            # Force less ticks for internal colorbars
            if internalcb: 
                nlabels = nlabels * 1.5
            valids = ['1','2','3','4','5','6','7','8','9']
            if nlabels > 10:
                valids = ['1','2','3','4','5','6','8']
            if nlabels > 19:
                valids = ['1','2','5']
            if nlabels > 28:
                valids = ['1']
            # for label in cb.ax.yaxis.get_ticklabels()[::labelincrement]:
            for labi,label in enumerate(cb.ax.yaxis.get_ticklabels()):
                labeltext = label.get_text().replace('$','').replace('{','').replace('}','').replace(r'\mbox{\textbf{--}}','').replace('-','').replace('.','').lstrip('0')
                if not labeltext:
                    continue
                firstdigit = labeltext[0]
                if not firstdigit in valids: 
                    label.set_visible(False)

    # Add Vlasiator watermark
    if (wmark or wmarkb) and not axes:
        if wmark:
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

    # Find required precision
    mintickinterval = np.amax(boxcoords)-np.amin(boxcoords)
    if not tickinterval:
        fig.canvas.draw() # draw to get tick positions
    for axisi, axis in enumerate([ax1.xaxis, ax1.yaxis]):
        if tickinterval:
            axis.set_major_locator(mtick.MultipleLocator(tickinterval))
            mintickinterval = tickinterval
        else: # Find tick interval
            thesetickvalues = axis.get_major_locator()()
            mintickinterval = min(mintickinterval,abs(thesetickvalues[1]-thesetickvalues[0]))

    # Adjust axis tick labels
    for axisi, axis in enumerate([ax1.xaxis, ax1.yaxis]):
        # Set required decimal precision
        pt.plot.decimalprecision_ax = '0'
        precision_a, precision_b = '{:.1e}'.format(mintickinterval).split('e')
        # e.g. 9.0e-1 means we need precision 1
        if int(precision_b)<1: pt.plot.decimalprecision_ax = str(abs(-int(precision_b)))
        # Find maximum possible lengths of axis tick labels
        # Only counts digits
        axisminmax = boxcoords[axisi*2:axisi*2+2]
        ticklens = [ len(re.sub(r'\D',"",pt.plot.axisfmt(bc,None))) for bc in axisminmax]
        tickmaxlens = np.amax(ticklens[0:1])
        
        # Custom tick formatter
        axis.set_major_formatter(mtick.FuncFormatter(pt.plot.axisfmt))
        ticklabs = axis.get_ticklabels()
        # Set boldface.
        for t in ticklabs:
            t.set_fontweight("black")
            # If label has >3 numbers, tilt it
            if tickmaxlens>3: 
                t.set_rotation(30)
                t.set_verticalalignment('top')
                t.set_horizontalalignment('right')

    # Or turn x-axis labels off
    if noxlabels:
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False) 
    # Or turn y-axis labels off
    if noylabels:
        for label in ax1.yaxis.get_ticklabels():
            label.set_visible(False)

    # Adjust layout. Uses tight_layout() but in fact this ensures 
    # that long titles and tick labels are still within the plot area.
    if axes:
        savefig_pad=0.01
        bbox_inches='tight'
    elif not noborder:
        plt.tight_layout()
        savefig_pad=0.05 # The default is 0.1
        bbox_inches=None
    else:
        plt.tight_layout(pad=0.01)
        savefig_pad=0.01
        bbox_inches='tight'
        
    # Save output or draw on-screen
    if not draw and not axes:
        try:
            plt.savefig(outputfile,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
        except:
            logging.info("Error with attempting to save figure.")
        logging.info(outputfile+"\n")
    elif not axes:
        # Draw on-screen
        plt.draw()
        plt.show()


def sheet_coordinate_finder(f, boxcoords, axisunit, cellids, reflevel, indexids, XmeshPass, YmeshPass, sheetlayer):
    """
    Helper function for plot_neutral_sheet
    """
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)    
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize

    # Read the FSgrid mesh
    try:
        [xsizefg, ysizefg, zsizefg] = f.get_fsgrid_mesh_size()
        xsizefg = int(xsizefg)
        ysizefg = int(ysizefg)
        zsizefg = int(zsizefg)
        [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = f.get_fsgrid_mesh_extent()
        cellsizefg = (xmaxfg-xminfg)/xsizefg
        pt.plot.plot_helpers.CELLSIZE = cellsizefg
    except:
        if xsize!=1 and ysize!=1 and zsize!=1:
            logging.info("Did not find fsgrid data, but found 3D DCCRG mesh. Attempting to adapt.")
            [xsizefg, ysizefg, zsizefg] = [xsize * 2**f.get_max_refinement_level(), ysize * 2**f.get_max_refinement_level(), zsize * 2**f.get_max_refinement_level()]
            [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = [xmin, ymin, zmin, xmax, ymax, zmax]
            cellsizefg = cellsize
            pt.plot.plot_helpers.CELLSIZE = cellsize
        else:
            logging.info("Found 2D DCCRG mesh without FSgrid data. Exiting.")
            return -1

    # Read Bx data
    sheet_datamap_info = f.read_variable_info("vg_b_vol", operator='x')

    sheet_datamap = sheet_datamap_info.data

    # Verify data shape
    if np.ndim(sheet_datamap)==0:
        logging.info("Error, read only single Bx value from vlsv file! sheet_datamap.shape being "+ str(sheet_datamap.shape))
        return -1
    

    #Define the box limits
    low = np.array([boxcoords[0], boxcoords[2], boxcoords[4]])*axisunit
    up = np.array([boxcoords[1], boxcoords[3], boxcoords[5]])*axisunit

    # Choose CellIDs inside the box
    ids, idx = ids3d.ids3d_box(cellids, low, up, reflevel, xsize, ysize, zsize, [xmin, ymin, zmin, xmax, ymax, zmax])

    sheet_datamap = sheet_datamap[indexids]
    sheet_datamap = sheet_datamap[idx]

    # Reshape surface data
    if np.ndim(sheet_datamap)==1:
        sheet_data_in_box = ids3d.idmesh3d2(ids, sheet_datamap, reflevel, xsize, ysize, zsize,  None)
    else:
        logging.info("Dimension error in constructing sheet!")
        return -1

    # Remove empty rows, columns and tubes
    empty_0 = np.where(~np.all(sheet_data_in_box==0, axis=0))
    empty_1 = np.where(~np.all(sheet_data_in_box==0, axis=1))

    boxMaskX = empty_1[0]
    boxMaskY = empty_0[0]
    boxMaskZ = empty_1[1]

    cropped_sheet_data = sheet_data_in_box[min(boxMaskX):max(boxMaskX)+1,:,:]
    cropped_sheet_data = cropped_sheet_data[:,min(boxMaskY):max(boxMaskY)+1,:]
    cropped_sheet_data = cropped_sheet_data[:,:,min(boxMaskZ):max(boxMaskZ)+1]


    # Keep removing the face of the rectangle with the largest number of invalid values until the surface has only proper values
    counter = 0
    while True:
        zero_counts = np.zeros(6)
        zero_counts[0] = np.count_nonzero(cropped_sheet_data[0, :, :]  == 0) # Face 0 (front)
        zero_counts[1] = np.count_nonzero(cropped_sheet_data[-1, :, :] == 0)  # Face 1 (back)
        zero_counts[2] = np.count_nonzero(cropped_sheet_data[:, 0, :]  == 0) # Face 2 (left)
        zero_counts[3] = np.count_nonzero(cropped_sheet_data[:, -1, :] == 0)  # Face 3 (right)
        zero_counts[4] = np.count_nonzero(cropped_sheet_data[:, :, 0]  == 0) # Face 4 (top)
        zero_counts[5] = np.count_nonzero(cropped_sheet_data[:, :, -1] == 0)  # Face 5 (bottom)

        face_to_be_removed = np.argmax(zero_counts)

        if face_to_be_removed==0:
            cropped_sheet_data = cropped_sheet_data[1:,:,:]
        elif face_to_be_removed==1:
            cropped_sheet_data = cropped_sheet_data[:-1,:,:]
        elif face_to_be_removed==2:
            cropped_sheet_data = cropped_sheet_data[:,1:,:]
        elif face_to_be_removed==3:
            cropped_sheet_data = cropped_sheet_data[:,:-1,:]
        elif face_to_be_removed==4:
            cropped_sheet_data = cropped_sheet_data[:,:,1:]
        elif face_to_be_removed==5:
            cropped_sheet_data = cropped_sheet_data[:,:,:-1]

        counter+=1
        if counter > 50:    # Failsafe
            logging.info("Error reading sheet variable vg_b_vol! Exiting.")
            return -1
        
        if np.all(zero_counts==0):
            break 

    spacing = ((xmaxfg-xminfg)/(xsizefg*axisunit), (ymaxfg-yminfg)/(ysizefg*axisunit), (zmaxfg-zminfg)/(zsizefg*axisunit))

    verts, faces, normals, arrays = measure.marching_cubes(cropped_sheet_data, level=0.,
                                                                   spacing=spacing,
                                                                   gradient_direction='descent',
                                                                   step_size=1,
                                                                   allow_degenerate=True, method='lewiner', mask=cropped_sheet_data!=0)
    

    #offset with respect to simulation domain corner
    verts[:,0] = verts[:,0] + low[0]/axisunit
    verts[:,1] = verts[:,1] + low[1]/axisunit
    verts[:,2] = verts[:,2] + low[2]/axisunit

    nverts = len(verts[:,0])
    all_coords = np.empty((nverts, 3))
    for i in np.arange(nverts):            
        coords = verts[i,:]*axisunit
        all_coords[i] = coords

    all_x = all_coords[:,0]     # Coordinates of the sheet
    all_y = all_coords[:,1]
    all_z = all_coords[:,2]

    folds=[]    # List to hold fold locations
    pointdict = {}  # Dictionary to keep track of which xy locations have already been recorded
    flagdict = {}   # Dictionary to keep track of fold locations

    for i in range(len(all_x)):
        key = (round(all_x[i]/cellsizefg),round(all_y[i]/cellsizefg))     # Key for hashing xy coordinates, precise to one cell
        
        if key not in pointdict:    # If no value given to XY coordinate, save sheet coordinates to dictionary
            pointdict[key] = [all_x[i],all_y[i],all_z[i]]
            
        else:   # If the XY coordinate already has a sheet coordinate associated with it, check if the sheet is folded and pick the upper (lower) z value

            if abs(round(all_z[i]/cellsizefg) - round(pointdict[key][2]/cellsizefg)) > 1:  # If z values differ by more than one fg dx
                if key not in flagdict:     # Only calculate folds if there are three different z values within the same xy location, record the first double instance
                    flagdict[key] = 0
                else:   # Three distinct z values for the same xy location: folding has occurred
                    folds.append([all_x[i],all_y[i]])
            
            if sheetlayer=='above':
                if all_z[i] > pointdict[key][2]:
                    pointdict[key] = [all_x[i],all_y[i],all_z[i]]
            else:
                if all_z[i] < pointdict[key][2]:
                    pointdict[key] = [all_x[i],all_y[i],all_z[i]]


    sheet_points = np.asarray(list(pointdict.values()))     # All sheet coordinates
    xy_points = sheet_points[:,0:2]     # Sheet XY values
    z_points = sheet_points[:,2]        # Sheet Z values

    z_interpolation = scipy.interpolate.RBFInterpolator(xy_points, z_points, neighbors=27)  # Function to interpolate the location of the sheet

    x_flat = XmeshPass.ravel()*axisunit
    y_flat = YmeshPass.ravel()*axisunit

    # Combine the flattened arrays into an array of coordinate pairs
    xy_pairs = np.column_stack((x_flat, y_flat))

    sheet_zs = z_interpolation(xy_pairs)    # Interpolated sheet location at grid values

    return np.hstack((xy_pairs, sheet_zs[:, np.newaxis])), folds    # Return list of sheet coordinates interpolated to grid values and information of the possible foldings
