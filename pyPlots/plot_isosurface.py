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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage import measure

import scipy
import os, sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data

# Run TeX typesetting through the full TeX engine instead of python's own mathtext. Allows
# for changing fonts, bold math symbols etc, but may cause trouble on some systems.
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['text.dvipnghack'] = 'True' # This hack might fix it on some systems
#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = 'cmmib10' #'cm' 

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

# Different style scientific format for colour bar ticks
def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}\times10^{{{}}}$'.format(a, b)


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
                    #
                    boxm=[],boxre=[],
                    colormap=None,
                    run=None,wmark=None, nocb=None,
                    unit=None, thick=1.0,scale=1.0,

                    vmin=None, vmax=None, lin=None, symlog=None,
                    ):

    ''' Plots a coloured plot with axes and a colour bar.

    :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj:     Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename
    :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/)
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
    :kword vmin,vmax:   min and max values for colour scale and colour bar. If no values are given,
                        min and max values for whole plot (non-zero rho regions only) are used.
    :kword lin:         Flag for using linear colour scaling instead of log
    :kword symlog:      Use logarithmic scaling, but linear when abs(value) is below the value given to symlog.
                        Allows symmetric quasi-logarithmic plots of e.g. transverse field components.
                        A given of 0 translates to a threshold of max(abs(vmin),abs(vmax)) * 1.e-2.
    :kword wmark:       If set to non-zero, will plot a Vlasiator watermark in the top left corner.
    :kword draw:        Set to nonzero in order to draw image on-screen instead of saving to file (requires x-windowing)

    :kword scale:       Scale text size (default=1.0)
    :kword thick:       line and axis thickness, default=1.0
    :kword nocb:        Set to suppress drawing of colourbar

    :returns:           Outputs an image to a file or to the screen.

    .. code-block:: python

    # Example usage:

    '''

    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    # watermarkimage=os.path.expandvars('$HOME/appl_taito/analysator/pyPlot/logo_color.png')

    outputprefix = ''
    if outputdir==None:
        outputdir=os.path.expandvars('$HOME/Plots/')
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
        filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
        f=pt.vlsvfile.VlsvReader(filename)
    else:
        print("Error, needs a .vlsv file name, python object, or directory and step")
        return
                
    # Scientific notation for colorbar ticks?
    if usesci==None:
        usesci=1
    
    if colormap==None:
        # Default values
        colormap="hot_desaturated"
        if color_op!=None:
            colormap="bwr"
    cmapuse=matplotlib.cm.get_cmap(name=colormap)

    fontsize=8*scale # Most text
    fontsize2=10*scale # Time title
    fontsize3=5*scale # Colour bar ticks

    # Plot title with time
    timeval=None
    timeval=f.read_parameter("time")
    if timeval==None:
        timeval=f.read_parameter("t")
    if timeval==None:
        print("Unknown time format encountered")

    # Plot title with time
    if title==None:        
        if timeval == None:    
            print("Unknown time format encountered")
            plot_title = ''
        else:
            #plot_title = "t="+str(np.int(timeval))+' s'
            plot_title = "t="+'{:4.2f}'.format(timeval)+' s'
    else:
        plot_title = title

    # step, used for file name
    if step!=None:
        stepstr = '_'+str(step).rjust(7,'0')
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
            print("Unknown operator "+color_op+", defaulting to None/magnitude for a vector.")
            color_op=None            
        else:
            # For components, always use linear scale, unless symlog is set
            color_opstr='_'+color_op
            if symlog==None:
                lin=1
    # Verify validity of operator
    if surf_op!=None:
        if surf_op!='x' and surf_op!='y' and surf_op!='z':
            print("Unknown operator "+surf_op)
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
            print("Found existing file "+savefigname+" of size zero. Re-rendering.")

    Re = 6.371e+6 # Earth radius in m
    #read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize
    cellids = f.read_variable("CellID")
    # xsize = f.read_parameter("xcells_ini")
    # ysize = f.read_parameter("ycells_ini")
    # zsize = f.read_parameter("zcells_ini")
    # xmin = f.read_parameter("xmin")
    # xmax = f.read_parameter("xmax")
    # ymin = f.read_parameter("ymin")
    # ymax = f.read_parameter("ymax")
    # zmin = f.read_parameter("zmin")
    # zmax = f.read_parameter("zmax")

    if (xsize==1) or (ysize==1) or (zsize==1):
        print("Error: isosurface plotting requires 3D spatial domain!")
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
            unitstr = r'm'
        if unit==3:
            unitstr = r'km'
        else:
            unitstr = r'$10^{'+str(int(unit))+'}$ m'
        unit = np.power(10,int(unit))
    else:
        unitstr = r'$\mathrm{R}_{\mathrm{E}}$'
        unit = Re
        
    # Scale data extent and plot box
    simext_org = simext
    simext=[i/unit for i in simext]
    boxcoords=[i/unit for i in boxcoords]

    if color_op==None and color_var!=None:
        color_op='pass'
        cb_title = color_var
        color_data = f.read_variable(color_var,cellids=[1,2])
        # If value was vector value, take magnitude
        if np.size(color_data) != 2:
            cb_title = r"$|"+color_var+"|$"
    elif color_op!=None and color_var!=None:
        cb_title = r" $"+color_var+"_"+color_op+"$"
    else: # color_var==None
        cb_title = ""
        nocb=1

    if f.check_variable(surf_var)!=True:
        print("Error, surface variable "+surf_var+" not found!")
        return
    if surf_op==None:
        surf_data = f.read_variable(surf_var)
        # If value was vector value, take magnitude
        if np.ndim(surf_data) != 1:
            surf_data = np.linalg.norm(np.asarray(surf_data),axis=-1)
    else:
        surf_data = f.read_variable(surf_var,operator=surf_op)            
    if np.ndim(surf_data)!=1:
        print("Error reading surface variable "+surf_var+"! Exiting.")
        return -1
    
    # Reshape data to ordered 3D arrays for plotting
    #color_data = color_data[cellids.argsort()].reshape([sizes[2],sizes[1],sizes[0]])
    surf_data = surf_data[cellids.argsort()].reshape([sizes[2],sizes[1],sizes[0]])
    # The data we have now is Z,Y,X
    # Rotate data so it's X,Y,Z
    surf_data = np.swapaxes(surf_data,0,2)

    if surf_level==None:
        surf_level = 0.5*(np.amin(surf_data)+np.amax(surf_data))
    print("Minimum found surface value "+str(np.amin(surf_data))+" surface level "+str(surf_level)+" max "+str(np.amax(surf_data)))

    # Select ploitting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if draw!=None:
        plt.switch_backend('TkAgg')
    else:
        plt.switch_backend('Agg')  

    # skimage.measure.marching_cubes_lewiner(volume, level=None, spacing=(1.0, 1.0, 1.0), 
    #                                        gradient_direction='descent', step_size=1, 
    #                                        allow_degenerate=True, use_classic=False)


    # #Generate mask for only visible section of data
    # # Apparently the marching cubes method does not ignore masked values, so this doesn't directly help.
    # [Xmesh,Ymesh, Zmesh] = scipy.meshgrid(np.linspace(simext[0],simext[1],num=sizes[0]),np.linspace(simext[2],simext[3],num=sizes[1]),np.linspace(simext[4],simext[5],num=sizes[2]))
    # print("simext",simext)
    # print("sizes",sizes)    
    # print("boxcoords", boxcoords)
    # maskgrid = np.ma.masked_where(Xmesh<(boxcoords[0]), Xmesh)
    # maskgrid = np.ma.masked_where(Xmesh>(boxcoords[1]), maskgrid)
    # maskgrid = np.ma.masked_where(Ymesh<(boxcoords[2]), maskgrid)
    # maskgrid = np.ma.masked_where(Ymesh>(boxcoords[3]), maskgrid)
    # maskgrid = np.ma.masked_where(Zmesh<(boxcoords[4]), maskgrid)
    # maskgrid = np.ma.masked_where(Zmesh>(boxcoords[5]), maskgrid)
    # surf_data = np.ma.asarray(surf_data)
    # surf_data.mask = maskgrid.mask
    # print(surf_data.count())

    surf_spacing = ((xmax-xmin)/(xsize*unit), (ymax-ymin)/(ysize*unit), (zmax-zmin)/(zsize*unit))
    verts, faces, normals, arrays = measure.marching_cubes_lewiner(surf_data, level=surf_level,
                                                                   spacing=surf_spacing,
                                                                   gradient_direction='descent',
                                                                   step_size=surf_step,
                                                                   allow_degenerate=True, use_classic=False)


    # offset with respect to simulation domain corner
    #print("simext",simext)
    verts[:,0] = verts[:,0] + simext[0]
    verts[:,1] = verts[:,1] + simext[2]
    verts[:,2] = verts[:,2] + simext[4]

    # # Crop surface to box area
    # verts = []
    # faces = []
    # for i in np.arange(len(verts_nocrop[:,0])):
    #     if ((verts_nocrop[i,0] > boxcoords[0]) and (verts_nocrop[i,0] < boxcoords[1]) and
    #         (verts_nocrop[i,1] > boxcoords[2]) and (verts_nocrop[i,1] < boxcoords[3]) and
    #         (verts_nocrop[i,2] > boxcoords[4]) and (verts_nocrop[i,2] < boxcoords[5])):
    #         verts.append(verts_nocrop[i,:])
    #         faces.append(faces_nocrop[i,:]) # This is now incorrect
    #     else:
    #         dropped = dropped+1
    # verts = np.asarray(verts)
    # faces = np.asarray(faces)

    # Next find color variable values at vertices
    if color_var != None:
        nverts = len(verts[:,0])
        print("Extracting color values for "+str(nverts)+" vertices and "+str(len(faces[:,0]))+" faces.")
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
        # Use interpolated values, WARNING periodic y (2.9 polar) hard-coded here /!\
        color_data = f.read_interpolated_variable(color_var, all_coords, operator=color_op, periodic=["False", "True", "False"])
        # Make sure color data is 1-dimensional (e.g. magnitude of E instead of 3 components)
        if np.ndim(color_data)!=1:
            color_data=np.linalg.norm(color_data, axis=-1)

    if color_var==None:
        # dummy norm
        print("No surface color given, using dummy setup")
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

        # If vminuse and vmaxuse are extracted from data, different signs, and close to each other, adjust to be symmetric
        # e.g. to plot transverse field components
        if vmin==None and vmax==None:
            if (vminuse*vmaxuse < 0) and (abs(abs(vminuse)-abs(vmaxuse))/abs(vminuse) < 0.4 ) and (abs(abs(vminuse)-abs(vmaxuse))/abs(vmaxuse) < 0.4 ):
                absval = max(abs(vminuse),abs(vmaxuse))
                if vminuse < 0:
                    vminuse = -absval
                    vmaxuse = absval
                else:
                    vminuse = absval
                    vmaxuse = -absval

        # Check that lower bound is valid for logarithmic plots
        if (vminuse <= 0) and (lin==None) and (symlog==None):
            # Drop negative and zero values
            vminuse = np.ma.amin(np.ma.masked_less_equal(color_data,0))

        # If symlog scaling is set:
        if symlog!=None:
            if symlog>0:
                linthresh = symlog 
            else:
                linthresh = max(abs(vminuse),abs(vmaxuse))*1.e-2

        # Lin or log colour scaling, defaults to log
        if lin==None:
            # Special SymLogNorm case
            if symlog!=None:
                #norm = SymLogNorm(linthresh=linthresh, linscale = 0.3, vmin=vminuse, vmax=vmaxuse, ncolors=cmapuse.N, clip=True)
                norm = SymLogNorm(linthresh=linthresh, linscale = 0.3, vmin=vminuse, vmax=vmaxuse, clip=True)
                maxlog=int(np.ceil(np.log10(vmaxuse)))
                minlog=int(np.ceil(np.log10(-vminuse)))
                logthresh=int(np.floor(np.log10(linthresh)))
                logstep=1
                ticks=([-(10**x) for x in range(logthresh, minlog+1, logstep)][::-1]
                        +[0.0]
                        +[(10**x) for x in range(logthresh, maxlog+1, logstep)] )
            else:
                norm = LogNorm(vmin=vminuse,vmax=vmaxuse)
                ticks = LogLocator(base=10,subs=list(range(10))) # where to show labels
        else:
            # Linear
            levels = MaxNLocator(nbins=255).tick_values(vminuse,vmaxuse)
            norm = BoundaryNorm(levels, ncolors=cmapuse.N, clip=True)
            ticks = np.linspace(vminuse,vmaxuse,num=7)            

        print("Selected color range: "+str(vminuse)+" to "+str(vmaxuse))

    # Create 300 dpi image of suitable size
    fig = plt.figure(figsize=[4.0,4.0],dpi=300)
    #ax1 = fig.gca(projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')

    # Generate virtual bounding box to get equal aspect
    maxrange = np.array([boxcoords[1]-boxcoords[0], boxcoords[3]-boxcoords[2], boxcoords[5]-boxcoords[4]]).max() / 2.0
    midvals = np.array([boxcoords[1]+boxcoords[0], boxcoords[3]+boxcoords[2], boxcoords[5]+boxcoords[4]]) / 2.0


    # Three options:
    # 2.9D ecliptic, 2.9D polar, or 3D
    if ysize < 0.2*xsize: # 2.9D polar, perform rotation
        generatedsurface = ax1.plot_trisurf(verts[:,2], verts[:,0], verts[:,1], triangles=faces,
                                            cmap=cmapuse, norm=norm, vmin=vminuse, vmax=vmaxuse, 
                                            lw=0, shade=False, edgecolors=None, antialiased=False)
        ax1.set_xlabel("z ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_ylabel("x ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_zlabel("y ["+unitstr+"]", fontsize=fontsize3)


        # Set camera angle
        ax1.view_init(elev=30., azim=90)
        # Set virtual bounding box
        ax1.set_xlim([midvals[2]-maxrange, midvals[2]+maxrange])
        ax1.set_ylim([midvals[0]-maxrange, midvals[0]+maxrange])
        ax1.set_zlim([midvals[1]-maxrange, midvals[1]+maxrange])
        ax1.tick_params(labelsize=fontsize3)#,width=1.5,length=3)
        
    else: # 3D or 2.9D ecliptic, leave as is
        generatedsurface = ax1.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces,
                                            cmap=cmapuse, norm=norm, vmin=vminuse, vmax=vmaxuse, 
                                            lw=0.2, shade=False, edgecolors=None)
        ax1.set_xlabel("x ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_ylabel("y ["+unitstr+"]", fontsize=fontsize3)
        ax1.set_zlabel("z ["+unitstr+"]", fontsize=fontsize3)
        # Set camera angle
        ax1.view_init(elev=30., azim=0)
        # Set virtual bounding box
        ax1.set_xlim([midvals[0]-maxrange, midvals[0]+maxrange])
        ax1.set_ylim([midvals[1]-maxrange, midvals[1]+maxrange])
        ax1.set_zlim([midvals[2]-maxrange, midvals[2]+maxrange])
        ax1.tick_params(labelsize=fontsize3)#,width=1.5,length=3)


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



    if cbtitle==None:
        cb_title_use = cb_title
    else:
        cb_title_use = cbtitle        


    if nocb==None:
        # First draw colorbar
        if usesci==0:        
            cb = fig.colorbar(generatedsurface,ticks=ticks, drawedges=False, fraction=0.023, pad=0.02)
        else:
            cb = fig.colorbar(generatedsurface,ticks=ticks,format=mtick.FuncFormatter(fmt),drawedges=False, fraction=0.046, pad=0.04)

        if len(cb_title_use)!=0:
            cb.ax.set_title(cb_title_use,fontsize=fontsize2,fontweight='bold')

        cb.ax.tick_params(labelsize=fontsize3)#,width=1.5,length=3)
        cb.outline.set_linewidth(thick)

        # if too many subticks:
        if lin==None and usesci!=0 and symlog==None:
            # Note: if usesci==0, only tick labels at powers of 10 are shown anyway.
            # For non-square pictures, adjust tick count
            nlabels = len(cb.ax.yaxis.get_ticklabels()) / ratio
            valids = ['1','2','3','4','5','6','7','8','9']
            if nlabels > 10:
                valids = ['1','2','3','4','5','6','8']
            if nlabels > 19:
                valids = ['1','2','5']
            if nlabels > 28:
                valids = ['1']
            # for label in cb.ax.yaxis.get_ticklabels()[::labelincrement]:
            for label in cb.ax.yaxis.get_ticklabels():
                # labels will be in format $x.0\times10^{y}$
                if not label.get_text()[1] in valids:
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
                plot_title = "t="+str(np.int(timeval))+' s   '
                ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')                
                try:
                    plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
                except:
                    plot_title = ""
                    ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')                
                    try:
                        plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
                    except:
                        print("Error:", sys.exc_info())
                        print("Error with attempting to save figure, sometimes due to matplotlib LaTeX integration.")
                        print("Usually removing the title should work, but this time even that failed.")                        
                        savechange = -1
        if savechange>0:
            print("Due to rendering error, replaced image title with "+plot_title)
        if savechange>=0:
            print(savefigname+"\n")
    else:
        plt.draw()
        plt.show()
