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
import os, sys, math
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from rotation import rotateVectorToVector,rotateVectorToVector_X

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
    a, b = '{:.1e}'.format(np.amax(abs(np.array(plot_vdf.boxcoords)))).split('e')
    precision = '0'
    if int(b)<1: precision = str(abs(-1-int(b)))
    f = r'{:.'+precision+r'f}'
    a = f.format(abs(x))
    if np.sign(x)<0: a = r'\mbox{\textbf{--}}'+a
    return r'$'+a+'$'


# find nearest spatial cell with vspace to cid
def getNearestCellWithVspace(vlsvReader,cid):
    cell_candidates = vlsvReader.read(mesh='SpatialGrid',tag='CELLSWITHBLOCKS')
    if len(cell_candidates)==0:
        print("Error: No velocity distributions found!")
        sys.exit()
    cell_candidate_coordinates = [vlsvReader.get_cell_coordinates(cell_candidate) for cell_candidate in cell_candidates]
    cell_coordinates = vlsvReader.get_cell_coordinates(cid)
    norms = np.sum((cell_candidate_coordinates - cell_coordinates)**2, axis=-1)**(1./2)
    norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
    return cell_candidates[i]

# Verify that given cell has a saved vspace
def verifyCellWithVspace(vlsvReader,cid):
    cell_candidates = vlsvReader.read(mesh='SpatialGrid',tag='CELLSWITHBLOCKS').tolist()
    # tolist returns a scalar (non-iterable) if array is 0-dim, this variable needs to be an iterable
    if type(cell_candidates) is not list:
        cell_candidates = [cell_candidates]
    found=False
    if cid in cell_candidates:
        found=True
    return found

# create a 2-dimensional histogram
def doHistogram(f,VX,VY,Voutofslice,vxBinEdges,vyBinEdges,vthick,wflux=None):
    # Flux weighting?
    if wflux is not None:
        fw = f*np.linalg.norm([VX,VY,Voutofslice]) # use particle flux as weighting in the histogram
    else:
        fw = f # use particle phase-space density as weighting in the histogram

    # Select cells which are within slice area
    if vthick!=0:
        indexes = [(abs(Voutofslice) <= 0.5*vthick) &
                   (VX > min(vxBinEdges)) & (VX < max(vxBinEdges)) & 
                   (VY > min(vyBinEdges)) & (VY < max(vyBinEdges)) ]
    else:
        indexes = [(VX > min(vxBinEdges)) & (VX < max(vxBinEdges)) & 
                   (VY > min(vyBinEdges)) & (VY < max(vyBinEdges)) ]

    # Gather histogram of values
    (nVhist,VXEdges,VYEdges) = np.histogram2d(VX[tuple(indexes)],VY[tuple(indexes)],bins=(vxBinEdges,vyBinEdges),weights=fw[tuple(indexes)],normed=0)
    # Gather histogram of how many cells were summed for the histogram
    (Chist,VXEdges,VYEdges) = np.histogram2d(VX[tuple(indexes)],VY[tuple(indexes)],bins=(vxBinEdges,vyBinEdges),normed=0)
    # Correct for summing multiple cells into one histogram output cell
    nonzero = np.where(Chist != 0)
    nVhist[nonzero] = np.divide(nVhist[nonzero],Chist[nonzero])

    if vthick==0:
        # slickethick=0, perform a projection. This is done by taking averages for each sampled stack of cells
        # (in order to deal with rotated sampling issues) and then rescaling the resultant 2D VDF with the unsampled
        # VDF in order to get the particle counts to agree. Here we gather the total particle counts for the
        # unprojected VDF.
        weights_total_all = fw[tuple(indexes)].sum()
        weights_total_proj = nVhist.sum()
        # Now rescale the projection back up in order to retain particle counts.
        nVhist = nVhist * weights_total_all/weights_total_proj

    # Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa
    # and y values on the ordinate axis. Rather, x is histogrammed along the first dimension of the array (vertical),
    # and y along the second dimension of the array (horizontal). This ensures compatibility with histogramdd.
    nVhist = nVhist.transpose()

    # Flux weighting
    if wflux is not None:
        dV = np.abs(vxBinEdges[-1] - vxBinEdges[-2]) # assumes constant bin size
        nVhist = np.divide(nVhist,(dV*4*np.pi)) # normalization

    return (nVhist,VXEdges,VYEdges)
  

# analyze velocity space in a spatial cell (velocity space reducer)
def vSpaceReducer(vlsvReader, cid, slicetype, normvect, VXBins, VYBins, pop="proton", 
                  slicethick=None, wflux=None, center=None, setThreshold=None,normvectX=None):
    # check if velocity space exists in this cell
    if vlsvReader.check_variable('fSaved'): #restart files will not have this value        
        if vlsvReader.read_variable('fSaved',cid) != 1.0:
            return (False,0,0,0)
    if vlsvReader.check_variable('vg_f_saved'): #restart files will not have this value        
        if vlsvReader.read_variable('vg_f_saved',cid) != 1.0:
            return (False,0,0,0)
        
    # Assume velocity cells are cubes
    [vxsize, vysize, vzsize] = vlsvReader.get_velocity_mesh_size(pop=pop)
    # Account for 4x4x4 cells per block
    vxsize = 4*vxsize
    vysize = 4*vysize
    vzsize = 4*vzsize
    [vxmin, vymin, vzmin, vxmax, vymax, vzmax] = vlsvReader.get_velocity_mesh_extent(pop=pop)
    inputcellsize=(vxmax-vxmin)/vxsize
    print("Input velocity grid cell size "+str(inputcellsize))

    velcells = vlsvReader.read_velocity_cells(cid, pop=pop)
    velcellslist = list(zip(*velcells.items()))
    
    # check that velocity space has cells
    if(len(velcellslist) <= 0):
        return (False,0,0,0)
    
    f = np.asarray(velcellslist[1])
    V = vlsvReader.get_velocity_cell_coordinates(velcellslist[0], pop=pop)
    print("Found "+str(len(V))+" v-space cells")

    # center on highest f-value
    if center == "peak":
        peakindex = np.argmax(f)
        Vpeak = V[peakindex,:]
        V = V - Vpeak
        print(peakindex)
        print("Transforming to frame of peak f-value, travelling at speed "+str(Vpeak))
    elif not center is None:
        if len(center)==3: # assumes it's a vector
            print("Transforming to frame travelling at speed "+str(center))
            V = V - center
        else:
            print("Error in shape of center vector! Give in form (vx,vy,vz).")

    if setThreshold is None:
        # Drop all velocity cells which are below the sparsity threshold. Otherwise the plot will show buffer
        # cells as well.
        if vlsvReader.check_variable('MinValue') == True: # Sparsity threshold used to be saved as MinValue
            setThreshold = vlsvReader.read_variable('MinValue',cid)
            print("Found a vlsv file MinValue of "+str(setThreshold))
        elif vlsvReader.check_variable(pop+"/EffectiveSparsityThreshold") == True:
            setThreshold = vlsvReader.read_variable(pop+"/EffectiveSparsityThreshold",cid)
            print("Found a vlsv file value "+pop+"/EffectiveSparsityThreshold"+" of "+str(setThreshold))
        elif vlsvReader.check_variable(pop+"/vg_effectivesparsitythreshold") == True:
            setThreshold = vlsvReader.read_variable(pop+"/vg_effectivesparsitythreshold",cid)
            print("Found a vlsv file value "+pop+"/vg_effectivesparsitythreshold"+" of "+str(setThreshold))
        else:
            print("Warning! Unable to find a MinValue or EffectiveSparsityThreshold value from the .vlsv file.")
            print("Using a default value of 1.e-16. Override with setThreshold=value.")
            setThreshold = 1.e-16
    ii_f = np.where(f >= setThreshold)
    print("Dropping velocity cells under setThreshold value "+str(setThreshold))
    if len(ii_f) < 1:
        return (False,0,0,0)
    f = f[ii_f]
    V = V[ii_f,:][0,:,:]

    if slicethick is None:
        # Geometric magic to widen the slice to assure that each cell has some velocity grid points inside it.
        samplebox=np.array([ [0.0,0.0,0.0], [0.0,0.0,1.0], [0.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,0.0], [1.0,0.0,1.0], [1.0,1.0,0.0], [1.0,1.0,1.0] ])
        sbrot = rotateVectorToVector(samplebox,normvect)
        rotminx=np.amin(sbrot[:,0])
        rotmaxx=np.amax(sbrot[:,0])
        rotminy=np.amin(sbrot[:,1])
        rotmaxy=np.amax(sbrot[:,1])
        rotminz=np.amin(sbrot[:,2])
        rotmaxz=np.amax(sbrot[:,2])
        gridratio = np.amax([ rotmaxx-rotminx, rotmaxy-rotminy, rotmaxz-rotminz ])
        if gridratio > 1.0:  # adds a 5% margin to slice thickness
            gridratio = 1.05*gridratio
        slicethick=inputcellsize*gridratio
    else:
        slicethick=inputcellsize*slicethick
    if slicethick!=0:
        print("Performing slice with a counting thickness of "+str(slicethick))
    else:
        print("Projecting total VDF to a single plane")

    if slicetype=="xy":
        VX = V[:,0]
        VY = V[:,1]
        Voutofslice = V[:,2]
    elif slicetype=="yz":
        VX = V[:,1]
        VY = V[:,2]
        Voutofslice = V[:,0]
    elif slicetype=="xz":
        VX = V[:,0]
        VY = V[:,2]
        Voutofslice = V[:,1]
    elif slicetype=="vecperp":
        # Find velocity components in rotated frame where normavect is outofslice and optional
        # normvectX is in VX direction
        N = np.array(normvect)/np.sqrt(normvect[0]**2 + normvect[1]**2 + normvect[2]**2)
        Vrot = rotateVectorToVector(V,N)
        if normvectX is not None:
            NX = np.array(normvectX)/np.sqrt(normvectX[0]**2 + normvectX[1]**2 + normvectX[2]**2)
            NXrot = rotateVectorToVector(NX,N)
            Vrot2 = rotateVectorToVector_X(Vrot,NXrot)
            Vrot = Vrot2
        VX = Vrot[:,0]
        VY = Vrot[:,1]
        Voutofslice = Vrot[:,2]
    elif slicetype=="Bperp" or slicetype=="Bpara" or slicetype=="Bpara1":
        # Find velocity components in rotated frame where B is aligned with Z and BcrossV is aligned with X
        N = np.array(normvect)/np.sqrt(normvect[0]**2 + normvect[1]**2 + normvect[2]**2)
        NX = np.array(normvectX)/np.sqrt(normvectX[0]**2 + normvectX[1]**2 + normvectX[2]**2)
        Vrot = rotateVectorToVector(V,N) # transforms V to frame where z is aligned with N=B
        NXrot = rotateVectorToVector(NX,N) # transforms NX=BcrossV to frame where z is aligned with N=B (hence NXrot in XY plane)
        Vrot2 = rotateVectorToVector_X(Vrot,NXrot) # transforms Vrot to frame where x is aligned with NXrot (hence preserves z)
        # Choose the correct components for this plot
        if slicetype=="Bperp":
            VX = Vrot2[:,0] # the X axis of the slice is BcrossV=perp1
            VY = Vrot2[:,1] # the Y axis of the slice is Bcross(BcrossV)=perp2
            Voutofslice = Vrot2[:,2] # the Z axis of the slice is B
        elif slicetype=="Bpara":
            VX = Vrot2[:,2] # the X axis of the slice is B
            VY = Vrot2[:,1] # the Y axis of the slice is Bcross(BcrossV)=perp2
            Voutofslice = Vrot2[:,0] # the Z axis of the slice is -BcrossV=perp1
            # intuition says this should be -Vrot2[:,0], but testing of profiles across the VDF prove otherwise
        elif slicetype=="Bpara1":
            VX = Vrot2[:,2] # the X axis of the slice is B
            VY = Vrot2[:,0] # the Y axis of the slice is BcrossV=perp1
            Voutofslice = Vrot2[:,1] # the Z axis of the slice is Bcross(BcrossV)=perp2

        # Calculations for verification of rotation:
        testvectors = np.array([N,NX,np.cross(N,NX)]) # verifies B, BcrossV, and Bcross(BcrossV)
        testrot = rotateVectorToVector(testvectors,N) # transforms testvectors to frame where z is aligned with N=B
        testrot2 = rotateVectorToVector_X(testrot,NXrot) # transforms testrot to frame where x is aligned with NXrot (hence preserves z)
        if abs(1.0-np.linalg.norm(NXrot))>1.e-3:
            print("Error in rotation: NXrot not a unit vector")
        if abs(NXrot[2]) > 1.e-3:
            print("Error in rotation: NXrot not in x-y-plane")
        for count,testvect in enumerate(testrot2):
            if abs(1.0-np.linalg.norm(testvect))>1.e-3:
                print("Error in rotation: testvector ",count,testvect," not a unit vector")
            if abs(1.0-np.amax(testvect))>1.e-3:
                print("Error in rotation: testvector ",count,testvect," largest component is not unity")

    else:
        print("Error finding rotation of v-space!")
        return (False,0,0,0)

    # create 2-dimensional histogram of velocity components perpendicular to slice-normal vector
    (binsXY,edgesX,edgesY) = doHistogram(f,VX,VY,Voutofslice,VXBins,VYBins, slicethick, wflux=wflux)
    return (True,binsXY,edgesX,edgesY)


def plot_vdf(filename=None,
             vlsvobj=None,
             filedir=None, step=None,
             cellids=None, pop="proton",
             coordinates=None, coordre=None, 
             outputdir=None, outputfile=None,
             nooverwrite=None,
             draw=None,axisunit=None,title=None, cbtitle=None,
             tickinterval=None,
             colormap=None, box=None, nocb=None, internalcb=None,
             run=None, thick=1.0,
             wmark=None, wmarkb=None, 
             fmin=None, fmax=None, slicethick=None, cellsize=None,
             xy=None, xz=None, yz=None,
             normal=None, normalx=None,
             bpara=None, bpara1=None, bperp=None,
             coordswap=None,
             bvector=None,
             cbulk=None, center=None, wflux=None, setThreshold=None,
             noborder=None, scale=1.0,
             biglabel=None, biglabloc=None,
             noxlabels=None, noylabels=None,
             axes=None, cbaxes=None,
             contours=None
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
     
    :kword cellids:     LIST of cell IDs to plot VDF for
    :kword coordinates: LIST of 3-element spatial coordinate lusts to plot VDF for (given in metres)
    :kword coordre:     LIST of 3-element spatial coordinate lists to plot VDF for (given in Earth radii)
    :kword pop:         Population to plot, default proton

    :kword colormap:    colour scale for plot, use e.g. hot_desaturated, jet, viridis, plasma, inferno,
                        magma, parula, nipy_spectral, RdBu, bwr
    :kword run:         run identifier, used for constructing output filename
    :kword title:       string to use as plot title instead of time.
                        Special case: Set to "msec" to plot time with millisecond accuracy or "musec"
                        for microsecond accuracy. "sec" is integer second accuracy.
    :kword cbtitle:     string to use as colorbar title instead of phase space density of flux

    :kword contours:    Set to number of contours to draw

    :kword fmin,fmax:   min and max values for colour scale and colour bar. If no values are given,
                        min and max values for whole plot are used.

    :kword box:         extents of plotted velocity grid as [x0,x1,y0,y1] (in m/s)
    :kword axisunit:    Plot v-axes using 10^{axisunit} m/s (default: km/s)
    :kword tickinterval: Interval at which to have ticks on axes
   
    :kword xy:          Perform slice in x-y-direction
    :kword xz:          Perform slice in x-z-direction
    :kword yz:          Perform slice in y-z-direction
    :kword normal:      Perform slice in plane perpendicular to given vector
    :kword normalx:     X-axis direction for slice in plane perpendicular to given vector

    :kword bpara:       Perform slice in B_para / B_perp2 plane
    :kword bpara1:       Perform slice in B_para / B_perp1 plane
    :kword bperp:       Perform slice in B_perp1 / B_perp2 plane
                        If no plane is given, default is simulation plane (for 2D simulations)

    :kword coordswap:   Swap the parallel and perpendicular coordinates
    :kword bvector:     Plot a magnetic field vector projection in-plane

    :kword cbulk:       Center plot on position of total bulk velocity (or if not available,
                        bulk velocity for this population)
    :kword center:      Center plot on provided 3-element velocity vector position (in m/s)
                        If set instead to "bulk" will center on bulk velocity
                        If set instead to "peak" will center on velocity with highest phase-space density
    :kword wflux:       Plot flux instead of distribution function
    :kword slicethick:  Thickness of slice as multiplier of cell size (default: 1 or minimum for good coverage).
                        This can be set to zero in order to project the whole VDF to a plane.
    :kword cellsize:    Plotting grid cell size as multiplier of input cell size (default: 1 or minimum for good coverage)
    :kword setThreshold: Use given setThreshold value instead of EffectiveSparsityThreshold or MinValue value read from file
                        Useful if EffectiveSparsityThreshold wasn't saved, or user wants to draw buffer cells
                        with values below the sparsity threshold

    :kword wmark:       If set to non-zero, will plot a Vlasiator watermark in the top left corner. If set to a text
                        string, tries to use that as the location, e.g. "NW","NE","SW","SW"
    :kword wmarkb:      As for wmark, but uses an all-black Vlasiator logo.

    :kword draw:        Draw image on-screen instead of saving to file (requires x-windowing)
    :kword nocb:        Suppress plotting of colourbar legend
    :kword internalcb:  Set to draw colorbar inside plot instead of outside. If set to a text
                        string, tries to use that as the location, e.g. "NW","NE","SW","SW"

    :kword biglabel:    Plot large label (in top-left corner)
    :kword biglabloc:   Move large label to: 0: NW 1: NE 2: SE 3: SW corner

    :kword axes:        Provide the routine a set of axes to draw within instead of generating a new image.
    :kword cbaxes:      Provide the routine a set of axes for the colourbar.

    :kword noborder:    Plot figure edge-to-edge without borders (default off)
    :kword noxlabels:   Suppress x-axis labels and title
    :kword noylabels:   Suppress y-axis labels and title
    :kword scale:       Scale text size (default=1.0)
    :kword thick:       line and axis thickness, default=1.0
    

    :returns:           Outputs an image to a file or to the screen.

    .. code-block:: python

    # Example usage:

    pt.plot.plot_vdf(filename="/proj/vlasov/2D/ABC/distributions/distributions.0000100.vlsv",
                     coordre=[[11.7,-1.6,0.]], cbulk=1, bpara=1,box=[-2e6,2e6,-2e6,2e6],draw=1)

    pt.plot.plot_vdf(filename="/proj/vlasov/2D/ABC/distributions/distributions.0000100.vlsv",
                     cellids=1670561, xy=1, box=[-2e6,2e6,-2e6,2e6], slicethick=5)

    pt.plot.plot_vdf(filename="/proj/vlasov/2D/ABC/distributions/distributions.0000100.vlsv",
                     cellids=[1670561,], xz=1, box=[-2e6,2e6,-2e6,2e6], slicethick=0)


    Note tilted slices: By default, the program samples the V-space with a slice where each cell is cube the
    dimensions of which are found by performing a rotation on a sample square and finding the maximum xyz-extent. This ensures
    adequate coverage and decreases sampling effects. This behaviour can be overridden with the slicethick and cellsize keywords.

    Setting a value of slicethick=0 will result in the whole VDF being flattened into a single plane. This may result in
    some slight sampling artefacts, but is a good measure of complex populations.

    '''

    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    watermarkimageblack=os.path.join(os.path.dirname(__file__), 'logo_black.png')
    
    # Input file or object
    if filename is not None:
        vlsvReader=pt.vlsvfile.VlsvReader(filename)
    elif vlsvobj is not None:
        vlsvReader=vlsvobj
    elif ((filedir is not None) and (step is not None)):
        filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
        vlsvReader=pt.vlsvfile.VlsvReader(filename)
    else:
        print("Error, needs a .vlsv file name, python object, or directory and step")
        return

    if colormap is None:
        colormap="hot_desaturated"
    cmapuse=matplotlib.cm.get_cmap(name=colormap)

    fontsize=8*scale # Most text
    fontsize2=10*scale # Time title
    fontsize3=5*scale # Colour bar ticks
    fontsize4=12*scale # Big label

    # Plot title with time
    timeval=vlsvReader.read_parameter("time")

    # Plot title with time
    if title is None or title=="msec" or title=="musec":        
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

    if draw is None and axes is None:
        # step, used for file name
        if step is not None:
            stepstr = '_'+str(step).rjust(7,'0')
        else:
            if timeval != None:
                stepstr = '_t'+str(np.int(timeval))
            else:
                stepstr = ''

        # If run name isn't given, just put "plot" in the output file name
        if run is None:
            run='plot'
            # If working within CSC filesystem, make a guess:
            if filename is not None:
                if type(filename) is str:
                    if filename[0:16]=="/proj/vlasov/2D/":
                        run = filename[16:19]
        
        # Indicate projection in file name
        projstr=""
        if slicethick==0:
            projstr="_proj"

        # Verify directory
        if outputfile is None:
            if outputdir is None: # default initial path
                savefigdir=os.path.expandvars('$HOME/Plots/')
            else:
                savefigdir=outputdir
            # Sub-directories can still be defined in the "run" variable
            savefigname = savefigdir+run
        else: 
            if outputdir is not None:
                savefigname = outputdir+outputfile
            else:
                savefigname = outputfile
            
        # Re-check to find actual target sub-directory
        savefigprefixind = savefigname.rfind('/')
        if savefigprefixind >= 0:
            savefigdir = savefigname[:savefigprefixind+1]
            savefigprefix = savefigname[savefigprefixind+1:]
        else:
            savefigdir = "./"
            savefigprefix = savefigname

        # Ensure output directory exists
        if not os.path.exists(savefigdir):
            try:
                os.makedirs(savefigdir)
            except:
                pass

        if not os.access(savefigdir, os.W_OK):
            print("No write access for directory "+outputdir2+"! Exiting.")
            return



    # If population isn't defined i.e. defaults to protons, check if 
    # instead should use old version "avgs"
    if pop=="proton":
       if not vlsvReader.check_population(pop):
           if vlsvReader.check_population("avgs"):
               pop="avgs"
               #print("Auto-switched to population avgs")
           else:
               print("Unable to detect population "+pop+" in .vlsv file!")
               sys.exit()
    else:
        if not vlsvReader.check_population(pop):
            print("Unable to detect population "+pop+" in .vlsv file!")
            sys.exit()       

    #read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = vlsvReader.get_spatial_mesh_size()
    [vxsize, vysize, vzsize] = vlsvReader.get_velocity_mesh_size(pop=pop)
    [vxmin, vymin, vzmin, vxmax, vymax, vzmax] = vlsvReader.get_velocity_mesh_extent(pop=pop)
    inputcellsize=(vxmax-vxmin)/vxsize

    # account for 4x4x4 cells per block
    vxsize = 4*vxsize
    vysize = 4*vysize
    vzsize = 4*vzsize

    Re = 6.371e+6 # Earth radius in m
    # unit of velocity
    velUnit = 1e3
    velUnitStr = r'[km s$^{-1}$]'
    if axisunit is not None:
        velUnit = np.power(10,int(axisunit))
        if np.isclose(axisunit,0):
            velUnitStr = r'[m s$^{-1}$]'
        else:
            velUnitStr = r'[$10^{'+str(int(axisunit))+'}$ m s$^{-1}$]'

    # Select ploitting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if axes is None: # If axes are provided, leave backend as-is.
        if draw is not None:
            if str(matplotlib.get_backend()) is not 'TkAgg':
                plt.switch_backend('TkAgg')
        else:
            if str(matplotlib.get_backend()) is not 'Agg':
                plt.switch_backend('Agg')  

    if (cellids is None and coordinates is None and coordre is None):
        print("Error: must provide either cell id's or coordinates")
        return -1

    if coordre is not None:
        # Transform to metres
        coordinates = (Re*np.asarray(coordre)).tolist()

    if coordinates is not None:
        # Check if coordinates given were actually just a single 3-element list
        # instead of a list of 3-element lists:
        if type(coordinates[0]) is not list:
            coordinates = [coordinates]

        # Calculate cell IDs from given coordinates        
        xReq = np.asarray(coordinates).T[0]
        yReq = np.asarray(coordinates).T[1]
        zReq = np.asarray(coordinates).T[2]
        if xReq.shape == yReq.shape == zReq.shape:
            #print('Number of points: ' + str(xReq.shape[0]))
            pass
        else:
            print('ERROR: bad coordinate variables given')
            sys.exit()
        cidsTemp = []
        for ii in range(xReq.shape[0]):
            cidRequest = (np.int64)(vlsvReader.get_cellid(np.array([xReq[ii],yReq[ii],zReq[ii]])))
            cidNearestVspace = -1
            if cidRequest > 0:
                cidNearestVspace = getNearestCellWithVspace(vlsvReader,cidRequest)
            else:
                print('ERROR: cell not found')
                sys.exit()
            if (cidNearestVspace <= 0):
                print('ERROR: cell with vspace not found')
                sys.exit()
            xCid,yCid,zCid = vlsvReader.get_cell_coordinates(cidRequest)
            xVCid,yVCid,zVCid = vlsvReader.get_cell_coordinates(cidNearestVspace)
            print('Point: ' + str(ii+1) + '/' + str(xReq.shape[0]))
            print('Requested coordinates : ' + str(xReq[ii]/Re) + ', ' + str(yReq[ii]/Re) + ', ' + str(zReq[ii]/Re))
            print('Nearest spatial cell  : ' + str(xCid/Re)    + ', ' + str(yCid/Re)    + ', ' + str(zCid/Re))
            print('Nearest vspace        : ' + str(xVCid/Re)   + ', ' + str(yVCid/Re)   + ', ' + str(zVCid/Re))
            cidsTemp.append(cidNearestVspace)
        cellids = np.unique(cidsTemp).tolist()
        print('Unique cells with vspace found: ' + str(len(cidsTemp)))
    #else:
    #    print('Using given cell ids and assuming vspace is stored in them')

    # Ensure that we now have a list of cellids instead of just a single cellid
    if type(cellids) is not list:
        print("Converting given cellid to a single-element list of cellids.")
        cellids = [cellids]

    if coordinates is None and coordre is None:
        # User-provided cellids
        for cellid in cellids:
            if not verifyCellWithVspace(vlsvReader, cellid):
                print("Error, cellid "+str(cellid)+" does not contain a VDF!")
                return


    if draw is not None or axes is not None:
        # Program was requested to draw to screen or existing axes instead of saving to a file. 
        # Just handle the first cellid.
        if len(cellids) > 1:
            cellids = [cellids[0]]
            print("User requested on-screen display, only plotting first requested cellid!")

    print("\n")
    for cellid in cellids:
        # Initialise some values
        fminuse=None
        fmaxuse=None

        x,y,z = vlsvReader.get_cell_coordinates(cellid)
        print('cellid ' + str(cellid) + ', x = ' + str(x) + ', y = ' + str(y)  + ', z = ' + str(z))

        # Extracts Vbulk (used in case (i) slice in B-frame and/or (ii) cbulk is neither None nor a string
        Vbulk=None
        if vlsvReader.check_variable('moments'):
            # This should be a restart file
            Vbulk = vlsvReader.read_variable('restart_V',cellid)
        elif vlsvReader.check_variable(pop+'/vg_v'):
            # multipop v5 bulk file
            Vbulk = vlsvReader.read_variable(pop+'/vg_v',cellid)
        elif vlsvReader.check_variable(pop+'/V'):
            # multipop bulk file
            Vbulk = vlsvReader.read_variable(pop+'/V',cellid)
        elif vlsvReader.check_variable(pop+'/vg_v'):
            # multipop V5 bulk file
            Vbulk = vlsvReader.read_variable(pop+'/vg_v',cellid)
        else:
            # regular bulk file, currently analysator supports pre- and post-multipop files with "V"
            Vbulk = vlsvReader.read_variable('V',cellid)
        if Vbulk is None:
            print("Error in finding plasma bulk velocity!")
            sys.exit()

        # If necessary, find magnetic field
        if bvector is not None or bpara is not None or bperp is not None or bpara1 is not None:
            # First check if volumetric fields are present
            if vlsvReader.check_variable("B_vol"):
                Bvect = vlsvReader.read_variable("B_vol", cellid)
            elif vlsvReader.check_variable("vg_b_vol"):
                Bvect = vlsvReader.read_variable("vg_b_vol", cellid)
            # Otherwise perform linear reconstruction to find
            # approximation of cell-center value
            else:
                # Find dimension of simulation
                if ysize==1 or zsize==1: # 2D
                    cellidlist = [cellid,cellid+1,cellid+xsize]
                else:
                    cellidlist = [cellid,cellid+1,cellid+xsize,cellid+xsize*ysize]
                # Read raw data for the required cells    
                if vlsvReader.check_variable("B"):
                    Braw = vlsvReader.read_variable("B", cellidlist)
                elif (vlsvReader.check_variable("background_B") and vlsvReader.check_variable("perturbed_B")):
                    # used e.g. for restart files
                    BGB = vlsvReader.read_variable("background_B", cellidlist)
                    PERBB = vlsvReader.read_variable("perturbed_B", cellidlist)
                    Braw = BGB+PERBB
                else:
                    print("Error finding B vector direction!")
                # Non-reconstruction version, using just cell-face-values
                # Bvect = Braw[0]
                # Now average in each face direction (not proper reconstruction)
                if ysize==1: #polar
                    Bvect=np.array([0.5*(Braw[0][0]+Braw[1][0]), Braw[0][1], 0.5*(Braw[0][2]+Braw[2][2])])
                elif zsize==1: # ecliptic
                    Bvect=np.array([0.5*(Braw[0][0]+Braw[1][0]), 0.5*(Braw[0][1]+Braw[2][1]), Braw[0][2]])
                else: # 3D, verify this?
                    Bvect=np.array([0.5*(Braw[0][0]+Braw[1][0]), 0.5*(Braw[0][1]+Braw[2][1]), 0.5*(Braw[0][2]+Braw[3][2])])

        # Check slice to perform (and possibly normal vector)
        normvect=None
        normvectX=None
        if xy is None and xz is None and yz is None and normal is None and bpara is None and bpara1 is None and bperp is None:
            # Use default slice for this simulation
            # Check if ecliptic or polar run
            if ysize==1: # polar
                xz=1
                slicetype="xz"
                pltxstr=r"$v_x$ "+velUnitStr
                pltystr=r"$v_z$ "+velUnitStr
                normvect=[0,1,0] # used just for cell size normalisation
            elif zsize==1: # ecliptic
                xy=1
                slicetype="xy"
                pltxstr=r"$v_x$ "+velUnitStr
                pltystr=r"$v_y$ "+velUnitStr
                normvect=[0,0,1] # used just for cell size normalisation
            else:
                print("Problem finding default slice direction")
                yz=1
                slicetype="yz"
                pltxstr=r"$v_y$ "+velUnitStr
                pltystr=r"$v_z$ "+velUnitStr
                normvect=[1,0,0] # used just for cell size normalisation
        elif normal is not None:
            if len(normal)==3:
                slicetype="vecperp"
                normvect=normal
                pltxstr=r"$v_1$ "+velUnitStr
                pltystr=r"$v_2$ "+velUnitStr
            else:
                print("Error parsing slice normal vector!")
                sys.exit()
            if normalx is not None:
                if len(normalx)==3:
                    normvectX=normalx
                    if not np.isclose((np.array(normvect)*np.array(normvectX)).sum(), 0.0):
                        print("Error, normalx dot normal is not zero!")
                        sys.exit()
                else:
                    print("Error parsing slice normalx vector!")
                    sys.exit()
        elif xy is not None:
            slicetype="xy"
            pltxstr=r"$v_x$ "+velUnitStr
            pltystr=r"$v_y$ "+velUnitStr
            normvect=[0,0,1] # used just for cell size normalisation
        elif xz is not None:
            slicetype="xz"
            pltxstr=r"$v_x$ "+velUnitStr
            pltystr=r"$v_z$ "+velUnitStr
            normvect=[0,1,0] # used just for cell size normalisation
        elif yz is not None:
            slicetype="yz"
            pltxstr=r"$v_y$ "+velUnitStr
            pltystr=r"$v_z$ "+velUnitStr
            normvect=[1,0,0] # used just for cell size normalisation
        elif bpara is not None or bpara1 is not None or bperp is not None:
            if Bvect.shape==(1,3):
                Bvect = Bvect[0]
            normvect = Bvect

            # Ensure bulkV has some value
            if np.linalg.norm(Vbulk) < 1e-10:
                Vbulk = [-1,0,0]
                print("Warning, read zero bulk velocity from file. Using VX=-1 for rotation.")
            # Calculates BcrossV
            BcrossV = np.cross(Bvect,Vbulk)
            normvectX = BcrossV
            if bperp is not None:
                # slice in b_perp1/b_perp2
                slicetype="Bperp"
                #pltxstr=r"$v_{\perp 1}$ "+velUnitStr
                #pltystr=r"$v_{\perp 2}$ "+velUnitStr
                pltxstr=r"$v_{B \times V}$ "+velUnitStr
                pltystr=r"$v_{B \times (B \times V)}$ "+velUnitStr
            elif bpara1 is not None:
                # slice in b_parallel/b_perp1 plane
                slicetype="Bpara1"
                #pltxstr=r"$v_{\parallel}$ "+velUnitStr
                #pltystr=r"$v_{\perp 1}$ "+velUnitStr
                pltxstr=r"$v_{B}$ "+velUnitStr
                pltystr=r"$v_{B \times V}$ "+velUnitStr
            else:
                # slice in b_parallel/b_perp2 plane
                slicetype="Bpara"
                #pltxstr=r"$v_{\parallel}$ "+velUnitStr
                #pltystr=r"$v_{\perp 2}$ "+velUnitStr
                pltxstr=r"$v_{B}$ "+velUnitStr
                pltystr=r"$v_{B \times (B \times V)}$ "+velUnitStr


        if draw is None and axes is None:
            if outputfile is None:
                savefigname=savefigdir+savefigprefix+"_vdf_"+pop+"_cellid_"+str(cellid)+stepstr+"_"+slicetype+projstr+".png"
            else:
                savefigname=outputfile
            # Check if target file already exists and overwriting is disabled
            if (nooverwrite is not None and os.path.exists(savefigname)):            
                if os.stat(savefigname).st_size > 0: # Also check that file is not empty
                    print("Found existing file "+savefigname+". Skipping.")
                    return
                else:
                    print("Found existing file "+savefigname+" of size zero. Re-rendering.")

        # Extend velocity space and each cell to account for slice directions oblique to axes
        normvect = np.array(normvect)
        normvect = normvect/np.linalg.norm(normvect)
        if normvectX is not None:
            normvectX = np.array(normvectX)
            normvectX = normvectX/np.linalg.norm(normvectX)


        if cbulk is not None or center=='bulk':
            center=None # Finds the bulk velocity and places it in the center vector
            print("Transforming to plasma frame")
            if type(cbulk) is str:
                if vlsvReader.check_variable(cbulk):
                    center = vlsvReader.read_variable(cbulk,cellid)
                    print("Found bulk frame from variable "+cbulk)
            else:
                center = Vbulk


        # Geometric magic to stretch the grid to assure that each cell has some velocity grid points inside it.
        # Might still be incorrect, erring on the side of caution.
        # norm_srt = sorted(abs(normvect))
        # if cellsize is None:
        #     if norm_srt[1] > 0:
        #         temp = norm_srt[0]/norm_srt[1]
        #         aratio = (1.+temp)/np.sqrt( 1+temp**2)
        #     else:
        #         aratio = 1.
        #     gridratio = aratio * norm_srt[2] * (1. + aratio * np.sqrt(norm_srt[0]**2 + norm_srt[1]**2) / norm_srt[2] )
        #     if gridratio>1.0:
        #         gridratio = gridratio*1.01 # account for numerical inaccuracies
        # else:
        #     gridratio = 1.

        if cellsize is None:
            samplebox=np.array([ [0.0,0.0,0.0], [0.0,0.0,1.0], [0.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,0.0], [1.0,0.0,1.0], [1.0,1.0,0.0], [1.0,1.0,1.0] ])
            sbrot = rotateVectorToVector(samplebox,normvect)
            if normvectX is not None:
                sbrot = rotateVectorToVector_X(sbrot,normvectX)
            rotminx=np.amin(sbrot[:,0])
            rotmaxx=np.amax(sbrot[:,0])
            rotminy=np.amin(sbrot[:,1])
            rotmaxy=np.amax(sbrot[:,1])
            rotminz=np.amin(sbrot[:,2])
            rotmaxz=np.amax(sbrot[:,2])
            gridratio = np.amax([ rotmaxx-rotminx, rotmaxy-rotminy, rotmaxz-rotminz ])
            if gridratio > 1.0:  # adds a 5% margin to slice thickness
                gridratio = 1.05*gridratio
        else:
            gridratio = cellsize

        # num must be vxsize+1 or vysize+1 in order to do both edges for each cell
        VXBins = np.linspace(vxmin*gridratio,vxmax*gridratio,num=vxsize+1)
        VYBins = np.linspace(vymin*gridratio,vymax*gridratio,num=vysize+1)            
        
        # Read velocity data into histogram
        (checkOk,binsXY,edgesX,edgesY) = vSpaceReducer(vlsvReader,cellid,slicetype,normvect,VXBins, VYBins,pop=pop,
                                                       slicethick=slicethick, wflux=wflux,
                                                       center=center,setThreshold=setThreshold,normvectX=normvectX)

        # Check that data is ok and not empty
        if checkOk == False:
            print('ERROR: error from velocity space reducer. No velocity cells?')
            continue

        # Perform swap of coordinate axes, if requested
        if coordswap is not None:
            temp = edgesX
            edgesX = edgesY
            edgesY = temp
            temp = pltxstr
            pltxstr = pltystr
            pltystr = temp
            binsXY = binsXY.T
        # Boldface axis labels
        pltxstr = r'\textbf{'+pltxstr+'}'
        pltystr = r'\textbf{'+pltystr+'}'

        # If no other plotting fmin fmax values are given, take min and max of array
        if fmin is not None:
            fminuse=fmin
        else:
            nzindex = np.where(binsXY > 0)
            if np.any(nzindex):
                fminuse=np.amin(binsXY[nzindex])
            else:
                fminuse = 1e-20 # No valid values! use extreme default.

        if fmax is not None:
            fmaxuse=fmax
        else:
            nzindex = np.where(binsXY > 0)
            if np.any(nzindex):
                fmaxuse=np.amax(binsXY[nzindex])
            else:
                fmaxuse = 1e-10 # No valid values! use extreme default.

        print("Active f range is "+str(fminuse)+" to "+str(fmaxuse))

        norm = LogNorm(vmin=fminuse,vmax=fmaxuse)
        ticks = LogLocator(base=10,subs=list(range(10))) # where to show labels

        if box is not None:  # extents of plotted velocity grid as [x0,y0,x1,y1]
            xvalsrange=[box[0],box[1]]
            yvalsrange=[box[2],box[3]]
        else:
            # Find extent of nonzero data
            xindexrange = [vxsize,0] #last, first cell
            yindexrange = [vysize,0] #last, first cell
            for xi in range(len(edgesX)-1):
                for yi in range(len(edgesY)-1):
#                    if binsXY[xi,yi] > 0:
                    if binsXY[yi,xi] > 0:
                        xindexrange[0] = np.amin([xindexrange[0],xi])
                        xindexrange[1] = np.amax([xindexrange[1],xi])
                        yindexrange[0] = np.amin([yindexrange[0],yi])
                        yindexrange[1] = np.amax([yindexrange[1],yi])

            # leave some buffer
            xindexrange[0] =  np.max([0, 4 * int(np.floor((xindexrange[0]-2.)/4.)) ])
            xindexrange[1] =  np.min([len(edgesX)-1, 4 * int(np.ceil((xindexrange[1]+2.)/4.)) ])
            yindexrange[0] =  np.max([0, 4 * int((np.floor(yindexrange[0]-2.)/4.)) ])
            yindexrange[1] =  np.min([len(edgesY)-1, 4 * int(np.ceil((yindexrange[1]+2.)/4.)) ])

            # If empty VDF: plot whole v-space
            if ((xindexrange==[vxsize,0]) and (yindexrange==[vysize,0])):
                xindexrange = [0,vxsize]
                yindexrange = [0,vysize]

            xvalsrange = [ edgesX[xindexrange[0]] , edgesX[xindexrange[1]] ]
            yvalsrange = [ edgesY[yindexrange[0]] , edgesY[yindexrange[1]] ]

        boxcoords = np.array([xvalsrange[0],xvalsrange[1],yvalsrange[0],yvalsrange[1]])/velUnit
        plot_vdf.boxcoords = boxcoords # make available for axis formatter

            # TODO make plot area square if it's almost square?

        # Define figure size        
        ratio = (yvalsrange[1]-yvalsrange[0])/(xvalsrange[1]-xvalsrange[0])
        # default for square figure is figsize=[4.0,3.15]
        figsize = [4.0,3.15*ratio]

        # Plot the slice         
        [XmeshXY,YmeshXY] = scipy.meshgrid(edgesX/velUnit,edgesY/velUnit) # Generates the mesh to map the data to

        if axes is None:
            # Create 300 dpi image of suitable size
            fig = plt.figure(figsize=figsize,dpi=300)
            ax1 = plt.gca() # get current axes
        else:
            ax1=axes
        fig1 = ax1.pcolormesh(XmeshXY,YmeshXY,binsXY, cmap=colormap,norm=norm)
        ax1.set_xlim([val/velUnit for val in xvalsrange])
        ax1.set_ylim([val/velUnit for val in yvalsrange])
        ax1.set_aspect('equal')

        # Grid
        #plt.grid(color='grey',linestyle='-')
        #plt.minorticks_on()
        ax1.grid(color='grey',linestyle='-',lw=thick)
        ax1.tick_params(axis='x',which='minor')
        ax1.tick_params(axis='y',which='minor')

        # plot contours?
        if contours is not None:
            contdraw=ax1.contour(XmeshXY[:-1,:-1]+0.5*(XmeshXY[1,0]-XmeshXY[0,0]),
                                 YmeshXY[:-1,:-1]+0.5*(YmeshXY[0,1]-YmeshXY[0,0]),
                                 binsXY,np.logspace(math.log10(fminuse),math.log10(fmaxuse),int(contours)),
                                 linewidths=thick*0.5, colors='black')

        for axiss in ['top','bottom','left','right']:
            ax1.spines[axiss].set_linewidth(thick)

        ax1.xaxis.set_tick_params(width=thick,length=4)
        ax1.yaxis.set_tick_params(width=thick,length=4)
        ax1.xaxis.set_tick_params(which='minor',width=thick*0.8,length=2)
        ax1.yaxis.set_tick_params(which='minor',width=thick*0.8,length=2)

        if len(plot_title)>0:
            plot_title = r"\textbf{"+plot_title+"}"            
            ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')

        #fig.canvas.draw() # draw to get tick positions

        # Find maximum possible lengths of axis tick labels
        # Only counts digits
        ticklens = [ len(re.sub(r'\D',"",axisfmt(bc,None))) for bc in boxcoords]
        tickmaxlens = [np.amax(ticklens[0:1]),np.amax(ticklens[2:3])]

        # Adjust axis tick labels
        for axisi, axis in enumerate([ax1.xaxis, ax1.yaxis]):
            if tickinterval is not None:
                axis.set_major_locator(mtick.MultipleLocator(tickinterval))
            # Custom tick formatter
            axis.set_major_formatter(mtick.FuncFormatter(axisfmt))
            ticklabs = axis.get_ticklabels()
            # Set boldface.
            for t in ticklabs: # note that the tick labels haven't yet been populated with text
                t.set_fontweight("black")
                # If label has >3 numbers, tilt it
                if tickmaxlens[axisi]>3: 
                    t.set_rotation(30)
                    t.set_verticalalignment('top')
                    t.set_horizontalalignment('right')

        if noxlabels is None:
            #plt.xlabel(pltxstr,fontsize=fontsize,weight='black')
            #plt.xticks(fontsize=fontsize,fontweight='black')
            ax1.set_xlabel(pltxstr,fontsize=fontsize,weight='black')
            for item in ax1.get_xticklabels():
                item.set_fontsize(fontsize)
                item.set_fontweight('black')
            ax1.xaxis.offsetText.set_fontsize(fontsize)
        if noylabels is None:
            #plt.ylabel(pltystr,fontsize=fontsize,weight='black')
            #plt.yticks(fontsize=fontsize,fontweight='black')
            ax1.set_ylabel(pltystr,fontsize=fontsize,weight='black')
            for item in ax1.get_yticklabels():
                item.set_fontsize(fontsize)
                item.set_fontweight('black')
            ax1.yaxis.offsetText.set_fontsize(fontsize)

        if biglabel is not None:
            if biglabloc is None:
                biglabloc = 0 # default top-left corner               
            if biglabloc==0:
                BLcoords=[0.02,0.98]
                BLha = "left"
                BLva = "top"
            elif biglabloc==1:
                BLcoords=[0.98,0.98]
                BLha = "right"
                BLva = "top"
            elif biglabloc==2:
                BLcoords=[0.98,0.02]
                BLha = "right"
                BLva = "bottom"
            elif biglabloc==3:
                BLcoords=[0.02,0.02]
                BLha = "left"
                BLva = "bottom"

            plt.text(BLcoords[0],BLcoords[1],biglabel, fontsize=fontsize4,weight='black', transform=ax1.transAxes, ha=BLha, va=BLva,color='k',bbox=dict(facecolor='white', alpha=0.5, edgecolor=None))


        if bvector is not None and bpara is None and bperp is None and bpara1 is None:
            # Draw vector of magnetic field direction
            if xy is not None and coordswap is None:
                binplane = [Bvect[0],Bvect[1]]
            if xy is not None and coordswap is not None:
                binplane = [Bvect[1],Bvect[0]]
            if xz is not None and coordswap is None:
                binplane = [Bvect[0],Bvect[2]]
            if xz is not None and coordswap is not None:
                binplane = [Bvect[2],Bvect[0]]
            if yz is not None and coordswap is None:
                binplane = [Bvect[1],Bvect[2]]
            if yz is not None and coordswap is not None:
                binplane = [Bvect[2],Bvect[1]]
            # normalize
            bvector = binplane/np.linalg.norm(binplane)
            # Length default is 1/5 of axis length
            bvectormultiplier = np.amin([yvalsrange[1]-yvalsrange[0],xvalsrange[1]-xvalsrange[0]])/(velUnit*5.)
            bvector *= bvectormultiplier
#            ax1.arrow(0,0,bvector[0],bvector[1],width=0.02*thick,head_width=0.1*thick,
#                      head_length=0.2*thick,zorder=10,color='k')
            #ax1.arrow(origin[0],origin[1],bvector[0],bvector[1],zorder=10,color='k')
            ax1.annotate("", xy=(bvector[0],bvector[1]), xytext=(0,0),
                        arrowprops=dict(arrowstyle="->"),zorder=10)

        if nocb is None:
            cbtitleuse=None
            if cbtitle is not None:
                if type(cbtitle) is str:
                    cbtitleuse = cbtitle
            if cbtitleuse is None:
                if wflux is None:
                    cbtitleuse=r"$f(v)\,[\mathrm{m}^{-6} \,\mathrm{s}^{3}]$"
                else:
                    cbtitleuse=r"flux $F\,[\mathrm{m}^{-2} \,\mathrm{s}^{-1} \,\mathrm{sr}^{-1}]$"

            if cbaxes is not None:
                cax = cbaxes
                cbdir="right"
                horalign="left"
            elif internalcb is None:
                # Witchcraft used to place colourbar
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbdir="right"
                horalign="left"
            else:
                # Colorbar within plot area
                cbloc=1
                cbdir="left"
                horalign="right"
                if type(internalcb) is str:
                    if internalcb=="NW":
                        cbloc=2
                        cbdir="right"
                        horalign="left"
                    if internalcb=="SW": 
                        cbloc=3
                        cbdir="right"
                        horalign="left"
                    if internalcb=="SE": 
                        cbloc=4
                        cbdir="left"
                        horalign="right"
                cax = inset_axes(ax1, width="5%", height="35%", loc=cbloc, 
                                 bbox_transform=ax1.transAxes, borderpad=1.0)
                # borderpad default value is 0.5, need to increase it to make room for colorbar title

            # Colourbar title
            cbtitleuse = r"\textbf{"+cbtitleuse+"}"

            # First draw colorbar
            cb = plt.colorbar(fig1,ticks=ticks,cax=cax)
            cb.outline.set_linewidth(thick)
            cb.ax.yaxis.set_ticks_position(cbdir)
            if cbaxes is None:
                cb.ax.tick_params(labelsize=fontsize3)#,width=1.5,length=3)
                cb_title = cax.set_title(cbtitleuse,fontsize=fontsize3,fontweight='bold', horizontalalignment=horalign)
            else:
                cb.ax.tick_params(labelsize=fontsize)
                cb_title = cax.set_title(cbtitleuse,fontsize=fontsize,fontweight='bold', horizontalalignment=horalign)
            cb_title.set_position((0.,1.+0.025*scale)) # avoids having colourbar title too low when fontsize is increased
                    


        if noxlabels is not None:
            for label in ax1.xaxis.get_ticklabels():
                label.set_visible(False)
        if noylabels is not None:
            for label in ax1.yaxis.get_ticklabels():
                label.set_visible(False)       

        # Adjust layout. Uses tight_layout() but in fact this ensures 
        # that long titles and tick labels are still within the plot area.
        if axes is not None:
            savefig_pad=0.01
            bbox_inches='tight'
        elif noborder is None:
            plt.tight_layout()
            savefig_pad=0.05 # The default is 0.1
            bbox_inches=None
        else:
            plt.tight_layout(pad=0.01)
            savefig_pad=0.01
            bbox_inches='tight'

        # Add Vlasiator watermark
        if (wmark is not None or wmarkb is not None) and axes is None:
            if wmark is not None:
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

        # Save output or draw on-screen
        if draw is None and axes is None:
            try:
                plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
            except:
                print("Error with attempting to save figure due to matplotlib LaTeX integration.")
            print(savefigname+"\n")
        elif axes is None:
            # Draw on-screen
            plt.draw()
            plt.show()
