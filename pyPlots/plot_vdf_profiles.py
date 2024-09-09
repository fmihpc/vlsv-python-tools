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
import pytools as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys, math
import re
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from rotation import rotateVectorToVector,rotateVectorToVector_X

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

# create three 1-dimensional profiles
def doProfiles(f,VX,VY,Voutofslice,slicethick):
    # Select cells which are within slice area
    indexes1 = np.array([(Voutofslice < 0.5*slicethick) & (Voutofslice >= -0.5*slicethick) & (VY < 0.5*slicethick) & (VY >= -0.5*slicethick)])[0]
    indexes2 = np.array([(Voutofslice < 0.5*slicethick) & (Voutofslice >= -0.5*slicethick) & (VX < 0.5*slicethick) & (VX >= -0.5*slicethick)])[0]
    indexes3 = np.array([(VX < 0.5*slicethick) & (VX >= -0.5*slicethick) & (VY < 0.5*slicethick) & (VY >= -0.5*slicethick)])[0]
    
    bins1=bins2=bins3=axis1=axis2=axis3=[]
    if np.any(indexes1):
        inax1 = VX[indexes1]
        range1a = np.amin(inax1)
        range1b = np.amax(inax1)+slicethick
        nbins1 = int((range1b-range1a)/slicethick)
        bins1, axis1 = np.histogram(inax1, nbins1, range=[range1a,range1b], weights=f[indexes1])
        nums1, axis1 = np.histogram(inax1, nbins1, range=[range1a,range1b])
        nonzero = np.where(nums1 != 0)
        bins1[nonzero] = np.divide(bins1[nonzero],nums1[nonzero])

    if np.any(indexes2):
        inax2 = VY[indexes2]
        range2a = np.amin(inax2)
        range2b = np.amax(inax2)+slicethick
        nbins2 = int((range2b-range2a)/slicethick)
        bins2, axis2 = np.histogram(inax2, nbins2, range=[range2a,range2b], weights=f[indexes2])
        nums2, axis2 = np.histogram(inax2, nbins2, range=[range2a,range2b])
        nonzero = np.where(nums2 != 0)
        bins2[nonzero] = np.divide(bins2[nonzero],nums2[nonzero])

    if np.any(indexes3):
        inax3 = Voutofslice[indexes3]
        range3a = np.amin(inax3)
        range3b = np.amax(inax3)+slicethick
        nbins3 = int((range3b-range3a)/slicethick)
        bins3, axis3 = np.histogram(inax3, nbins3, range=[range3a,range3b], weights=f[indexes3])
        nums3, axis3 = np.histogram(inax3, nbins3, range=[range3a,range3b])
        nonzero = np.where(nums3 != 0)
        bins3[nonzero] = np.divide(bins3[nonzero],nums3[nonzero])

    return (bins1,bins2,bins3,axis1[:-1],axis2[:-1],axis3[:-1])

# analyze velocity space in a spatial cell (velocity space reducer)
def vSpaceReducer(vlsvReader, cid, slicetype, normvect, pop="proton", 
                  center=None, setThreshold=None,normvectX=None):
    # check if velocity space exists in this cell
    if vlsvReader.check_variable('fSaved'): #restart files will not have this value
        if vlsvReader.read_variable('fSaved',cid) != 1.0:
            return (False,0,0,0)
    if vlsvReader.check_variable('vg_f_saved'): #restart files will not have this value        
        if vlsvReader.read_variable('vg_f_saved',cid) != 1.0:
            return (False,0,0,0)

    # Assume velocity cells are cubes
    [vxsize, vysize, vzsize] = vlsvReader.get_velocity_mesh_size(pop=pop)
    vxsize = int(vxsize)
    vysize = int(vysize)
    vzsize = int(vzsize)
    # Account for WID3 cells per block
    widval=4 #default WID=4
    if vlsvReader.check_parameter("velocity_block_width"):
        widval = vlsvReader.read_parameter("velocity_block_width")
    vxsize = widval*vxsize
    vysize = widval*vysize
    vzsize = widval*vzsize
    [vxmin, vymin, vzmin, vxmax, vymax, vzmax] = vlsvReader.get_velocity_mesh_extent(pop=pop)
    inputcellsize=(vxmax-vxmin)/vxsize
    logging.info("Input velocity grid cell size "+str(inputcellsize))

    velcells = vlsvReader.read_velocity_cells(cid, pop=pop)
    velcellslist = list(zip(*velcells.items()))
    
    # check that velocity space has cells
    if(len(velcellslist) <= 0):
        return (False,0,0,0)
    
    f = np.asarray(velcellslist[1])
    V = vlsvReader.get_velocity_cell_coordinates(velcellslist[0], pop=pop)
    logging.info("Found "+str(len(V))+" v-space cells")

    # center on highest f-value
    if center == "peak":
        peakindex = np.argmax(f)
        Vpeak = V[peakindex,:]
        V = V - Vpeak
        logging.info(peakindex)
        logging.info("Transforming to frame of peak f-value, travelling at speed "+str(Vpeak))
    elif not center is None:
        if len(center)==3: # assumes it's a vector
            logging.info("Transforming to frame travelling at speed "+str(center))
            V = V - center
        else:
            logging.info("Error in shape of center vector! Give in form (vx,vy,vz).")

    if setThreshold==None:
        # Drop all velocity cells which are below the sparsity threshold. Otherwise the plot will show buffer
        # cells as well.
        if vlsvReader.check_variable('MinValue') == True: # Sparsity threshold used to be saved as MinValue
            setThreshold = vlsvReader.read_variable('MinValue',cid)
            logging.info("Found a vlsv file MinValue of "+str(setThreshold))
        elif vlsvReader.check_variable(pop+"/EffectiveSparsityThreshold") == True:
            setThreshold = vlsvReader.read_variable(pop+"/EffectiveSparsityThreshold",cid)
            logging.info("Found a vlsv file value "+pop+"/EffectiveSparsityThreshold"+" of "+str(setThreshold))
        elif vlsvReader.check_variable(pop+"/vg_effectivesparsitythreshold") == True:
            setThreshold = vlsvReader.read_variable(pop+"/vg_effectivesparsitythreshold",cid)
            logging.info("Found a vlsv file value "+pop+"/vg_effectivesparsitythreshold"+" of "+str(setThreshold))
        else:
            logging.info("Warning! Unable to find a MinValue or EffectiveSparsityThreshold value from the .vlsv file.")
            logging.info("Using a default value of 1.e-16. Override with setThreshold=value.")
            setThreshold = 1.e-16
    ii_f = np.where(f >= setThreshold)
    logging.info("Dropping velocity cells under setThreshold value "+str(setThreshold))
    if len(ii_f) < 1:
        return (False,0,0,0)
    f = f[ii_f]
    V = V[ii_f,:][0,:,:]

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
        Voutofslice = -V[:,1]
    elif slicetype=="vecperp":
        N = np.array(normvect)/np.sqrt(normvect[0]**2 + normvect[1]**2 + normvect[2]**2)
        Vrot = rotateVectorToVector(V,N) # aligns the Z axis of V with normvect
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
        elif slicetype=="Bpara1":
            VX = Vrot2[:,2] # the X axis of the slice is B
            VY = Vrot2[:,0] # the Y axis of the slice is BcrossV=perp1
            Voutofslice = Vrot2[:,1] # the Z axis of the slice is Bcross(BcrossV)=perp2

        # Calculations for verification of rotation:
        testvectors = np.array([N,NX,np.cross(N,NX)]) # verifies B, BcrossV, and Bcross(BcrossV)
        testrot = rotateVectorToVector(testvectors,N) # transforms testvectors to frame where z is aligned with N=B
        testrot2 = rotateVectorToVector_X(testrot,NXrot) # transforms testrot to frame where x is aligned with NXrot (hence preserves z)
        if abs(1.0-np.linalg.norm(NXrot))>1.e-3:
            logging.info("Error in rotation: NXrot not a unit vector")
        if abs(NXrot[2]) > 1.e-3:
            logging.info("Error in rotation: NXrot not in x-y-plane")
        for count,testvect in enumerate(testrot2):
            if abs(1.0-np.linalg.norm(testvect))>1.e-3:
                logging.info("Error in rotation: testvector " + str((count,testvect)) + " not a unit vector")
            if abs(1.0-np.amax(testvect))>1.e-3:
                logging.info("Error in rotation: testvector " + str((count,testvect)) + " largest component is not unity")

    else:
        logging.info("Error finding rotation of v-space!")
        return (False,0,0,0)

    # create three 1-dimensional profiles across VDF
    (bins1,bins2,bins3,axis1,axis2,axis3) = doProfiles(f,VX,VY,Voutofslice,slicethick)
    return (True,bins1,bins2,bins3,axis1,axis2,axis3)


def plot_vdf_profiles(filename=None,
             vlsvobj=None,
             filedir=None, step=None,
             cellids=None, pop="proton",
             coordinates=None, coordre=None, 
             outputdir=None, outputfile=None,
             nooverwrite=None,
             draw=None,axisunit=None,title=None,
             tickinterval=None,
             lin=None,
             run=None, thick=1.0,
             wmark=None, wmarkb=None, 
             fmin=None, fmax=None,
             vmin=None, vmax=None,
             xy=None, xz=None, yz=None, normal=None,
             bpara=None, bpara1=None, bperp=None,
             cbulk=None, cpeak=None, center=None, setThreshold=None,
             axes=None
             ):

    ''' Plots vdf values along axis-aligned lines (see axis definitions).

    :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj:     Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename
    :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/ or override with PTOUTPUTDIR)
                        If directory does not exist, it will be created. If the string does not end in a
                        forward slash, the final parti will be used as a perfix for the files.
    :kword nooverwrite: Set to only perform actions if the target output file does not yet exist                    
     
    :kword cellids:     LIST of cell IDs to plot VDF for
    :kword coordinates: LIST of 3-element spatial coordinate lusts to plot VDF for (given in metres)
    :kword coordre:     LIST of 3-element spatial coordinate lists to plot VDF for (given in Earth radii)
    :kword pop:         Population to plot, default proton

    :kword run:         run identifier, used for constructing output filename
    :kword title:       string to use as plot title instead of time.
                        Special case: Set to "msec" to plot time with millisecond accuracy or "musec"
                        for microsecond accuracy. "sec" is integer second accuracy.
    :kword fmin,fmax:   min and max values for colour scale and colour bar. If no values are given,
                        min and max values for whole plot are used.
    :kword vmin,vmax:   min and max values for x-axis

    :kword axisunit:    Plot v-axes using 10^{axisunit} m/s (default: km/s)
    :kword tickinterval: Interval at which to have ticks on axes
    :kword lin:         Plot using linear y-axis (default log)
   
    :kword xy:          Perform slice in x-y-direction
    :kword xz:          Perform slice in x-z-direction
    :kword yz:          Perform slice in y-z-direction
    :kword normal:      Perform slice in plane perpendicular to given vector
    :kword bpara:       Perform slice in B_para / B_perp2 plane
    :kword bpara1:       Perform slice in B_para / B_perp1 plane
    :kword bperp:       Perform slice in B_perp1 / B_perp2 plane
                        If no plane is given, default is simulation plane (for 2D simulations)

    :kword cbulk:       Center plot on position of total bulk velocity (or if not available,
                        bulk velocity for this population)
    :kword cpeak:       Center plot on velocity with highest phase-space density
    :kword center:      Center plot on provided 3-element velocity vector position (in m/s)
                        If set instead to "bulk" will center on bulk velocity
                        If set instead to "peak" will center on velocity with highest phase-space density
    :kword setThreshold: Use given setThreshold value instead of EffectiveSparsityThreshold or MinValue value read from file
                        Useful if EffectiveSparsityThreshold wasn't saved, or user wants to draw buffer cells
                        with values below the sparsity threshold

    :kword wmark:       If set to non-zero, will plot a Vlasiator watermark in the top left corner. If set to a text
                        string, tries to use that as the location, e.g. "NW","NE","SW","SW"
    :kword wmarkb:      As for wmark, but uses an all-black Vlasiator logo.

    :kword draw:        Draw image on-screen instead of saving to file (requires x-windowing)
    :kword axes:        Provide the routine a set of axes to draw within instead of generating a new image.

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
    if filename!=None:
        vlsvReader=pt.vlsvfile.VlsvReader(filename)
    elif vlsvobj!=None:
        vlsvReader=vlsvobj
    elif ((filedir!=None) and (step!=None)):
        filename = glob.glob(filedir+'bulk*'+str(step).rjust(7,'0')+'.vlsv')[0]
        #filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
        vlsvReader=pt.vlsvfile.VlsvReader(filename)
    else:
        logging.info("Error, needs a .vlsv file name, python object, or directory and step")
        return

    scale=1
    fontsize=8*scale # Most text
    fontsize2=10*scale # Time title
    fontsize3=5*scale # Colour bar ticks
    fontsize4=12*scale # Big label

    # Plot title with time
    timeval=vlsvReader.read_parameter("time")

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

    if draw==None and axes==None:
        # step, used for file name
        if step!=None:
            stepstr = '_'+str(step).rjust(7,'0')
        else:
            if timeval != None:
                stepstr = '_t'+str(int(timeval))
            else:
                stepstr = ''

        # If run name isn't given, just put "plot" in the output file name
        if run==None:
            run='plot'
            # If working within CSC filesystem, make a guess:
            if filename!=None:
                if type(filename) is str:
                    if filename[0:16]=="/proj/vlasov/2D/":
                        run = filename[16:19]
        
        # Verify directory
        if outputfile==None:
            if outputdir==None: # default initial path
                savefigdir=pt.plot.defaultoutputdir
            else:
                savefigdir=outputdir
            # Sub-directories can still be defined in the "run" variable
            savefigname = savefigdir+run
        else: 
            if outputdir!=None:
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
            logging.info("No write access for directory "+savefigdir+"! Exiting.")
            return



    # If population isn't defined i.e. defaults to protons, check if 
    # instead should use old version "avgs"
    if pop=="proton":
       if not vlsvReader.check_population(pop):
           if vlsvReader.check_population("avgs"):
               pop="avgs"
               #logging.info("Auto-switched to population avgs")
           else:
               logging.info("Unable to detect population "+pop+" in .vlsv file!")
               sys.exit()
    else:
        if not vlsvReader.check_population(pop):
            logging.info("Unable to detect population "+pop+" in .vlsv file!")
            sys.exit()       

    #read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = vlsvReader.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    [vxsize, vysize, vzsize] = vlsvReader.get_velocity_mesh_size(pop=pop)
    vxsize = int(vxsize)
    vysize = int(vysize)
    vzsize = int(vzsize)
    [vxmin, vymin, vzmin, vxmax, vymax, vzmax] = vlsvReader.get_velocity_mesh_extent(pop=pop)
    inputcellsize=(vxmax-vxmin)/vxsize

    # Account for WID3 cells per block
    widval=4 #default WID=4
    if vlsvReader.check_parameter("velocity_block_width"):
        widval = vlsvReader.read_parameter("velocity_block_width")
    vxsize = widval*vxsize
    vysize = widval*vysize
    vzsize = widval*vzsize

    Re = 6.371e+6 # Earth radius in m
    # unit of velocity
    velUnit = 1e3
    velUnitStr = r'[km s$^{-1}$]'
    if axisunit!=None:
        velUnit = np.power(10,int(axisunit))
        if np.isclose(axisunit,0):
            velUnitStr = r'[m s$^{-1}$]'
        else:
            velUnitStr = r'[$10^{'+str(int(axisunit))+'}$ m s$^{-1}$]'

    # Select ploitting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if axes is None: # If axes are provided, leave backend as-is.
        if draw is not None:
            if str(matplotlib.get_backend()) is not pt.backend_interactive: #'TkAgg': 
                plt.switch_backend(pt.backend_interactive)
        else:
            if str(matplotlib.get_backend()) is not pt.backend_noninteractive: #'Agg':
                plt.switch_backend(pt.backend_noninteractive)  

    if (cellids==None and coordinates==None and coordre==None):
        logging.info("Error: must provide either cell id's or coordinates")
        return -1

    if coordre!=None:
        # Transform to metres
        coordinates = (Re*np.asarray(coordre)).tolist()

    if coordinates!=None:
        # Check if coordinates given were actually just a single 3-element list
        # instead of a list of 3-element lists:
        if type(coordinates[0]) is not list:
            coordinates = [coordinates]

        # Calculate cell IDs from given coordinates        
        xReq = np.asarray(coordinates).T[0]
        yReq = np.asarray(coordinates).T[1]
        zReq = np.asarray(coordinates).T[2]
        if xReq.shape == yReq.shape == zReq.shape:
            #logging.info('Number of points: ' + str(xReq.shape[0]))
            pass
        else:
            logging.info('ERROR: bad coordinate variables given')
            sys.exit()
        cidsTemp = []
        for ii in range(xReq.shape[0]):
            cidRequest = (np.int64)(vlsvReader.get_cellid(np.array([xReq[ii],yReq[ii],zReq[ii]])))
            cidNearestVspace = -1
            if cidRequest > 0:
                cidNearestVspace = vlsvReader.get_cellid_with_vdf(np.array([xReq[ii],yReq[ii],zReq[ii]]), pop = pop)   # deprecated getNearestCellWithVspace(). needs testing
            else:
                logging.info('ERROR: cell not found')
                sys.exit()
            if (cidNearestVspace <= 0):
                logging.info('ERROR: cell with vspace not found')
                sys.exit()
            xCid,yCid,zCid = vlsvReader.get_cell_coordinates(cidRequest)
            xVCid,yVCid,zVCid = vlsvReader.get_cell_coordinates(cidNearestVspace)
            logging.info('Point: ' + str(ii+1) + '/' + str(xReq.shape[0]))
            logging.info('Requested coordinates : ' + str(xReq[ii]/Re) + ', ' + str(yReq[ii]/Re) + ', ' + str(zReq[ii]/Re))
            logging.info('Nearest spatial cell  : ' + str(xCid/Re)    + ', ' + str(yCid/Re)    + ', ' + str(zCid/Re))
            logging.info('Nearest vspace        : ' + str(xVCid/Re)   + ', ' + str(yVCid/Re)   + ', ' + str(zVCid/Re))
            cidsTemp.append(cidNearestVspace)
        cellids = np.unique(cidsTemp).tolist()
        logging.info('Unique cells with vspace found: ' + str(len(cidsTemp)))
    #else:
    #    logging.info('Using given cell ids and assuming vspace is stored in them')

    # Ensure that we now have a list of cellids instead of just a single cellid
    if type(cellids) is not list:
        logging.info("Converting given cellid to a single-element list of cellids.")
        cellids = [cellids]

    if coordinates==None and coordre==None:
        # User-provided cellids
        for cellid in cellids:
            if not verifyCellWithVspace(vlsvReader, cellid):
                logging.info("Error, cellid "+str(cellid)+" does not contain a VDF!")
                return


    if draw!=None or axes!=None:
        # Program was requested to draw to screen or existing axes instead of saving to a file. 
        # Just handle the first cellid.
        if len(cellids) > 1:
            cellids = [cellids[0]]
            logging.info("User requested on-screen display, only plotting first requested cellid!")

    logging.info("\n")
    for cellid in cellids:
        # Initialise some values
        fminuse=None
        fmaxuse=None

        x,y,z = vlsvReader.get_cell_coordinates(cellid)
        logging.info('cellid ' + str(cellid) + ', x = ' + str(x) + ', y = ' + str(y)  + ', z = ' + str(z))

        # Extracts Vbulk (used in case (i) slice in B-frame and/or (ii) cbulk is neither None nor a string
        Vbulk=None
        if vlsvReader.check_variable('moments'):
            # This should be a restart file
            Vbulk = vlsvReader.read_variable('restart_V',cellid)
        elif vlsvReader.check_variable('vg_moments'):
            # This should be a restart file
            Vbulk = vlsvReader.read_variable('restart_V',cellid)
        elif vlsvReader.check_variable(pop+'/vg_v'):
            # multipop bulk file
            Vbulk = vlsvReader.read_variable(pop+'/vg_v',cellid)
        elif vlsvReader.check_variable(pop+'/V'):
            # multipop bulk file
            Vbulk = vlsvReader.read_variable(pop+'/V',cellid)
        elif vlsvReader.check_variable('V'):
            # regular bulk file, currently analysator supports pre- and post-multipop files with "V"
            Vbulk = vlsvReader.read_variable('V',cellid)
        # if Vbulk is None:
        #     logging.info("Error in finding plasma bulk velocity!")
        #     sys.exit()

        # If necessary, find magnetic field
        if bpara!=None or bperp!=None or bpara1!=None:
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
                if vlsvReader.check_variable("vg_b"):
                    Braw = vlsvReader.read_variable("vg_b", cellidlist)
                elif vlsvReader.check_variable("fg_b"):
                    Braw = vlsvReader.read_variable("fg_b", cellidlist)
                elif vlsvReader.check_variable("B"):
                    Braw = vlsvReader.read_variable("B", cellidlist)
                elif (vlsvReader.check_variable("background_B") and vlsvReader.check_variable("perturbed_B")):
                    # used e.g. for restart files
                    BGB = vlsvReader.read_variable("background_B", cellidlist)
                    PERBB = vlsvReader.read_variable("perturbed_B", cellidlist)
                    Braw = BGB+PERBB
                else:
                    logging.info("Error finding B vector direction!")
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
        if xy==None and xz==None and yz==None and normal==None and bpara==None and bpara1==None and bperp==None:
            # Use default slice for this simulation
            # Check if ecliptic or polar run
            if ysize==1: # polar
                xz=1
                slicetype="xz"
                pltxstr=r"$v_x$"
                pltystr=r"$v_z$"
                pltzstr=r"$v_y$"
                normvect=[0,1,0] # used just for cell size normalisation
            elif zsize==1: # ecliptic
                xy=1
                slicetype="xy"
                pltxstr=r"$v_x$"
                pltystr=r"$v_y$"
                pltzstr=r"$v_z$"
                normvect=[0,0,1] # used just for cell size normalisation
            else:
                logging.info("Problem finding default slice direction")
                yz=1
                slicetype="yz"
                pltxstr=r"$v_y$"
                pltystr=r"$v_z$"
                pltzstr=r"$v_x$"
                normvect=[1,0,0] # used just for cell size normalisation
        elif normal!=None:
            if len(normal)==3:
                slicetype="vecperp"
                normvect=normal
                pltxstr=r"$v_1$"
                pltystr=r"$v_2$"
                pltzstr=r"$v_3$"
            else:
                logging.info("Error parsing slice normal vector!")
                sys.exit()
        elif xy!=None:
            slicetype="xy"
            pltxstr=r"$v_x$"
            pltystr=r"$v_y$"
            pltzstr=r"$v_z$"
            normvect=[0,0,1] # used just for cell size normalisation
        elif xz!=None:
            slicetype="xz"
            pltxstr=r"$v_x$"
            pltystr=r"$v_z$"
            pltzstr=r"$v_y$"
            normvect=[0,1,0] # used just for cell size normalisation
        elif yz!=None:
            slicetype="yz"
            pltxstr=r"$v_y$"
            pltystr=r"$v_z$"
            pltzstr=r"$v_x$"
            normvect=[1,0,0] # used just for cell size normalisation
        elif bpara!=None or bpara1!=None or bperp!=None:
            if Bvect.shape==(1,3):
                Bvect = Bvect[0]
            normvect = Bvect

            # Calculates BcrossV
            BcrossV = np.cross(Bvect,Vbulk)
            normvectX = BcrossV

            if bperp!=None:
                # slice in b_perp1/b_perp2
                slicetype="Bperp"
                #pltxstr=r"$v_{\perp 1}$"
                #pltystr=r"$v_{\perp 2}$"
                pltxstr=r"$v_{B \times V}$"
                pltystr=r"$v_{B \times (B \times V)}$"
                pltzstr=r"$v_{B}$"
            elif bpara1!=None:
                # slice in b_parallel/b_perp1 plane
                slicetype="Bpara1"
                #pltxstr=r"$v_{\parallel}$"
                #pltystr=r"$v_{\perp 1}$"
                pltxstr=r"$v_{B}$"
                pltystr=r"$v_{B \times V}$"
                pltzstr=r"$v_{B \times (B \times V)}$"
            else:
                # slice in b_parallel/b_perp2 plane
                slicetype="Bpara"
                #pltxstr=r"$v_{\parallel}$"
                #pltystr=r"$v_{\perp 2}$"
                pltxstr=r"$v_{B}$"
                pltystr=r"$v_{B \times (B \times V)}$"
                pltzstr=r"$v_{B \times V}$"

        if draw==None and axes==None:
            if outputfile==None:
                savefigname=savefigdir+savefigprefix+"_vdf_"+pop+"_cellid_"+str(cellid)+stepstr+"_"+slicetype+".png"
            else:
                savefigname=outputfile
            # Check if target file already exists and overwriting is disabled
            if (nooverwrite!=None and os.path.exists(savefigname)):            
                if os.stat(savefigname).st_size > 0: # Also check that file is not empty
                    logging.info("Found existing file "+savefigname+". Skipping.")
                    return
                else:
                    logging.info("Found existing file "+savefigname+" of size zero. Re-rendering.")

        # Extend velocity space and each cell to account for slice directions oblique to axes
        normvect = np.array(normvect)
        normvect = normvect/np.linalg.norm(normvect)
        if normvectX is not None:
            normvectX = np.array(normvectX)
            normvectX = normvectX/np.linalg.norm(normvectX)


        if cpeak!=None:
            center='peak'
        if cbulk!=None or center == 'bulk':
            center=None # Finds the bulk velocity and places it in the center vector
            logging.info("Transforming to plasma frame")
            if type(cbulk) is str:
                if vlsvReader.check_variable(cbulk):
                    center = vlsvReader.read_variable(cbulk,cellid)
                    logging.info("Found bulk frame from variable "+cbulk)
            else:
                center = Vbulk

        # Read velocity data into profiles
                
        (checkOk,bins1,bins2,bins3,axis1,axis2,axis3)= vSpaceReducer(vlsvReader, cellid, slicetype, normvect, pop=pop, 
                                                       center=center,setThreshold=setThreshold,normvectX=normvectX)

        # Check that data is ok and not empty
        if checkOk == False:
            logging.info('ERROR: error from velocity space reducer. No velocity cells?')
            continue

        # If no other plotting fmin fmax values are given, take min and max of array
        if fmin!=None:
            fminuse=fmin
        else:
            #fminuse=np.nanmin(np.concatenate((bins1,np.concatenate((bins2,bins3)))))
            fminuse=np.nanmin(np.concatenate((bins1,bins2,bins3)))*0.8
        if fmax!=None:
            fmaxuse=fmax
        else:
            #fmaxuse=np.nanmax(np.concatenate((bins1,np.concatenate((bins2,bins3)))))
            fmaxuse=np.nanmax(np.concatenate((bins1,bins2,bins3)))*1.5
        logging.info("Active f range is "+str(fminuse)+" to "+str(fmaxuse))

        # If no other plotting fmin fmax values are given, take min and max of array
        if vmin!=None:
            vminuse=vmin
        else:
            vminuse=np.nanmin(np.concatenate((axis1,axis2,axis3)))/velUnit
            if vminuse < 0:
                vminuse = vminuse*1.2
            else:
                vminuse = vminuse*0.8
        if vmax!=None:
            vmaxuse=vmax
        else:
            vmaxuse=np.nanmax(np.concatenate((axis1,axis2,axis3)))/velUnit
            if vmaxuse < 0:
                vmaxuse = vmaxuse*0.8
            else:
                vmaxuse = vmaxuse*1.2
        logging.info("Active v range is "+str(vminuse)+" to "+str(vmaxuse))

        axis1=np.array(axis1)/velUnit
        axis2=np.array(axis2)/velUnit
        axis3=np.array(axis3)/velUnit

        # Define figure size        
        #ratio = (yvalsrange[1]-yvalsrange[0])/(xvalsrange[1]-xvalsrange[0])
        # default for square figure is figsize=[4.0,3.15]
        figsize = [4.0,3.15]

        if axes==None:
            # Create 300 dpi image of suitable size
            fig = plt.figure(figsize=figsize,dpi=300)
            ax1 = plt.gca() # get current axes
        else:
            ax1=axes

        if lin is None:
            ax1.set_yscale('log')
        ax1.plot(axis1,bins1, drawstyle='steps-mid', label=pltxstr)
        ax1.plot(axis2,bins2, drawstyle='steps-mid', label=pltystr)
        ax1.plot(axis3,bins3, drawstyle='steps-mid', label=pltzstr)
        ax1.set_ylim([fminuse,fmaxuse])
        ax1.set_xlim([vminuse,vmaxuse])

        
        # ax1.set_xlim([val/velUnit for val in xvalsrange])
        # ax1.set_ylim([val/velUnit for val in yvalsrange])

        # Grid
        #plt.grid(color='grey',linestyle='-')
        #plt.minorticks_on()
        #ax1.grid(color='grey',linestyle='-',lw=thick)
        ax1.tick_params(axis='x',which='minor')
        ax1.tick_params(axis='y',which='minor')

        for axiss in ['top','bottom','left','right']:
            ax1.spines[axiss].set_linewidth(thick)

        ax1.xaxis.set_tick_params(width=thick,length=4*thick)
        ax1.yaxis.set_tick_params(width=thick,length=4*thick)
        ax1.xaxis.set_tick_params(which='minor',width=thick*0.8,length=2*thick)
        ax1.yaxis.set_tick_params(which='minor',width=thick*0.8,length=2*thick)

        if len(plot_title)>0:
            if not os.getenv('PTNOLATEX'):
                plot_title = r"\textbf{"+plot_title+"}"            
            ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')

        handles, labels = ax1.get_legend_handles_labels()
        #ax1.legend(handles, labels, fontsize=fontsize2, handlelength=0.6)
        ax1.legend(handles, labels, fontsize=fontsize2)

        # Find maximum possible lengths of axis tick labels
        # Only counts digits
        # ticklens = [ len(re.sub(r'\D',"",pt.plot.axisfmt(bc,None))) for bc in boxcoords]
        # tickmaxlens = [np.amax(ticklens[0:1]),np.amax(ticklens[2:3])]

        # # Adjust axis tick labels
        # for axisi, axis in enumerate([ax1.xaxis, ax1.yaxis]):
        #     if tickinterval!=None:
        #         axis.set_major_locator(mtick.MultipleLocator(tickinterval))
        #     # Custom tick formatter
        #     axis.set_major_formatter(mtick.FuncFormatter(pt.plot.axisfmt))
        #     ticklabs = axis.get_ticklabels()
        #     # Set boldface.
        #     for t in ticklabs: # note that the tick labels haven't yet been populated with text
        #         t.set_fontweight("black")
        #         # If label has >3 numbers, tilt it
        #         if tickmaxlens[axisi]>3: 
        #             t.set_rotation(30)
        #             t.set_verticalalignment('top')
        #             t.set_horizontalalignment('right')

        if True:
            #plt.xlabel(pltxstr,fontsize=fontsize,weight='black')
            #plt.xticks(fontsize=fontsize,fontweight='black')
            ax1.set_xlabel(velUnitStr,fontsize=fontsize,weight='black')
            for item in ax1.get_xticklabels():
                item.set_fontsize(fontsize)
                item.set_fontweight('black')
            ax1.xaxis.offsetText.set_fontsize(fontsize)
        if True:
            #plt.ylabel(pltystr,fontsize=fontsize,weight='black')
            #plt.yticks(fontsize=fontsize,fontweight='black')
            if not os.getenv('PTNOLATEX'):
                ylabelstr = r"$f(v)\,[\mathrm{m}^{-6} \,\mathrm{s}^{3}]$"
            else:
                ylabelstr = r"$f(v)\,[m^{-6} s^{3}]$"

            ax1.set_ylabel(ylabelstr,fontsize=fontsize,weight='black')
            for item in ax1.get_yticklabels():
                item.set_fontsize(fontsize)
                item.set_fontweight('black')
            ax1.yaxis.offsetText.set_fontsize(fontsize)

        # Adjust layout. Uses tight_layout() but in fact this ensures 
        # that long titles and tick labels are still within the plot area.
        if axes is not None:
            savefig_pad=0.01
            bbox_inches='tight'
        else:
            plt.tight_layout(pad=0.01)
            savefig_pad=0.01
            bbox_inches='tight'

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

        # Save output or draw on-screen
        if draw==None and axes==None:
            try:
                plt.savefig(savefigname,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
            except:
                logging.info("Error with attempting to save figure due to matplotlib LaTeX integration.")
            logging.info(savefigname+"\n")
            plt.close()
        elif axes==None:
            # Draw on-screen
            plt.draw()
            plt.show()
