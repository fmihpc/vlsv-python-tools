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
import warnings
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys, math
import re
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm,Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator,LinearLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from rotation import rotateVectorToVector,rotateVectorToVector_X

from plot_vdf import verifyCellWithVspace, doHistogram, resampleReducer, vSpaceReducer

def plot_vdfdiff(filename1=None, filename2=None,
             vlsvobj1=None, vlsvobj2=None,
             filedir=None, step=None,
             cellids=None, cellids2=None,
             pop="proton",
             coordinates=None, coordre=None,
             outputdir=None, outputfile=None,
             nooverwrite=None,
             draw=None,axisunit=None,axiskmps=None,title=None, cbtitle=None,
             tickinterval=None,
             colormap=None, box=None, nocb=None, internalcb=None,
             run=None, thick=1.0,
             wmark=None, wmarkb=None,
             fmin=None, fmax=None, slicethick=None, reducer='integrate', resampler=False,
             cellsize=None,
             xy=None, xz=None, yz=None,
             normal=None, normalx=None,
             bpara=None, bpara1=None, bperp=None,
             coordswap=None,
             bvector=None,bvectorscale=0.2,
             cbulk=None, center=None, wflux=None, setThreshold=None,
             noborder=None, scale=1.0, scale_text=8.0, scale_title=10.0,scale_cb=5.0,scale_label=12.0,
             biglabel=None, biglabloc=None,
             noxlabels=None, noylabels=None,
             axes=None, cbaxes=None,
             contours=None
             ):

    ''' Plots the difference between VDFs as a coloured 2D plot with axes and a colour bar.

    :kword filename1:   path to .vlsv file to use for input. Assumes a bulk file.
    :kword filename2:   path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj1:    Optionally provide a python vlsvfile object instead
    :kword vlsvobj2:    Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename
    :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/ or override with PTOUTPUTDIR)
                        If directory does not exist, it will be created. If the string does not end in a
                        forward slash, the final parti will be used as a perfix for the files.
    :kword nooverwrite: Set to only perform actions if the target output file does not yet exist

    :kword cellids:     LIST of cell IDs to plot VDF for
    :kword cellids2:    Optional: LIST of cell IDs from input file 2. Same as cellids if omitted.
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
    :kword axiskmps:    Plot v-axes using 10^{axiskmps} km/s (default: km/s, when the kword has a value)
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
    :kword bvectorscale: Scale of bvector (default: 0.2 in units of axis lengths)

    :kword cbulk:       Center plot on position of total bulk velocity (or if not available,
                        bulk velocity for this population)
    :kword center:      Center plot on provided 3-element velocity vector position (in m/s)
                        If set instead to "bulk" will center on bulk velocity
                        If set instead to "peak" will center on velocity with highest phase-space density
    :kword wflux:       Plot flux instead of distribution function
    :kword slicethick:  Thickness of slice as multiplier of cell size (default: 1 or minimum for good coverage).
                        This can be set to zero in order to project the whole VDF to a plane.
    :kword reducer:     How to reduce to 2D - default 'integrate' for LOS integration
                        and reduced units, 'average' for old (slightly questionable) behaviour
    :kword resampler:   Resample onto a regular grid? Default: yes, use False to disable.
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
    :kword scale:       Scale text size everywhere (default=1.0)
    :kword scale_text:  Most text additional scale factor (default=8.0)
    :kword scale_title: Title additional scale factor (default=10.0)
    :kword scale_cb:    Colour bar text additional scale factor (default=5.0)
    :kword scale_label: Big label text additional scale factor (default=12.0)
    :kword thick:       line and axis thickness, default=1.0


    :returns:           Outputs an image to a file or to the screen.

    .. code-block:: python

    # Example usage:

    pt.plot.plot_vdfdiff("bulk.0000040.vlsv","bulk.0000048.vlsv",cellids=1,xy=1,slicethick=1,colormap="seismic",
                     fmin=-1e-9,fmax=1e-9,cbulk=1)


    See documentation of plot_vdf() for more usage information.

    '''

    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    watermarkimageblack=os.path.join(os.path.dirname(__file__), 'logo_black.png')

    # Input file or object
    if filename1 is not None:
        vlsvReader1=pt.vlsvfile.VlsvReader(filename1)
    else:
        logging.info("Error, needs a .vlsv file name")
        return
    if filename2 is not None:
        vlsvReader2=pt.vlsvfile.VlsvReader(filename2)
    else:
        logging.info("Error, needs a .vlsv file name")
        return

    if colormap is None:
        colormap="seismic"
    cmapuse=matplotlib.cm.get_cmap(name=colormap)

    fontsize=scale_text*scale # Most text
    fontsize2=scale_title*scale # Time title
    fontsize3=scale_cb*scale # Colour bar ticks
    fontsize4=scale_label*scale # Big label

    # Plot title with time
    timeval=vlsvReader1.read_parameter("time")

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

    if reducer == "average":
        logging.info("V-space reduction via averages")
        warnings.warn('Average-reduction is kept for backward-compatibility for now; consider using "integrate"!')
        pass
    elif reducer == "integrate":
        logging.info("V-space reduction via integration")
        pass
    else:
        raise ValueError("Unknown reducer ("+reducer+'), accepted values are "average", "integrate"')

    if wflux is not None:
        warnings.warn("Does flux weighting make sense? Tread carefully.")

    if draw is None and axes is None:
        # step, used for file name
        if step is not None:
            stepstr = '_'+str(step).rjust(7,'0')
        else:
            if timeval != None:
                stepstr = '_t'+str(int(timeval))
            else:
                stepstr = ''

        # If run name isn't given, just put "plot" in the output file name
        if run is None:
            run='plot'
            # If working within CSC filesystem, make a guess:
            if filename1 is not None:
                if type(filename1) is str:
                    if filename1[0:16]=="/proj/vlasov/2D/":
                        run = filename1[16:19]

        # Indicate projection in file name
        projstr=""
        if slicethick==0:
            projstr="_proj"

        # Verify directory
        if outputfile is None:
            if outputdir is None: # default initial path
                savefigdir=pt.plot.defaultoutputdir
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
            logging.info("No write access for directory "+savefigdir+"! Exiting.")
            return



    # If population isn't defined i.e. defaults to protons, check if
    # instead should use old version "avgs"
    if pop=="proton":
       if not vlsvReader1.check_population(pop):
           if vlsvReader1.check_population("avgs"):
               pop="avgs"
               #logging.info("Auto-switched to population avgs")
           else:
               logging.info("Unable to detect population "+pop+" in .vlsv file!")
               sys.exit()
    else:
        if not vlsvReader1.check_population(pop):
            logging.info("Unable to detect population "+pop+" in .vlsv file!")
            sys.exit()

    #read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = vlsvReader1.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    [vxsize, vysize, vzsize] = vlsvReader1.get_velocity_mesh_size(pop=pop)
    vxsize = int(vxsize)
    vysize = int(vysize)
    vzsize = int(vzsize)
    [vxmin, vymin, vzmin, vxmax, vymax, vzmax] = vlsvReader1.get_velocity_mesh_extent(pop=pop)
    inputcellsize=(vxmax-vxmin)/vxsize

    # Account for WID3 cells per block
    widval=4 #default WID=4
    if vlsvReader1.check_parameter("velocity_block_width"):
        widval = vlsvReader1.read_parameter("velocity_block_width")
    vxsize = widval*vxsize
    vysize = widval*vysize
    vzsize = widval*vzsize

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
    if axiskmps is not None:
        velUnit = np.power(10,int(axiskmps)+3)
        if np.isclose(axiskmps,0):
            velUnitStr = r'[km s$^{-1}$]'
        else:
            velUnitStr = r'[$10^{'+str(int(axiskmps))+'}$ km s$^{-1}$]'

    # Select plotting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if axes is None: # If axes are provided, leave backend as-is.
        if draw is not None:
            if str(matplotlib.get_backend()) is not pt.backend_interactive: #'TkAgg':
                plt.switch_backend(pt.backend_interactive)
        else:
            if str(matplotlib.get_backend()) is not pt.backend_noninteractive: #'Agg':
                plt.switch_backend(pt.backend_noninteractive)

    if (cellids is None and coordinates is None and coordre is None):
        logging.info("Error: must provide either cell id's or coordinates")
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
            #logging.info('Number of points: ' + str(xReq.shape[0]))
            pass
        else:
            logging.info('ERROR: bad coordinate variables given')
            sys.exit()
        cidsTemp = []
        for ii in range(xReq.shape[0]):
            cidRequest = (np.int64)(vlsvReader1.get_cellid(np.array([xReq[ii],yReq[ii],zReq[ii]])))
            cidNearestVspace = -1
            if cidRequest > 0:
                cidNearestVspace = vlsvReader1.get_cellid_with_vdf(np.array([xReq[ii],yReq[ii],zReq[ii]]), pop=pop)
            else:
                logging.info('ERROR: cell not found')
                sys.exit()
            if (cidNearestVspace <= 0):
                logging.info('ERROR: cell with vspace not found')
                sys.exit()
            xCid,yCid,zCid = vlsvReader1.get_cell_coordinates(cidRequest)
            xVCid,yVCid,zVCid = vlsvReader1.get_cell_coordinates(cidNearestVspace)
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
    if type(cellids2) is not list:
        cellids2 = [cellids2]

    if coordinates is None and coordre is None:
        # User-provided cellids
        for cellid in cellids:
            if not verifyCellWithVspace(vlsvReader1, cellid):
                logging.info("Error, cellid "+str(cellid)+" in input file 1 does not contain a VDF!")
                return
            if not verifyCellWithVspace(vlsvReader2, cellid):
                logging.info("Error, cellid "+str(cellid)+" in input file 2 does not contain a VDF!")
                return


    if draw is not None or axes is not None:
        # Program was requested to draw to screen or existing axes instead of saving to a file.
        # Just handle the first cellid.
        if len(cellids) > 1:
            cellids = [cellids[0]]
            logging.info("User requested on-screen display, only plotting first requested cellid!")

    if cellids2 is None or cellids2[0] is None:
        cellids2 = cellids

    logging.info("\n")
    for cellid,cellid2 in list(map(list,zip(*(cellids,cellids2)))):
      #   logging.info(cellid,cellid2, cellids, cellids2)
        # Initialise some values
        fminuse=None
        fmaxuse=None

        x,y,z = vlsvReader1.get_cell_coordinates(cellid)
        logging.info('cellid ' + str(cellid) + ', x = ' + str(x) + ', y = ' + str(y)  + ', z = ' + str(z))
        x,y,z = vlsvReader2.get_cell_coordinates(cellid2)
        logging.info('cellid2 ' + str(cellid2) + ', x = ' + str(x) + ', y = ' + str(y)  + ', z = ' + str(z))

        # Extracts Vbulk (used in case (i) slice in B-frame and/or (ii) cbulk is neither None nor a string
        Vbulk=None
        Vbulk2=None
        if vlsvReader1.check_variable('moments'):
            # This should be a restart file
            Vbulk = vlsvReader1.read_variable('restart_V',cellid)
            Vbulk2 = vlsvReader2.read_variable('restart_V',cellid2)
        elif vlsvReader1.check_variable(pop+'/vg_v'):
            # multipop v5 bulk file
            Vbulk = vlsvReader1.read_variable(pop+'/vg_v',cellid)
            Vbulk2 = vlsvReader2.read_variable(pop+'/vg_v',cellid2)
        elif vlsvReader1.check_variable(pop+'/V'):
            # multipop bulk file
            Vbulk = vlsvReader1.read_variable(pop+'/V',cellid)
            Vbulk2 = vlsvReader2.read_variable(pop+'/V',cellid2)
        elif vlsvReader1.check_variable(pop+'/vg_v'):
            # multipop V5 bulk file
            Vbulk = vlsvReader1.read_variable(pop+'/vg_v',cellid)
            Vbulk2 = vlsvReader2.read_variable(pop+'/vg_v',cellid2)
        else:
            # regular bulk file, currently analysator supports pre- and post-multipop files with "V"
            Vbulk = vlsvReader1.read_variable('V',cellid)
            Vbulk2 = vlsvReader2.read_variable('V',cellid2)
        if Vbulk is None:
            logging.info("Error in finding plasma bulk velocity!")
            sys.exit()

        # If necessary, find magnetic field
        if bvector is not None or bpara is not None or bperp is not None or bpara1 is not None:
            # First check if volumetric fields are present
            if vlsvReader1.check_variable("B_vol"):
                Bvect = vlsvReader1.read_variable("B_vol", cellid)
                Bvect2 = vlsvReader2.read_variable("B_vol", cellid2)
            elif vlsvReader1.check_variable("vg_b_vol"):
                Bvect = vlsvReader1.read_variable("vg_b_vol", cellid)
                Bvect2 = vlsvReader1.read_variable("vg_b_vol", cellid2)
            # Otherwise perform linear reconstruction to find
            # approximation of cell-center value
            else:
                # Find dimension of simulation
                if ysize==1 or zsize==1: # 2D
                    cellidlist = [cellid,cellid+1,cellid+xsize]
                else:
                    cellidlist = [cellid,cellid+1,cellid+xsize,cellid+xsize*ysize]
                # Read raw data for the required cells
                if vlsvReader1.check_variable("B"):
                    Braw = vlsvReader1.read_variable("B", cellidlist)
                elif (vlsvReader1.check_variable("background_B") and vlsvReader1.check_variable("perturbed_B")):
                    # used e.g. for restart files
                    BGB = vlsvReader1.read_variable("background_B", cellidlist)
                    PERBB = vlsvReader1.read_variable("perturbed_B", cellidlist)
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

                # TODO: I guess we should do this interpolation in the second file too, instead of assuming identical Bvect
                Bvect2 = Bvect

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
                logging.info("Problem finding default slice direction")
                yz=1
                slicetype="yz"
                pltxstr=r"$v_y$ "+velUnitStr
                pltystr=r"$v_z$ "+velUnitStr
                normvect=[1,0,0] # used just for cell size normalisation
            normvect2=normvect
        elif normal is not None:
            if len(normal)==3:
                slicetype="vecperp"
                normvect=normal
                normvect2=normvect
                pltxstr=r"$v_1$ "+velUnitStr
                pltystr=r"$v_2$ "+velUnitStr
            else:
                logging.info("Error parsing slice normal vector!")
                sys.exit()
            if normalx is not None:
                if len(normalx)==3:
                    normvectX=normalx
                    if not np.isclose((np.array(normvect)*np.array(normvectX)).sum(), 0.0):
                        logging.info("Error, normalx dot normal is not zero!")
                        sys.exit()
                else:
                    logging.info("Error parsing slice normalx vector!")
                    sys.exit()
        elif xy is not None:
            slicetype="xy"
            pltxstr=r"$v_x$ "+velUnitStr
            pltystr=r"$v_y$ "+velUnitStr
            normvect=[0,0,1] # used just for cell size normalisation
            normvect2=normvect
        elif xz is not None:
            slicetype="xz"
            pltxstr=r"$v_x$ "+velUnitStr
            pltystr=r"$v_z$ "+velUnitStr
            normvect=[0,1,0] # used just for cell size normalisation
            normvect2=normvect
        elif yz is not None:
            slicetype="yz"
            pltxstr=r"$v_y$ "+velUnitStr
            pltystr=r"$v_z$ "+velUnitStr
            normvect=[1,0,0] # used just for cell size normalisation
            normvect2=normvect
        elif bpara is not None or bpara1 is not None or bperp is not None:
            if Bvect.shape==(1,3):
                Bvect = Bvect[0]
            normvect = Bvect
            normvect2 = Bvect2

            # Ensure bulkV has some value
            if np.linalg.norm(Vbulk) < 1e-10:
                Vbulk = [-1,0,0]
                logging.info("Warning, read zero bulk velocity from file. Using VX=-1 for rotation.")
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
                    logging.info("Found existing file "+savefigname+". Skipping.")
                    return
                else:
                    logging.info("Found existing file "+savefigname+" of size zero. Re-rendering.")

        # Extend velocity space and each cell to account for slice directions oblique to axes
        normvect = np.array(normvect)
        normvect = normvect/np.linalg.norm(normvect)
        normvect2 = np.array(normvect2)
        normvect2 = normvect2/np.linalg.norm(normvect2)
        if normvectX is not None:
            normvectX = np.array(normvectX)
            normvectX = normvectX/np.linalg.norm(normvectX)

        if cbulk is None and center is None:
            center = np.zeros((3,))
            center2 = np.zeros((3,))
        elif cbulk is not None or type(center) is str and center=='bulk':
            center=None # Finds the bulk velocity and places it in the center vector
            logging.info("Transforming to plasma frame")
            if type(cbulk) is str:
                if vlsvReader1.check_variable(cbulk):
                    center = vlsvReader1.read_variable(cbulk,cellid)
                    logging.info("Found bulk frame from variable "+cbulk)
                if vlsvReader2.check_variable(cbulk):
                    center2 = vlsvReader2.read_variable(cbulk,cellid2)
                    logging.info("Found bulk frame from variable "+cbulk)
            else:
                center = Vbulk
                center2 = Vbulk2


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
        (checkOk,binsXY,edgesX,edgesY) = vSpaceReducer(vlsvReader1,cellid,slicetype,normvect,VXBins, VYBins,pop=pop,
                                                       slicethick=slicethick, reducer=reducer, resampler=resampler, wflux=wflux,
                                                       center=center,setThreshold=setThreshold,normvectX=normvectX)

        (checkOk2,binsXY2,edgesX,edgesY) = vSpaceReducer(vlsvReader2,cellid2,slicetype,normvect2,VXBins, VYBins,pop=pop,
                                                       slicethick=slicethick, reducer=reducer, resampler=resampler, wflux=wflux,
                                                       center=center2,setThreshold=setThreshold,normvectX=normvectX)

        # Calc difference
        binsXY-=binsXY2;

        # Check that data is ok and not empty
        if checkOk == False or checkOk2 == False:
            logging.info('ERROR: error from velocity space reducer. No velocity cells?')
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
        pltxstr = pt.plot.textbfstring(pltxstr)
        pltystr = pt.plot.textbfstring(pltystr)

        # If no other plotting fmin fmax values are given, take min and max of array
        if fmin is not None:
            fminuse=fmin
        else:
            nzindex = np.where(binsXY != 0)
            if np.any(nzindex):
                fminuse=np.amin(binsXY[nzindex])
            else:
                fminuse = -1 # No valid values! use extreme default.

        if fmax is not None:
            fmaxuse=fmax
        else:
            nzindex = np.where(binsXY != 0)
            if np.any(nzindex):
                fmaxuse=np.amax(binsXY[nzindex])
            else:
                fmaxuse = 1 # No valid values! use extreme default.

        # If neither min or max have been set by hand, make plot extents symmetric
        if fmax is None and fmin is None:
            fextreme = max(np.fabs(fminuse),np.fabs(fmaxuse))
            fminuse = -fextreme
            fmaxuse = fextreme

        logging.info("Active f range is "+str(fminuse)+" to "+str(fmaxuse))
        norm = Normalize(vmin=fminuse,vmax=fmaxuse)

        ticks = LinearLocator()

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
            xindexrange[0] =  np.max([0, widval * int(np.floor((xindexrange[0]-0.5*widval)/widval)) ])
            xindexrange[1] =  np.min([len(edgesX)-1, widval * int(np.ceil((xindexrange[1]+0.5*widval)/widval)) ])
            yindexrange[0] =  np.max([0, widval * int((np.floor(yindexrange[0]-0.5*widval)/widval)) ])
            yindexrange[1] =  np.min([len(edgesY)-1, widval * int(np.ceil((yindexrange[1]+0.5*widval)/widval)) ])

            # If empty VDF: plot whole v-space
            if ((xindexrange==[vxsize,0]) and (yindexrange==[vysize,0])):
                xindexrange = [0,vxsize]
                yindexrange = [0,vysize]

            xvalsrange = [ edgesX[xindexrange[0]] , edgesX[xindexrange[1]] ]
            yvalsrange = [ edgesY[yindexrange[0]] , edgesY[yindexrange[1]] ]

        boxcoords = np.array([xvalsrange[0],xvalsrange[1],yvalsrange[0],yvalsrange[1]])/velUnit
        # Set required decimal precision
        precision_a, precision_b = '{:.1e}'.format(np.amax(abs(np.array(boxcoords)))).split('e')
        pt.plot.decimalprecision_ax = '0'
        if int(precision_b)<1: pt.plot.decimalprecision_ax = str(abs(-1-int(precision_b)))

        # TODO make plot area square if it's almost square?

        # Define figure size
        ratio = (yvalsrange[1]-yvalsrange[0])/(xvalsrange[1]-xvalsrange[0])
        # default for square figure is figsize=[4.0,3.15]
        figsize = [4.0,3.15*ratio]

        # Plot the slice
        [XmeshXY,YmeshXY] = np.meshgrid(edgesX/velUnit,edgesY/velUnit) # Generates the mesh to map the data to

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
                                 binsXY,fminuse,fmaxuse,int(contours),
                                 linewidths=thick*0.5, colors='black')

        for axiss in ['top','bottom','left','right']:
            ax1.spines[axiss].set_linewidth(thick)

        ax1.xaxis.set_tick_params(width=thick,length=4)
        ax1.yaxis.set_tick_params(width=thick,length=4)
        ax1.xaxis.set_tick_params(which='minor',width=thick*0.8,length=2)
        ax1.yaxis.set_tick_params(which='minor',width=thick*0.8,length=2)

        if len(plot_title)>0:
            plot_title = pt.plot.textbfstring(plot_title)
            ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold')

        #fig.canvas.draw() # draw to get tick positions

        # Find maximum possible lengths of axis tick labels
        # Only counts digits
        ticklens = [ len(re.sub(r'\D',"",pt.plot.axisfmt(bc,None))) for bc in boxcoords]
        tickmaxlens = [np.amax(ticklens[0:1]),np.amax(ticklens[2:3])]

        # Adjust axis tick labels
        for axisi, axis in enumerate([ax1.xaxis, ax1.yaxis]):
            if tickinterval is not None:
                axis.set_major_locator(mtick.MultipleLocator(tickinterval))
            # Custom tick formatter
            axis.set_major_formatter(mtick.FuncFormatter(pt.plot.axisfmt))
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
            ax1.set_xlabel(pltxstr,fontsize=fontsize,weight='black')
            for item in ax1.get_xticklabels():
                item.set_fontsize(fontsize)
                item.set_fontweight('black')
            ax1.xaxis.offsetText.set_fontsize(fontsize)
        if noylabels is None:
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
            bvectormultiplier = np.amin([yvalsrange[1]-yvalsrange[0],xvalsrange[1]-xvalsrange[0]])/(velUnit)
            bvector *= bvectormultiplier*bvectorscale
#            ax1.arrow(0,0,bvector[0],bvector[1],width=0.02*thick,head_width=0.1*thick,
#                      head_length=0.2*thick,zorder=10,color='k')
            #ax1.arrow(origin[0],origin[1],bvector[0],bvector[1],zorder=10,color='k')
            ax1.annotate("", xy=(bvector[0],bvector[1]), xytext=(0,0),
                        arrowprops=dict(arrowstyle="->"),zorder=10)

        if nocb is None:
            cb_title_use=None
            if cbtitle is not None:
                if type(cbtitle) is str:
                    cb_title_use = cbtitle
            if cb_title_use is None:
                if wflux is None:
                    if reducer == 'average':
                        cb_title_use=r"$\Delta f(v)\,["+pt.plot.rmstring('m')+r"^{-6} \,"+pt.plot.rmstring('s')+r"^{3}]$"
                    elif reducer == 'integrate':
                        cb_title_use=r"$\Delta f(v)\,["+pt.plot.rmstring('m')+r"^{-5} \,"+pt.plot.rmstring('s')+r"^{2}]$"
                else:
                    if reducer == 'average':
                        cb_title_use=r"flux $\Delta F\,["+pt.plot.rmstring('m')+r"^{-2} \,"+pt.plot.rmstring('s')+r"^{-1} \,"+pt.plot.rmstring('sr')+r"^{-1}]$"
                    elif reducer == 'integrate':
                        cb_title_use=r"flux $\Delta F\,["+pt.plot.rmstring('m')+r"^{-1} \,"+pt.plot.rmstring('s')+r"^{-2} \,"+pt.plot.rmstring('sr')+r"^{-1}]$"

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
            cb_title_use = pt.plot.mathmode(pt.plot.bfstring(cb_title_use))

            # First draw colorbar
            cb = plt.colorbar(fig1,ticks=ticks,cax=cax)
            cb.outline.set_linewidth(thick)
            cb.ax.yaxis.set_ticks_position(cbdir)
            if cbaxes is None:
                cb.ax.tick_params(labelsize=fontsize3)#,width=1.5,length=3)
                cb_title = cax.set_title(cb_title_use,fontsize=fontsize3,fontweight='bold', horizontalalignment=horalign)
            else:
                cb.ax.tick_params(labelsize=fontsize)
                cb_title = cax.set_title(cb_title_use,fontsize=fontsize,fontweight='bold', horizontalalignment=horalign)
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
                plt.close()
            except:
                logging.info("Error with attempting to save figure due to matplotlib LaTeX integration.")
            logging.info(savefigname+"\n")
        elif axes is None:
            # Draw on-screen
            plt.draw()
            plt.show()
