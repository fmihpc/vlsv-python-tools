import matplotlib
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.ticker import LogLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ids3d
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import mpl_toolkits.mplot3d.art3d as art3d

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


# Create the 3d axes and the coordinate axes for the 3d plot
def axes3d(fig, reflevel,  xr, yr, zr, xsize, ysize, zsize):
    # get 3d axes
    ax = fig.gca(projection='3d')



    # create 3d coordinate axes lines which cuts the point at xr, yr and zr
    line=art3d.Line3D(*zip((0, yr, zr), (xsize*2**reflevel, yr, zr)),
                      color='black', linewidth=0.8, alpha=0.25, zorder=100)
    ax.add_line(line)
    line=art3d.Line3D(*zip((xr, 0, zr), (xr, ysize*2**reflevel, zr)),
                      color='black', linewidth=0.8, alpha=0.25, zorder=100)
    ax.add_line(line)
    line=art3d.Line3D(*zip((xr, yr, 0), (xr, yr, ysize*2**reflevel)),
                      color='black', linewidth=0.8, alpha=0.25, zorder=100)
    ax.add_line(line)



    # spacing between ticks
    s = int((xsize*2**reflevel)/20)
    # widths of the ticks
    l = int((xsize*2**reflevel)/100)


    ###################
    # plot ticks STARTS    

    # plot the first ticks which position > xr or in the others yr or zr
    i = 1
    while xr+i*s < xsize*2**reflevel:
      line=art3d.Line3D(*zip((xr+i*s, yr-l, zr), (xr+i*s, yr+l, zr)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      line=art3d.Line3D(*zip((xr+i*s, yr, zr-l), (xr+i*s, yr, zr+l)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      i += 1

    # plot the last ticks which position < xr or in the others yr or zr
    i = 1
    while xr-i*s > 0:
      line=art3d.Line3D(*zip((xr-i*s, yr-l, zr), (xr-i*s, yr+l, zr)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      line=art3d.Line3D(*zip((xr-i*s, yr, zr-l), (xr-i*s, yr, zr+l)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      i += 1


    i = 1
    while yr+i*s < ysize*2**reflevel:
      line=art3d.Line3D(*zip((xr-l, yr+i*s, zr), (xr+l, yr+i*s, zr)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      line=art3d.Line3D(*zip((xr, yr+i*s, zr-l), (xr, yr+i*s, zr+l)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      i += 1

    i = 1
    while yr-i*s > 0:
      line=art3d.Line3D(*zip((xr-l, yr-i*s, zr), (xr+l, yr-i*s, zr)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      line=art3d.Line3D(*zip((xr, yr-i*s, zr-l), (xr, yr-i*s, zr+l)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      i += 1


    i = 1
    while zr+i*s < zsize*2**reflevel:
      line=art3d.Line3D(*zip((xr, yr-l, zr+i*s), (xr, yr+l, zr+i*s)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      line=art3d.Line3D(*zip((xr-l, yr, zr+i*s), (xr+l, yr, zr+i*s)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      i += 1

    i = 1
    while zr-i*s > 0:
      line=art3d.Line3D(*zip((xr, yr-l, zr-i*s), (xr, yr+l, zr-i*s)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      line=art3d.Line3D(*zip((xr-l, yr, zr-i*s), (xr+l, yr, zr-i*s)),
                        color='black', linewidth=0.7, alpha=0.25, zorder=100+i)
      ax.add_line(line)
      i += 1

    # plot ticks ENDS
    #################



    # set the basic 3d coordinate axes off
    ax.set_axis_off()

    # return axes
    return ax




def plot_threeslice(filename=None,
                  vlsvobj=None,
                  filedir=None, step=None,
                  var=None, op=None, operator=None,
                  colormap=None,
                  thick=1.0,scale=1.0,
                  expression=None,
                  vscale=1.0,
                  cutpoint=None
                  ):

    ''' Plots a 3d plot constructed of three 2d cut throughs.

    :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj:     Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename                    

    :kword var:         variable to plot, e.g. rho, RhoBackstream, beta, Temperature, MA, Mms, va, vms,
                        E, B, v, V or others. Accepts any variable known by analysator.
                        Per-population variables are simply given as "proton/rho" etc
    :kword operator:    Operator to apply to variable: None, x, y, or z. Vector variables return either
                        the queried component, or otherwise the magnitude. 
    :kword op:          duplicate of operator
           
    :kword colormap:    colour scale for plot, use e.g. hot_desaturated, jet, viridis, plasma, inferno,
                        magma, parula, nipy_spectral, RdBu, bwr

    :kword scale:       Scale text size (default=1.0)
    :kword thick:       line and axis thickness, default=1.0

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

    :kword cutpoint:    The point which the all three 2D cut through cuts. 

    .. code-block:: python

    '''

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
    # Small internal colorbar needs increased font size


    Re = 6.371e+6 # Earth radius in m
    # read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize
    cellids = f.read_variable("CellID")


    ############################################
    # Read data and calculate required variables
    ############################################
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
            datamap_unit=r"${\times}$"+fmt(vscale,None)
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

        # Verify data shape
        datamap = datamap_info.data
        if np.ndim(datamap)==0:
            print("Error, read only single value from vlsv file!",datamap.shape)
            return -1
        if np.ndim(datamap)==2:
            if len(datamap[0,:])!=3:
                print("Error, expected array of 3-element vectors, found array of shape ",datamap.shape)
                return -1
            # 2-dimensional array: take magnitude of three-element vectors
            datamap = np.linalg.norm(datamap, axis=-1)
        if np.ndim(datamap)!=1:
            # Array dimensions not as expected
            print("Error reading variable "+var+"! Found array of shape ",datamap.shape,". Exiting.")
            return -1


        # sort cellid and datamap
        indexids = cellids.argsort()
        cellids = cellids[indexids]
        datamap = datamap[indexids]

        # find the highest refiment level
        reflevel = ids3d.refinement_level(xsize, ysize, zsize, cellids[-1])

        # coordinates of the point where the all three 2d cut throughs cuts
        xr = abs(xmin) + cutpoint[0]
        yr = abs(ymin) + cutpoint[1]
        zr = abs(zmin) + cutpoint[2]


        #############################################################
        # Create the needed data grids for the all three cut throughs
        idlist, indexlist = ids3d.ids3d(cellids, xr, reflevel, xsize, ysize, zsize, xmin=xmin, xmax=xmax)
        datamap1 = datamap[indexlist]
        datamap1 = ids3d.idmesh3d(idlist, datamap1, reflevel, xsize, ysize, zsize, 0)

        idlist, indexlist = ids3d.ids3d(cellids, yr, reflevel, xsize, ysize, zsize, ymin=ymin, ymax=ymax)
        datamap2 = datamap[indexlist]
        datamap2 = ids3d.idmesh3d(idlist, datamap2, reflevel, xsize, ysize, zsize, 1)

        idlist, indexlist = ids3d.ids3d(cellids, zr, reflevel, xsize, ysize, zsize, zmin=zmin, zmax=zmax)
        datamap3 = datamap[indexlist]
        datamap3 = ids3d.idmesh3d(idlist, datamap3, reflevel, xsize, ysize, zsize, 2)
    else:
        # Expression set, use generated or provided colorbar title
        cb_title_use = expression.__name__.replace("_","\_")



    # find the smallest and the largest values of the all three surface grids
    mini = [np.ma.amin(datamap1[datamap1 > 0]), np.ma.amin(datamap2[datamap2 > 0]), np.ma.amin(datamap3[datamap3 > 0])]
    maxi = [np.ma.amax(datamap1[datamap1 > 0]), np.ma.amax(datamap2[datamap2 > 0]), np.ma.amax(datamap3[datamap3 > 0])]


    # the smallest value of the surface grids
    vmin = np.ma.amin(mini)
    # the largest value of the surface grids
    vmax = np.ma.amax(maxi)


    # do we have logarithmic or linear norm # NOT FINISHED
    #norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    norm = LogNorm(vmin=vmin, vmax=vmax)


    # coordinates of the point where the all three 2d cut throughs cut in terms of cells
    xr = int(round((xr/(xmax - xmin))*xsize*2**reflevel))
    yr = int(round((yr/(ymax - ymin))*ysize*2**reflevel))
    zr = int(round((zr/(zmax - zmin))*zsize*2**reflevel))


    # create a new figure and a 3d axes with a custom 3d coordinate axes 
    fig = plt.figure(figsize=(12,12))
    ax = axes3d(fig, reflevel,  xr, yr, zr, xsize, ysize, zsize)


    #########################
    # PLOT THE 3D PLOT STARTS
    #########################

    # create 2d x, y, z arrays for a one surface element of the 3d plot
    y = np.arange(yr + 1)
    z = np.arange(zr + 1)
    Y,Z = np.meshgrid(y,z)
    X = Y*0 + xr

    # plot a surface element of the 3d plot
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap1[:zr, :yr])),
                    shade=False, antialiased=False)


    y = np.arange(yr, int(ysize*2**reflevel + 1))
    z = np.arange(zr + 1)
    Y,Z = np.meshgrid(y,z)
    X = Y*0 + xr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap1[:zr, yr:])),
                    shade=False, antialiased=False)


    y = np.arange(yr + 1)
    z = np.arange(zr, int(zsize*2**reflevel + 1))
    Y,Z = np.meshgrid(y,z)
    X = Y*0 + xr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap1[zr:, :yr])),
                    shade=False, antialiased=False)


    y = np.arange(yr, int(ysize*2**reflevel + 1))
    z = np.arange(zr, int(zsize*2**reflevel + 1))
    Y,Z = np.meshgrid(y,z)
    X = Y*0 + xr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap1[zr:, yr:])),
                    shade=False, antialiased=False)



    x = np.arange(xr + 1)
    z = np.arange(zr + 1)
    X,Z = np.meshgrid(x,z)
    Y = X*0 + yr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap2[:zr, :xr])), 
                    shade=False, antialiased=False)


    x = np.arange(xr, int(xsize*2**reflevel + 1))
    z = np.arange(zr + 1)
    X,Z = np.meshgrid(x,z)
    Y = X*0 + yr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap2[:zr, xr:])),
                    shade=False, antialiased=False)


    x = np.arange(xr + 1)
    z = np.arange(zr, int(zsize*2**reflevel + 1))
    X,Z = np.meshgrid(x,z)
    Y = X*0 + yr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap2[zr:, :xr])),
                    shade=False, antialiased=False)


    x = np.arange(xr, int(xsize*2**reflevel + 1))
    z = np.arange(zr, int(zsize*2**reflevel + 1))
    X,Z = np.meshgrid(x,z)
    Y = X*0 + yr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap2[zr:, xr:])),
                    shade=False, antialiased=False)



    x = np.arange(xr + 1)
    y = np.arange(yr + 1)
    X,Y = np.meshgrid(x,y)
    Z = X*0 + zr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap3[:yr, :xr])),
                    shade=False, antialiased=False)


    x = np.arange(xr, int(xsize*2**reflevel + 1))
    y = np.arange(yr + 1)
    X,Y = np.meshgrid(x,y)
    Z = X*0 + zr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap3[:yr, xr:])),
                    shade=False, antialiased=False)


    x = np.arange(xr + 1)
    y = np.arange(yr, int(ysize*2**reflevel + 1))
    X,Y = np.meshgrid(x,y)
    Z = X*0 + zr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap3[yr:, :xr])),
                    shade=False, antialiased=False)


    x = np.arange(xr, int(xsize*2**reflevel + 1))
    y = np.arange(yr, int(ysize*2**reflevel + 1))
    X,Y = np.meshgrid(x,y)
    Z = X*0 + zr

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    facecolors=plt.cm.plasma(norm(datamap3[yr:, xr:])),
                    shade=False, antialiased=False)

    #######################
    # PLOT THE 3D PLOT ENDS
    #######################



    # create a ScalarMappable object to make a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=norm)
    sm.set_array([])

    # compute the positions of the major ticks for the colorbar
    ab = np.log10(vmax) - np.log10(vmin)
    abfra = ab/6
    ticks = [10**(np.log10(vmin) + abfra*i) for i in range(7)]

    # draw the colorbar
    cb = plt.colorbar(sm, ticks=ticks, format=mtick.FuncFormatter(fmt), drawedges=False)


    #if len(cb_title_use)!=0:
    #  cb_title_use = r"\textbf{"+cb_title_use+"}"

    # define the size of the colorbar ticks
    cb.ax.tick_params(labelsize=fontsize3*1.7) # 1.7 is just a temporary scaling value
    # make a titel for the colorbar
    cb.ax.set_title(cb_title_use, fontsize=fontsize3*1.7, weight="bold", # same thing here as in the later
                    position=(0.,1.+0.025*scale), horizontalalignment="left")



    # show the 3d plot on-screen
    plt.show()
