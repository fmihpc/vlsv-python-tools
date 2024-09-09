import matplotlib
import logging
import pytools as pt
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import re
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm,LogNorm,SymLogNorm
from matplotlib.colors import LightSource
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.ticker import LogLocator
import matplotlib.ticker as mtick
import colormaps as cmaps
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from packaging.version import Version


import ids3d
#from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import mpl_toolkits.mplot3d.art3d as art3d

import time


# Create the 3d axes and the coordinate axes for the 3d plot
def axes3d(fig, reflevel, cutpoint, boxcoords, axisunit, axisunituse, tickinterval, fixedticks, scale,
           thick, axiscolor, viewangle, halfaxes, slices, drawEarth):

    Re = 6371e3
    deg2rad = np.pi/180.
    fontsize = 8*scale
    maxextent = np.amax([boxcoords[1]-boxcoords[0],boxcoords[3]-boxcoords[2],boxcoords[5]-boxcoords[4]])
    axextents = np.asarray(boxcoords) + 0.1*maxextent*np.asarray([-1,1,-1,1,-1,1])
    (xr,yr,zr) = [cutpoint[i]/axisunituse for i in range(0,3)]

    # Create 3d axes
    ax = fig.add_axes([.1,.1,.64,.8],projection='3d')

    # Thickness
    linewidth3d=0.5*thick
    tickwidth3d=0.25*thick
    # Arrow size
    axisarrowscale = np.cbrt((axextents[1]-axextents[0])*(axextents[3]-axextents[2])*(axextents[5]-axextents[4]))*0.015
    # Tick size
    ticklength = np.cbrt((axextents[1]-axextents[0])*(axextents[3]-axextents[2])*(axextents[5]-axextents[4]))*0.012

    # Dealing with the plot orientation
    azi,ele = viewangle
    if azi > 180.: # Make sure azimuths are within ]-180.,180]
        azi = azi - 360.
    if azi%90. == 0.: # Dirty hack to avoid potential division by zero later
        azi = azi + 1e-15
    ax.azim = azi
    ax.elev = ele

    # Near-Earth break distances if an axis line goes through the Earth
    RePerAxUnit = Re/axisunituse
    (cXm, cXp, cYm, cYp, cZm, cZp) = (RePerAxUnit,RePerAxUnit,RePerAxUnit,RePerAxUnit,RePerAxUnit,RePerAxUnit)
    if abs(azi) < 90.:
        cXm = (1. + abs(np.sin(ele*deg2rad)/np.tan(azi*deg2rad)))*RePerAxUnit
    else:
        cXp = (1. + abs(np.sin(ele*deg2rad)/np.tan(azi*deg2rad)))*RePerAxUnit
    if azi > 0.:
        cYm = (1. + abs(np.sin(ele*deg2rad)*np.tan(azi*deg2rad)))*RePerAxUnit
    else:
        cYp = (1. + abs(np.sin(ele*deg2rad)*np.tan(azi*deg2rad)))*RePerAxUnit
    if ele > 0.:
        cZm = (1.1/np.cos(ele*deg2rad))*RePerAxUnit
    else:
        cZp = (1.1/np.cos(ele*deg2rad))*RePerAxUnit

    # Earth-breaking conditions
    earthbreak_x = abs(yr) < RePerAxUnit and abs(zr) < RePerAxUnit and axextents[0]*axextents[1] < 0.
    earthbreak_y = abs(xr) < RePerAxUnit and abs(zr) < RePerAxUnit and axextents[2]*axextents[3] < 0.
    earthbreak_z = abs(xr) < RePerAxUnit and abs(yr) < RePerAxUnit and axextents[4]*axextents[5] < 0.
    earthvisible = drawEarth
    # Adapting line styles for axes and near-Earth break distances to the viewing angle
    frontaxisstyle = '-'
    if not halfaxes:
        backaxisstyle = '--'
    else:
        backaxisstyle = ''
    if ele >= 0.:
        styleZp = frontaxisstyle
        if 'z' in slices:
            styleZm = backaxisstyle
        else:
            styleZm = frontaxisstyle
    else:
        if 'z' in slices:
            styleZp = backaxisstyle
        else:
            styleZp = frontaxisstyle
        styleZm = frontaxisstyle
    if azi >= 0:
        styleYp = frontaxisstyle
        if 'y' in slices:
            styleYm = backaxisstyle
        else:
            styleYm = frontaxisstyle
        cYp = 1.
    else:
        if 'y' in slices:
            styleYp = backaxisstyle
        else:
            styleYp = frontaxisstyle
        styleYm = frontaxisstyle
    if abs(azi) <= 90:
        styleXp = frontaxisstyle
        if 'x' in slices:
            styleXm = backaxisstyle
        else:
            styleXm = frontaxisstyle
    else:
        if 'x' in slices:
            styleXp = backaxisstyle
        else:
            styleXp = frontaxisstyle
        styleXm = frontaxisstyle

    # Coefficients for axis label placement
    labposscale = np.cbrt((axextents[1]-axextents[0])*(axextents[3]-axextents[2])*(axextents[5]-axextents[4]))*0.02
    cXlabel = labposscale*(1. + (2.*abs(np.cos(azi*deg2rad))+2.*abs(np.sin(azi*deg2rad)))/abs(np.sin(ele*deg2rad)))
    cYlabel = labposscale*(1. + (2.*abs(np.sin(azi*deg2rad))+2.*abs(np.cos(azi*deg2rad)))/abs(np.sin(ele*deg2rad)))
    cZlabel = labposscale*(1. + 1.5*abs(np.tan(ele*deg2rad)))

    # Axes and units (default R_E)
    if np.isclose(axisunituse,Re):
        axisunitstr = pt.plot.rmstring('R')+'_'+pt.plot.rmstring('E')
    elif np.isclose(axisunituse,1):
        axisunitstr = pt.plot.rmstring('m')
    elif np.isclose(axisunituse,1000):
        axisunitstr = pt.plot.rmstring('km')
    else:
        axisunitstr = r'10^{'+str(int(axisunit))+'} '+pt.plot.rmstring('m')

    # Create axis lines intersecting at (xr,yr,zr)
    # -- x-axis --
    if not earthbreak_x:
        line=art3d.Line3D(*zip((axextents[0], yr, zr), (xr, yr, zr)),
                          color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
        ax.add_line(line) # this goes from the lowest X to the Earth or the cut point

        line=art3d.Line3D(*zip((xr, yr, zr), (axextents[1], yr, zr)),
                          color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
        ax.add_line(line) # this goes from the Earth or the cut point to the highest X

    else: # Special treatment of cases when the x-axis goes through the Earth
        # Distinguish four cases to avoid overlaying the Earth, based on the earthbreak condition and azi
        if xr <= 0. and abs(azi) < 90.: # looking from the dayside, cut point on the nightside
            line=art3d.Line3D(*zip((axextents[0], yr, zr), (min(xr,-cXm), yr, zr)),
                              color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

            if xr < -RePerAxUnit:
                line=art3d.Line3D(*zip((xr, yr, zr), (-cXm, yr, zr)),
                                  color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

            line=art3d.Line3D(*zip((cXp, yr, zr), (axextents[1], yr, zr)),
                              color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)
        elif xr > 0. and abs(azi) < 90.: # looking from the dayside, cut point also on the dayside
            if 'x' in slices and xr > RePerAxUnit: # Earth effectively hidden by the X slice, no need to break the axis
                earthvisible = False
                line=art3d.Line3D(*zip((axextents[0], yr, zr), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, zr), (axextents[1], yr, zr)),
                                  color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
            else:
                line=art3d.Line3D(*zip((axextents[0], yr, zr), (-cXm, yr, zr)),
                                  color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                if xr > RePerAxUnit:
                    line=art3d.Line3D(*zip((cXp, yr, zr), (xr, yr, zr)),
                                      color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
                    ax.add_line(line)

                line=art3d.Line3D(*zip((max(xr,cXp), yr, zr), (axextents[1], yr, zr)),
                                  color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
        elif xr <= 0.: # looking from the nightside, cut point also on the nightside
            if 'x' in slices and xr < -RePerAxUnit: # Earth effectively hidden by the X slice, no need to break the axis
                earthvisible = False
                line=art3d.Line3D(*zip((axextents[0], yr, zr), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, zr), (axextents[1], yr, zr)),
                                  color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
            else:
                line=art3d.Line3D(*zip((axextents[0], yr, zr), (min(xr,-cXm), yr, zr)),
                                  color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                if xr < -RePerAxUnit:
                    line=art3d.Line3D(*zip((xr, yr, zr), (-cXm, yr, zr)),
                                      color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
                    ax.add_line(line)

                line=art3d.Line3D(*zip((cXp, yr, zr), (axextents[1], yr, zr)),
                                  color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
        else: # looking from the nightside, cut point on the dayside
            line=art3d.Line3D(*zip((axextents[0], yr, zr), (-cXm, yr, zr)),
                              color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

            if xr > RePerAxUnit:
                line=art3d.Line3D(*zip((cXp, yr, zr), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleXm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

            line=art3d.Line3D(*zip((max(xr,cXp), yr, zr), (axextents[1], yr, zr)),
                              color=axiscolor, linestyle=styleXp, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

    # Make an arrow at the end of the axis
    if halfaxes and styleXp == backaxisstyle:
        line=art3d.Line3D(*zip((xr-1.7*axisarrowscale, yr-axisarrowscale, zr), (xr, yr, zr),
                               (xr-1.7*axisarrowscale, yr+axisarrowscale, zr)), color=axiscolor, linewidth=linewidth3d, alpha=1, zorder=20)
    else:
        line=art3d.Line3D(*zip((axextents[1]-1.7*axisarrowscale, yr-axisarrowscale, zr), (axextents[1], yr, zr),
                               (axextents[1]-1.7*axisarrowscale, yr+axisarrowscale, zr)), color=axiscolor, linewidth=linewidth3d, alpha=1, zorder=20)
    ax.add_line(line)

    # -- y-axis --
    if not earthbreak_y:
        line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, yr, zr)),
                          color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
        ax.add_line(line) # this goes from the lowest Y to the cut point

        line=art3d.Line3D(*zip((xr, yr, zr), (xr, axextents[3], zr)),
                          color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
        ax.add_line(line) # this goes from the cut point to the highest Y

    else: # Special treatment of cases when the y-axis goes through the Earth
        # Distinguish four cases to avoid overlaying the Earth, based on the earthbreak condition and azi
        if yr <= 0. and azi > 0.: # looking from the dawnside, cut point on the duskside
            line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, min(yr,-cYm), zr)),
                              color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)
            if yr < -RePerAxUnit:
                line=art3d.Line3D(*zip((xr, yr, zr), (xr, -cYm, zr)),
                                  color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

            line=art3d.Line3D(*zip((xr, cYp, zr), (xr, axextents[3], zr)),
                              color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)
        elif yr > 0. and azi > 0.: # looking from the dawnside, cut point also on the dawnside
            if 'y' in slices and yr > RePerAxUnit: # Earth effectively hidden by the Y slice, no need to break the axis
                earthvisible = False
                line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, zr), (xr, axextents[3], zr)),
                                  color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
            else:
                line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, -cYm, zr)),
                                  color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                if yr > RePerAxUnit:
                    line=art3d.Line3D(*zip((xr, cYp, zr), (xr, yr, zr)),
                                      color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
                    ax.add_line(line)

                line=art3d.Line3D(*zip((xr, max(yr,cYp), zr), (xr, axextents[3], zr)),
                                  color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
        elif yr <= 0.: # looking from the duskside, cut point also on the duskside
            if 'y' in slices and yr < -RePerAxUnit: # Earth effectively hidden by the Y slice, no need to break the axis
                earthvisible = False
                line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, zr), (xr, axextents[3], zr)),
                                  color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
            else:
                line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, min(yr,-cYm), zr)),
                                  color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                if yr < -RePerAxUnit:
                    line=art3d.Line3D(*zip((xr, yr, zr), (xr, -cYm, zr)),
                                      color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
                    ax.add_line(line)

                line=art3d.Line3D(*zip((xr, cYp, zr), (xr, axextents[3], zr)),
                                  color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
        else: # looking from the duskside, cut point on the dawnside
            line=art3d.Line3D(*zip((xr, axextents[2], zr), (xr, -cYm, zr)),
                              color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

            if yr > RePerAxUnit:
                line=art3d.Line3D(*zip((xr, cYp, zr), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleYm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

            line=art3d.Line3D(*zip((xr, max(yr,cYp), zr), (xr, axextents[3], zr)),
                              color=axiscolor, linestyle=styleYp, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

    # Make an arrow at the end of the axis
    if halfaxes and styleYp == backaxisstyle:
        line=art3d.Line3D(*zip((xr-axisarrowscale, yr-1.7*axisarrowscale, zr), (xr, yr, zr),
                               (xr+axisarrowscale, yr-1.7*axisarrowscale, zr)), color=axiscolor, linewidth=linewidth3d, alpha=1, zorder=20)
    else:
        line=art3d.Line3D(*zip((xr-axisarrowscale, axextents[3]-1.7*axisarrowscale, zr), (xr, axextents[3], zr),
                               (xr+axisarrowscale, axextents[3]-1.7*axisarrowscale, zr)), color=axiscolor, linewidth=linewidth3d, alpha=1, zorder=20)
    ax.add_line(line)

    # -- z-axis --
    if not earthbreak_z:
        line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, zr)),
                          color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
        ax.add_line(line) # this goes from the lowest Z to the cut point

        line=art3d.Line3D(*zip((xr, yr, zr), (xr, yr, axextents[5])),
                          color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
        ax.add_line(line) # this goes from the cut point to the highest Z

    else: # Special treatment of cases when the z-axis goes through the Earth
        # Distinguish four cases to avoid overlaying the Earth, based on the earthbreak condition and ele
        if zr <= 0. and ele > 0.: # looking from the north, cut point south from the Earth
            line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, min(zr,-cZm))),
                              color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

            if zr < -RePerAxUnit:
                line=art3d.Line3D(*zip((xr, yr, zr), (xr, yr, -cZm)),
                                  color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

            line=art3d.Line3D(*zip((xr, yr, cZp), (xr, yr, axextents[5])),
                              color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)
        elif zr > 0. and ele > 0.: # looking from the north, cut point also north from the Earth
            if 'z' in slices and zr > RePerAxUnit: # Earth effectively hidden by the Z slice, no need to break the axis
                earthvisible = False
                line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, zr), (xr, yr, axextents[5])),
                                  color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
            else:
                line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, -cZm)),
                                  color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                if zr > RePerAxUnit:
                    line=art3d.Line3D(*zip((xr, yr, cZp), (xr, yr, zr)),
                                      color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
                    ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, max(zr,cZp)), (xr, yr, axextents[5])),
                                  color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
        elif zr < 0.: # looking from the south, cut point also south from the Earth
            if 'z' in slices and zr < -RePerAxUnit: # Earth effectively hidden by the Z slice, no need to break the axis
                earthvisible = False
                line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, zr), (xr, yr, axextents[5])),
                                  color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
            else:
                line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, min(zr,-cZm))),
                                  color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

                if zr < -RePerAxUnit:
                    line=art3d.Line3D(*zip((xr, yr, zr), (xr, yr, -cZm)),
                                      color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
                    ax.add_line(line)

                line=art3d.Line3D(*zip((xr, yr, cZp), (xr, yr, axextents[5])),
                                  color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)
        else: # looking from the south, cut point north from the Earth
            line=art3d.Line3D(*zip((xr, yr, axextents[4]), (xr, yr, -cZm)),
                              color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

            if zr > RePerAxUnit:
                line=art3d.Line3D(*zip((xr, yr, cZp), (xr, yr, zr)),
                                  color=axiscolor, linestyle=styleZm, linewidth=linewidth3d, alpha=1, zorder=20)
                ax.add_line(line)

            line=art3d.Line3D(*zip((xr, yr, max(zr,cZp)), (xr, yr, axextents[5])),
                              color=axiscolor, linestyle=styleZp, linewidth=linewidth3d, alpha=1, zorder=20)
            ax.add_line(line)

 # Make an arrow at the end of the axis
    if halfaxes and styleZp == backaxisstyle:
        line=art3d.Line3D(*zip((xr-axisarrowscale, yr, zr-1.7*axisarrowscale), (xr, yr, zr),
                               (xr+axisarrowscale, yr, zr-1.7*axisarrowscale)), color=axiscolor, linewidth=linewidth3d, alpha=1, zorder=20)
    else:
        line=art3d.Line3D(*zip((xr-axisarrowscale, yr, axextents[5]-1.7*axisarrowscale), (xr, yr, axextents[5]),
                               (xr+axisarrowscale, yr, axextents[5]-1.7*axisarrowscale)), color=axiscolor, linewidth=linewidth3d, alpha=1, zorder=20)
    ax.add_line(line)

    xlabelstr = pt.plot.mathmode(pt.plot.bfstring(r'X\,['+axisunitstr+']'))
    ylabelstr = pt.plot.mathmode(pt.plot.bfstring(r'Y\,['+axisunitstr+']'))
    zlabelstr = pt.plot.mathmode(pt.plot.bfstring(r'Z\,['+axisunitstr+']'))

    if halfaxes and styleXp == backaxisstyle:
        ax.text(axextents[0]-cXlabel, yr, zr,xlabelstr,fontsize=fontsize,ha='center',va='center',zorder=50, weight='black')
    else:
        ax.text(axextents[1]+cXlabel, yr, zr,xlabelstr,fontsize=fontsize,ha='center',va='center',zorder=50, weight='black')
    if halfaxes and styleYp == backaxisstyle:
        ax.text(xr, axextents[2]-cYlabel, zr,ylabelstr,fontsize=fontsize,ha='center',va='center',zorder=50, weight='black')
    else:
        ax.text(xr, axextents[3]+cYlabel, zr,ylabelstr,fontsize=fontsize,ha='center',va='center',zorder=50, weight='black')
    if halfaxes and styleZp == backaxisstyle:
        ax.text(xr, yr, axextents[4]-cZlabel,zlabelstr,fontsize=fontsize,ha='center',va='center',zorder=50, weight='black')
    else:
        ax.text(xr, yr, axextents[5]+cZlabel,zlabelstr,fontsize=fontsize,ha='center',va='center',zorder=50, weight='black')

    # Addition of the Earth at the origin of the domain
    if earthvisible:
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = RePerAxUnit * np.cos(u)*np.sin(v)
        y = RePerAxUnit * np.sin(u)*np.sin(v)
        z = RePerAxUnit * np.cos(v)
        albedo = np.zeros(u.shape)
        albedo = np.arctan(-50.*x)/np.pi+0.5
        levels = MaxNLocator(nbins=255).tick_values(0,1)
        norm = BoundaryNorm(levels, ncolors=255, clip=True)
        scalarmap = plt.cm.ScalarMappable(cmap='Greys',norm=norm)

        ax.plot_surface(x, y, z, facecolors=scalarmap.to_rgba(albedo),alpha=1,zorder=30)

    ax.set_xlim([boxcoords[0],boxcoords[1]])
    ax.set_ylim([boxcoords[2],boxcoords[3]])
    ax.set_zlim([boxcoords[4],boxcoords[5]])

    # ex.axis('equal') is broken and either does not work at all or produces incorrect output.
    # these are possible alternative approaches
    # ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])
    #  or 
    # limits = np.array([getattr(self.ax, f'get_{axis}lim')() for axis in 'xyz'])
    # ax.set_box_aspect(np.ptp(limits, axis = 1))
    try:
        limits = np.array([getattr(ax, 'get_{}lim'.format(axis))() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))
    except:
        logging.info("WARNING: ax.set_box_aspect() failed (not supported by this version of matplotlib?).")
        try:
            ax.auto_scale_xyz([boxcoords[0],boxcoords[1]],[boxcoords[2],boxcoords[3]],[boxcoords[4],boxcoords[5]])
        except:
            logging.info("WARNING: ax.auto_scale_xyz() failed (not supported by this version of matplotlib?).")

    # Draw ticks
    if not fixedticks: # Ticks are relative to the triple point
        txm = np.arange(xr,boxcoords[0]-0.1,-tickinterval/axisunituse)
        txp = np.arange(xr,boxcoords[1]+0.1,tickinterval/axisunituse)
        ticks_x = np.concatenate((np.flip(txm,axis=0),txp[1:]))

        tym = np.arange(yr,boxcoords[2]-0.1,-tickinterval/axisunituse)
        typ = np.arange(yr,boxcoords[3]+0.1,tickinterval/axisunituse)
        ticks_y = np.concatenate((np.flip(tym,axis=0),typ[1:]))

        tzm = np.arange(zr,boxcoords[4]-0.1,-tickinterval/axisunituse)
        tzp = np.arange(zr,boxcoords[5]+0.1,tickinterval/axisunituse)
        ticks_z = np.concatenate((np.flip(tzm,axis=0),tzp[1:]))
    else: # Ticks are relative to (0,0,0)
        txm = np.arange(0.,boxcoords[0]-0.1,-tickinterval/axisunituse)
        txp = np.arange(0.,boxcoords[1]+0.1,tickinterval/axisunituse)
        ticks_x = np.concatenate((np.flip(txm,axis=0),txp[1:]))

        tym = np.arange(0.,boxcoords[2]-0.1,-tickinterval/axisunituse)
        typ = np.arange(0.,boxcoords[3]+0.1,tickinterval/axisunituse)
        ticks_y = np.concatenate((np.flip(tym,axis=0),typ[1:]))

        tzm = np.arange(0.,boxcoords[4]-0.1,-tickinterval/axisunituse)
        tzp = np.arange(0.,boxcoords[5]+0.1,tickinterval/axisunituse)
        ticks_z = np.concatenate((np.flip(tzm,axis=0),tzp[1:]))

    # Avoid placing a tick on the Earth if it is visible in the plot
    if earthbreak_x and earthvisible:
        ticks_x = ticks_x[abs(ticks_x) > RePerAxUnit]
    if earthbreak_y and earthvisible:
        ticks_y = ticks_y[abs(ticks_y) > RePerAxUnit]
    if earthbreak_z and earthvisible:
        ticks_z = ticks_z[abs(ticks_z) > RePerAxUnit]

    # Remove possible ticks outside of the specified box
    ticks_x = ticks_x[ticks_x >= axextents[0]]
    ticks_x = ticks_x[ticks_x <= axextents[1]]
    ticks_y = ticks_y[ticks_y >= axextents[2]]
    ticks_y = ticks_y[ticks_y <= axextents[3]]
    ticks_z = ticks_z[ticks_z >= axextents[4]]
    ticks_z = ticks_z[ticks_z <= axextents[5]]


    if halfaxes: # do not create ticks if back axes are not shown
        if styleXp == backaxisstyle:
            ticks_x = ticks_x[ticks_x <= xr]
        else:
            ticks_x = ticks_x[ticks_x >= xr]
    for itick in range(0,len(ticks_x)):
        tick=art3d.Line3D(*zip((ticks_x[itick],yr-ticklength,zr),(ticks_x[itick],yr+ticklength,zr)),
                          color=axiscolor, linewidth=tickwidth3d, alpha=1, zorder=20)
        ax.add_line(tick)

        tick=art3d.Line3D(*zip((ticks_x[itick],yr,zr-ticklength),(ticks_x[itick],yr,zr+ticklength)),
                          color=axiscolor, linewidth=tickwidth3d, alpha=1, zorder=20)
        ax.add_line(tick)

    if xr < RePerAxUnit and abs(yr) < RePerAxUnit: # avoid placing a tick on the Earth if it is visible in the plot
        ticks_y = ticks_y[abs(ticks_y) > RePerAxUnit]
    if halfaxes: # do not create ticks if back axes are not shown
        if styleYp == backaxisstyle:
            ticks_y = ticks_y[ticks_y <= yr]
        else:
            ticks_y = ticks_y[ticks_y >= yr]
    for itick in range(0,len(ticks_y)):
        tick=art3d.Line3D(*zip((xr-ticklength,ticks_y[itick],zr),(xr+ticklength,ticks_y[itick],zr)),
                          color=axiscolor, linewidth=tickwidth3d, alpha=1, zorder=20)
        ax.add_line(tick)

        tick=art3d.Line3D(*zip((xr,ticks_y[itick],zr-ticklength),(xr,ticks_y[itick],zr+ticklength)),
                          color=axiscolor, linewidth=tickwidth3d, alpha=1, zorder=20)
        ax.add_line(tick)

    if xr < RePerAxUnit and abs(zr) < RePerAxUnit: # avoid placing a tick on the Earth if it is visible in the plot
        ticks_z = ticks_z[abs(ticks_z) > RePerAxUnit]
    if halfaxes: # do not create ticks if back axes are not shown
        if styleZp == backaxisstyle:
            ticks_z = ticks_z[ticks_z <= zr]
        else:
            ticks_z = ticks_z[ticks_z >= zr]
    for itick in range(0,len(ticks_z)):
        tick=art3d.Line3D(*zip((xr-ticklength,yr,ticks_z[itick]),(xr+ticklength,yr,ticks_z[itick])),
                          color=axiscolor, linewidth=tickwidth3d, alpha=1, zorder=20)
        ax.add_line(tick)

        tick=art3d.Line3D(*zip((xr,yr-ticklength,ticks_z[itick]),(xr,yr+ticklength,ticks_z[itick])),
                          color=axiscolor, linewidth=tickwidth3d, alpha=1, zorder=20)
        ax.add_line(tick)

    # set the basic 3d coordinate axes off
    ax.set_axis_off()

    # return axes
    return ax




def plot_threeslice(filename=None,
                  vlsvobj=None,
                  filedir=None, step=None, run=None,
                  outputdir=None, outputfile=None,
                  nooverwrite=False,
                  var=None, op=None, operator=None,
                  boxm=None,boxre=None,
                  colormap=None, vmin=None, vmax=None,
                  symmetric=False, absolute=None,
                  lin=None, symlog=None, nocb=False,
                  cbtitle=None, title=None, halfaxes=False,
                  usesci=True, axisunit=None,axiscolor=None,
                  shadings=None, draw=None,
                  tickinterval=None, fixedticks=False,
                  pass_full=None,
                  wmark=False,wmarkb=False,
                  nomask=None,
                  Earth=True,
                  thick=1.0,scale=1.0,
                  expression=None,
                  vscale=1.0,viewangle=(-60.,30.),
                  cutpointm=None,cutpointre=None,slices=None
                  ):

    ''' Plots a 3d plot constructed of three 2d cut throughs.

    :kword filename:    path to .vlsv file to use for input. Assumes a bulk file.
    :kword vlsvobj:     Optionally provide a python vlsvfile object instead
    :kword filedir:     Optionally provide directory where files are located and use step for bulk file name
    :kword step:        output step index, used for constructing output (and possibly input) filename
    :kword run:         run identifier, used for constructing output filename
    :kword outputdir:   path to directory where output files are created (default: $HOME/Plots/ or override with PTOUTPUTDIR)
                        If directory does not exist, it will be created. If the string does not end in a
                        forward slash, the final part will be used as a prefix for the files.
    :kword outputfile:  Singular output file name
    :kword draw:        Set to anything but None or False in order to draw image on-screen instead of saving to file (requires x-windowing)

    :kword nooverwrite: Set to only perform actions if the target output file does not yet exist                    

    :kword var:         variable to plot, e.g. rho, RhoBackstream, beta, Temperature, MA, Mms, va, vms,
                        E, B, v, V or others. Accepts any variable known by analysator/pytools.
                        Per-population variables are simply given as "proton/rho" etc
    :kword operator:    Operator to apply to variable: None, x, y, or z. Vector variables return either
                        the queried component, or otherwise the magnitude. 
    :kword op:          duplicate of operator

    :kword boxm:        zoom box extents [x0,x1,y0,y1,z0,z1] in metres (default and truncate to: whole simulation box)
    :kword boxre:       zoom box extents [x0,x1,y0,y1,z0,z1] in Earth radii (default and truncate to: whole simulation box)

    :kword colormap:    colour scale for plot, use e.g. hot_desaturated, jet, viridis, plasma, inferno,
                        magma, parula, nipy_spectral, RdBu, bwr
    :kword vmin,vmax:   min and max values for colour scale and colour bar. If no values are given,
                        min and max values for whole plot (non-zero rho regions only) are used.
    :kword symmetric:   Set the absolute value of vmin and vmax to the greater of the two
    :kword absolute:    Plot the absolute of the evaluated variable
    :kword lin:         Flag for using linear colour scaling instead of log
    :kword symlog:      Use logarithmic scaling, but linear when abs(value) is below the value given to symlog.
                        Allows symmetric quasi-logarithmic plots of e.g. transverse field components.
                        A given of 0 translates to a threshold of max(abs(vmin),abs(vmax)) * 1.e-2, but this can
                        result in the innermost tick marks overlapping. In this case, using a larger value for 
                        symlog is suggested.
    :kword nocb:        Set to suppress drawing of colourbar
    :kword title:       string to use as plot title instead of time.
                        Special case: Set to "msec" to plot time with millisecond accuracy or "musec"
                        for microsecond accuracy. "sec" is integer second accuracy.
    :kword cbtitle:     string to use as colorbar title instead of map name
    :kword halfaxes:    Flag to prevent plotting the hidden half of the axes, otherwise shown as dashed lines
                        (default: False)
    :kwird usesci:      Use scientific notation for colorbar ticks? (default: True)
    :kword axisunit:    Plot axes using 10^{axisunit} m (default: Earth radius R_E)
    :kword axiscolor:   Color for drawing axes (default black)
    :kword shadings:    Dimming values for the surfaces normal to the X, Y, Z directions, respectively,
                        to better distinguish the slices, e.g. "(0.6,0.8,1.0). Default using the angles between
                        the surface normal directions and the viewing angle.
    :kword tickinterval:Axis tick interval, expressed in 10^{axisunit} (default: 10 Earth radii)
    :kword fixedticks:  Set ticks at fixed locations instead of relative to the triple cut point (default: False)

    :kword pass_full:   Set to anything but None in order to pass the full arrays instead of a zoomed-in section

    :kword wmark:       If set to non-zero, will plot a Vlasiator watermark in the top left corner. If set to a text
                        string, tries to use that as the location, e.g. "NW","NE","SW","SW"
    :kword wmarkb:      As for wmark, but uses an all-black Vlasiator logo.
    :kword nomask:      Do not mask plotting based on proton density
    :kword Earth:       Draw Earth at origin (default True)

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
                        variables. Set to None to search for a default scaling.
    :kword viewangle:   Azimuth and elevation angles giving the point of view on the 3D axes, in degrees.
                        (default=(-60.,30.); corresponds to dayside, morningside, northern hemisphere)

    :kword cutpointm:    Coordinates of the point through which all three 2D cuts must pass [m]
    :kword cutpointre:  Coordinates of the point through which all three 2D cuts must pass [rE]
    :kword slices:      Normal directions of the slices to plot, default='xyz'

    .. code-block:: python

    '''

    t0 = time.time()

    # Verify the location of this watermark image
    watermarkimage=os.path.join(os.path.dirname(__file__), 'logo_color.png')
    watermarkimageblack=os.path.join(os.path.dirname(__file__), 'logo_black.png')

    # Switch None-keywords to empty lists (this way subsequent calls get correct empty default values
    if boxm is None:
        boxm=[],
    if boxre is None:
        boxre=[]
#    if pass_vars is None:    # Not yet implemented
#        pass_vars=[]

    # Change certain falsy values:
    if not lin and lin != 0:
        lin = None
    if not symlog and symlog != 0:
        symlog = None
    if symlog is True:
        symlog = 0
    if (filedir == ''):
        filedir = './'
    if (outputdir == ''):
        outputdir = './'

    # Input file or object
    if filename!=None:
        f=pt.vlsvfile.VlsvReader(filename)
    elif ((filedir!=None) and (step!=None)):
        filename = glob.glob(filedir+'bulk*'+str(step).rjust(7,'0')+'.vlsv')[0]
        f=pt.vlsvfile.VlsvReader(filename)
    elif vlsvobj!=None:
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

    if axiscolor is None:
        axiscolor='black'

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
            plot_title = "t="+timeformat.format(timeval)+' s '
    else:
        plot_title = ''+title

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
        if filename:
            # If working within CSC filesystem, make a guess:
            if filename[0:16]=="/proj/vlasov/3D/":
                run = filename[16:19]

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
                lin=True
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
        varstr=var.replace("/","_")

    if not outputfile: # Generate filename
        if not outputdir: # default initial path
            outputdir=pt.plot.defaultoutputdir
        # Sub-directories can still be defined in the "run" variable
        outputfile = outputdir+run+"_threeSlice_"+varstr+operatorfilestr+stepstr+".png"
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

    # The plot will be saved in a new figure ('draw' and 'axes' keywords not implemented yet)
    if str(matplotlib.get_backend()) != pt.backend_noninteractive: #'Agg':
        plt.switch_backend(pt.backend_noninteractive)

    # Select plotting back-end based on on-screen plotting or direct to file without requiring x-windowing
    if draw:
        if str(matplotlib.get_backend()) != pt.backend_interactive: #'TkAgg': 
            plt.switch_backend(pt.backend_interactive)
    else:
        if str(matplotlib.get_backend()) != pt.backend_noninteractive: #'Agg':
            plt.switch_backend(pt.backend_noninteractive)  

    Re = 6.371e+6 # Earth radius in m
    # read in mesh size and cells in ordinary space
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    xsize = int(xsize)
    ysize = int(ysize)
    zsize = int(zsize)
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    cellsize = (xmax-xmin)/xsize
    cellids = f.read_variable("CellID")
    sizes = [xsize,ysize,zsize]

    # Read the FSgrid mesh
    [xsizefg, ysizefg, zsizefg] = f.get_fsgrid_mesh_size()
    xsizefg = int(xsizefg)
    ysizefg = int(ysizefg)
    zsizefg = int(zsizefg)
    [xminfg, yminfg, zminfg, xmaxfg, ymaxfg, zmaxfg] = f.get_fsgrid_mesh_extent()
    cellsizefg = (xmaxfg-xminfg)/xsizefg
    pt.plot.plot_helpers.CELLSIZE = cellsizefg
    
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

    # Simulation domain outer boundaries
    simext=[xmin,xmax,ymin,ymax,zmin,zmax]

    # Coordinates of the point through which the three slices pass
    if not cutpointm:
        if cutpointre:
            cutpoint = np.asarray(cutpointre) * Re
        else: # default to [0,0,0]
            logging.info('No cut point coordinates given, defaulting to origin')
            cutpoint = np.asarray([0.,0.,0.])
    else:
        cutpoint = np.asarray(cutpointm)
    # Slices to be plotted (defined by their normal direction)
    if slices is None:
        slices = 'xyz'

    # Select window to draw
    if len(boxm)==6:
        boxcoords=list(boxm)
    elif len(boxre)==6:
        boxcoords=[i*Re for i in boxre]
    else:
        boxcoords=list(simext)

    # Truncate to simulation extents
    boxcoords[0] = max(boxcoords[0],simext[0])
    boxcoords[1] = min(boxcoords[1],simext[1])
    boxcoords[2] = max(boxcoords[2],simext[2])
    boxcoords[3] = min(boxcoords[3],simext[3])
    boxcoords[4] = max(boxcoords[4],simext[4])
    boxcoords[5] = min(boxcoords[5],simext[5])

    # If cutpoint is outside box coordinates, turn off that slice
    if (cutpoint[0]<boxcoords[0]) or (cutpoint[0]>boxcoords[1]):
        slices = slices.replace('x','')
        logging.info("Note: adjusting box extents to include cut point (x)")
    if (cutpoint[1]<boxcoords[2]) or (cutpoint[1]>boxcoords[3]):
        slices = slices.replace('y','')
        logging.info("Note: adjusting box extents to include cut point (y)")
    if (cutpoint[2]<boxcoords[4]) or (cutpoint[2]>boxcoords[5]):
        slices = slices.replace('z','')
        logging.info("Note: adjusting box extents to include cut point (z)")

    if len(slices)==0:
        logging.info("Error: no active slices at cutpoint within box domain")
        return -1

    # Also, if necessary, adjust box coordinates to extend a bit beyond the cutpoint.
    # This is preferable to moving the cutpoint, and required by the quadrant-drawing method.
    boxcoords[0] = min(boxcoords[0],cutpoint[0]-2*cellsizefg)
    boxcoords[1] = max(boxcoords[1],cutpoint[0]+2*cellsizefg)
    boxcoords[2] = min(boxcoords[2],cutpoint[1]-2*cellsizefg)
    boxcoords[3] = max(boxcoords[3],cutpoint[1]+2*cellsizefg)
    boxcoords[4] = min(boxcoords[4],cutpoint[2]-2*cellsizefg)
    boxcoords[5] = max(boxcoords[5],cutpoint[2]+2*cellsizefg)

    # Re-truncate to simulation extents
    boxcoords[0] = max(boxcoords[0],simext[0])
    boxcoords[1] = min(boxcoords[1],simext[1])
    boxcoords[2] = max(boxcoords[2],simext[2])
    boxcoords[3] = min(boxcoords[3],simext[3])
    boxcoords[4] = max(boxcoords[4],simext[4])
    boxcoords[5] = min(boxcoords[5],simext[5])

    # Move cutpoint inwards if it too close to simulation outer extent.
    cutpoint[0] = max(cutpoint[0], simext[0]+2*cellsizefg)
    cutpoint[0] = min(cutpoint[0], simext[1]-2*cellsizefg)
    cutpoint[1] = max(cutpoint[1], simext[2]+2*cellsizefg)
    cutpoint[1] = min(cutpoint[1], simext[3]-2*cellsizefg)
    cutpoint[2] = max(cutpoint[2], simext[4]+2*cellsizefg)
    cutpoint[2] = min(cutpoint[2], simext[5]-2*cellsizefg)

    # Axes and units (default R_E)
    if axisunit is not None: # Use m or km or other
        if np.isclose(axisunit,0):
            axisunitstr = pt.plot.rmstring('m')
        elif np.isclose(axisunit,3):
            axisunitstr = pt.plot.rmstring('km')
        else:
            axisunitstr = r'$10^{'+str(int(axisunit))+'} '+pt.plot.rmstring('m')
        axisunituse = np.power(10,int(axisunit))
    else:
        axisunitstr = pt.plot.rmstring('R')+'_'+pt.plot.rmstring('E')
        axisunituse = Re

    # Scale data extent and plot box
    simext=[i/axisunituse for i in simext]
    boxcoords=[i/axisunituse for i in boxcoords]

    # Axis tick interval (default 10 R_E)
    if tickinterval is None:
        tickinterval = 10.*Re
    else:
        tickinterval = tickinterval * axisunituse


    ###################################################
    # Find the cellids corresponding to the 3 slices #
    ###################################################

    # {X = x0} slice
    sliceoffset = abs(xmin) + cutpoint[0]
    fgslice_x = int(sliceoffset/cellsizefg)
    if 'x' in slices: idlist_x, indexlist_x = ids3d.ids3d(cellids, sliceoffset, reflevel, xsize, ysize, zsize, xmin=xmin, xmax=xmax)

    # {Y = y0} slice
    sliceoffset = abs(ymin) + cutpoint[1]
    fgslice_y = int(sliceoffset/cellsizefg)
    if 'y' in slices: idlist_y, indexlist_y = ids3d.ids3d(cellids, sliceoffset, reflevel, xsize, ysize, zsize, ymin=ymin, ymax=ymax)

    # {Z = z0} slice
    sliceoffset = abs(zmin) + cutpoint[2]
    fgslice_z = int(sliceoffset/cellsizefg)
    if 'z' in slices: idlist_z, indexlist_z = ids3d.ids3d(cellids, sliceoffset, reflevel, xsize, ysize, zsize, zmin=zmin, zmax=zmax)

    #################################################
    # Find rhom map for use in masking out ionosphere
    #################################################
    if f.check_variable("moments"):
        rhomap = f.read_variable("vg_restart_rhom")
    elif f.check_variable("proton/vg_rho"):
        rhomap = f.read_variable("proton/vg_rho")
    elif f.check_variable("vg_rhom"):
        rhomap = f.read_variable("vg_rhom")
    else:
        # No rhomap found - do not perform any masking.
        nomask = True
    if not nomask:
        rhomap = rhomap[indexids]  # sort
        if 'x' in slices:
            rhomap_x = rhomap[indexlist_x] # find required cells (X cut)
            rhomap_x = ids3d.idmesh3d(idlist_x, rhomap_x, reflevel, xsize, ysize, zsize, 0, None)
        if 'y' in slices:
            rhomap_y = rhomap[indexlist_y] # find required cells (Y cut)
            rhomap_y = ids3d.idmesh3d(idlist_y, rhomap_y, reflevel, xsize, ysize, zsize, 1, None)
        if 'z' in slices:
            rhomap_z = rhomap[indexlist_z] # find required cells (Z cut)
            rhomap_z = ids3d.idmesh3d(idlist_z, rhomap_z, reflevel, xsize, ysize, zsize, 2, None)

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
        # Dummy variables
        datamap_x = []
        datamap_y = []
        datamap_z = []

        # Verify data shape
        if np.ndim(datamap)==0:
            logging.info("Error, read only single value from vlsv file! datamap.shape being " + str(datamap.shape))
            return -1

        if var.startswith('fg_'):
            # fsgrid reader returns array in correct shape but needs to be sliced and transposed
            if np.ndim(datamap)==3: # scalar variable
                datamap_x = datamap[fgslice_x,:,:]
                datamap_y = datamap[:,fgslice_y,:]
                datamap_z = datamap[:,:,fgslice_z]
            elif np.ndim(datamap)==4: # vector variable
                datamap_x = datamap[fgslice_x,:,:,:]
                datamap_y = datamap[:,fgslice_y,:,:]
                datamap_z = datamap[:,:,fgslice_z,:]
            elif np.ndim(datamap)==5:  # tensor variable
                datamap_x = datamap[fgslice_x,:,:,:,:]
                datamap_y = datamap[:,fgslice_y,:,:,:]
                datamap_z = datamap[:,:,fgslice_z,:,:]
            else:
                logging.info("Error in reshaping fsgrid datamap!") 
            datamap_x = np.squeeze(datamap_x)
            datamap_x = np.swapaxes(datamap_x, 0,1)
            datamap_y = np.squeeze(datamap_y)
            datamap_y = np.swapaxes(datamap_y, 0,1)
            datamap_z = np.squeeze(datamap_z)
            datamap_z = np.swapaxes(datamap_z, 0,1)

        else:
            # vlasov grid, AMR
            datamap = datamap[indexids]      # sort
            if 'x' in slices: datamap_x = datamap[indexlist_x] # find required cells (X cut)
            if 'y' in slices: datamap_y = datamap[indexlist_y] # find required cells (Y cut)
            if 'z' in slices: datamap_z = datamap[indexlist_z] # find required cells (Z cut)
            # Create the plotting grid
            if np.ndim(datamap)==1:   # scalar variable
                if 'x' in slices: datamap_x = ids3d.idmesh3d(idlist_x, datamap_x, reflevel, xsize, ysize, zsize, 0, None)
                if 'y' in slices: datamap_y = ids3d.idmesh3d(idlist_y, datamap_y, reflevel, xsize, ysize, zsize, 1, None)
                if 'z' in slices: datamap_z = ids3d.idmesh3d(idlist_z, datamap_z, reflevel, xsize, ysize, zsize, 2, None)
            elif np.ndim(datamap)==2: # vector variable
                if 'x' in slices: datamap_x = ids3d.idmesh3d(idlist_x, datamap_x, reflevel, xsize, ysize, zsize, 0, datamap.shape[1])
                if 'y' in slices: datamap_y = ids3d.idmesh3d(idlist_y, datamap_y, reflevel, xsize, ysize, zsize, 1, datamap.shape[1])
                if 'z' in slices: datamap_z = ids3d.idmesh3d(idlist_z, datamap_z, reflevel, xsize, ysize, zsize, 2, datamap.shape[1])
            elif np.ndim(datamap)==3: # tensor variable
                if 'x' in slices: datamap_x = ids3d.idmesh3d(idlist_x, datamap_x, reflevel, xsize, ysize, zsize, 0, (datamap.shape[1],datamap.shape[2]))
                if 'y' in slices: datamap_y = ids3d.idmesh3d(idlist_y, datamap_y, reflevel, xsize, ysize, zsize, 1, (datamap.shape[1],datamap.shape[2]))
                if 'z' in slices: datamap_z = ids3d.idmesh3d(idlist_z, datamap_z, reflevel, xsize, ysize, zsize, 2, (datamap.shape[1],datamap.shape[2]))
            else:
                logging.info("Dimension error in constructing 2D AMR slice!")
                return -1

    else:
        # Expression set, use generated or provided colorbar title
        cb_title_use = expression.__name__ + operatorstr
        logging.info('WARNING: Expressions have not been implemented yet')

    # Now, if map is a vector or tensor, reduce it down
    if 'x' in slices:
        if np.ndim(datamap_x)==3: # vector
            if datamap_x.shape[2]!=3:
                logging.info("Error, expected array of 3-element vectors, found array of shape " + str(datamap_x.shape))
                return -1
            datamap_x = np.linalg.norm(datamap_x, axis=-1)
        if np.ndim(datamap_x)==4: # tensor
            if datamap_x.shape[2]!=3 or datamap_x.shape[3]!=3:
                # This may also catch 3D simulation fsgrid variables
                logging.info("Error, expected array of 3x3 tensors, found array of shape " + str(datamap_x.shape))
                return -1
            datamap_x = datamap_x[:,:,0,0]+datamap_x[:,:,1,1]+datamap_x[:,:,2,2]
        if np.ndim(datamap_x)>=5: # Too many dimensions
            logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap_x.shape))
            return -1
        if np.ndim(datamap_x)!=2: # Too many dimensions
            logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap_x.shape))
            return -1

        # Scale final generated datamap if requested
        datamap_x = datamap_x * vscale
        if (absolute):
            datamap_x = abs(datamap_x)

    if 'y' in slices:
        if np.ndim(datamap_y)==3: # vector
            if datamap_y.shape[2]!=3:
                logging.info("Error, expected array of 3-element vectors, found array of shape " + str(datamap_y.shape))
                return -1
            datamap_y = np.linalg.norm(datamap_y, axis=-1)
        if np.ndim(datamap_y)==4: # tensor
            if datamap_y.shape[2]!=3 or datamap_y.shape[3]!=3:
                # This may also catch 3D simulation fsgrid variables
                logging.info("Error, expected array of 3x3 tensors, found array of shape " + str(datamap_y.shape))
                return -1
            datamap_y = datamap_y[:,:,0,0]+datamap_y[:,:,1,1]+datamap_y[:,:,2,2]
        if np.ndim(datamap_y)>=5: # Too many dimensions
            logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap_y.shape))
            return -1
        if np.ndim(datamap_y)!=2: # Too many dimensions
            logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap_y.shape))
            return -1

        # Scale final generated datamap if requested
        datamap_y = datamap_y * vscale
        if (absolute):
            datamap_y = abs(datamap_y)

    if 'z' in slices:
        if np.ndim(datamap_z)==3: # vector
            if datamap_z.shape[2]!=3:
                logging.info("Error, expected array of 3-element vectors, found array of shape " + str(datamap_z.shape))
                return -1
            datamap_z = np.linalg.norm(datamap_z, axis=-1)
        if np.ndim(datamap_z)==4: # tensor
            if datamap_z.shape[2]!=3 or datamap_z.shape[3]!=3:
                # This may also catch 3D simulation fsgrid variables
                logging.info("Error, expected array of 3x3 tensors, found array of shape " + str(datamap_z.shape))
                return -1
            datamap_z = datamap_z[:,:,0,0]+datamap_z[:,:,1,1]+datamap_z[:,:,2,2]
        if np.ndim(datamap_z)>=5: # Too many dimensions
            logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap_z.shape))
            return -1
        if np.ndim(datamap_z)!=2: # Too many dimensions
            logging.info("Error, too many dimensions in datamap, found array of shape " + str(datamap_z.shape))
            return -1

        # Scale final generated datamap if requested
        datamap_z = datamap_z * vscale
        if (absolute):
            datamap_z = abs(datamap_z)

    # scale the sizes to the heighest refinement level because
    # plotting is done at that level
    sizes[0] = int(sizes[0]*2**reflevel)
    sizes[1] = int(sizes[1]*2**reflevel)
    sizes[2] = int(sizes[2]*2**reflevel)
    finecellsize = cellsize / 2**reflevel

    # Allow title override
    if cbtitle is not None:
        # Here allow underscores for manual math mode
        cb_title_use = cbtitle

    # If automatic range finding is required, find min and max of array
    # Performs range-finding on a masked array to work even if array contains invalid values
    if vmin is not None:
        vminuse=vmin
    else: 
        vminuse = np.amax(datamap)
        if 'x' in slices: vminuse = min(vminuse,np.ma.amin(datamap_x))
        if 'y' in slices: vminuse = min(vminuse,np.ma.amin(datamap_y))
        if 'z' in slices: vminuse = min(vminuse,np.ma.amin(datamap_z))
    if vmax is not None:
        vmaxuse=vmax
    else:
        vmaxuse = np.amin(datamap)
        if 'x' in slices: vmaxuse = max(vmaxuse,np.ma.amax(datamap_x))
        if 'y' in slices: vmaxuse = max(vmaxuse,np.ma.amax(datamap_y))
        if 'z' in slices: vmaxuse = max(vmaxuse,np.ma.amax(datamap_z))

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
        vminuse = np.amax(datamap)
        if 'x' in slices: vminuse = min(vminuse,np.ma.amin(np.ma.masked_less_equal(datamap_x,0)))
        if 'y' in slices: vminuse = min(vminuse,np.ma.amin(np.ma.masked_less_equal(datamap_y,0)))
        if 'z' in slices: vminuse = min(vminuse,np.ma.amin(np.ma.masked_less_equal(datamap_z,0)))

    # Special case of very small vminuse values
    if ((vmin is None) or (vmax is None)) and (vminuse > 0) and (vminuse < vmaxuse*1.e-5):
        vminuse = vmaxuse*1e-5
        if lin is not None:
            vminuse = 0
    
    # If symlog scaling is set:
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
                #TODO: copy over matplotlib 3.3.0 implementation of SymLogNorm into pytools/analysator
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
        ticks = np.linspace(vminuse,vmaxuse,num=7)

    # Create the scalar mappable to define the face colouring of the surface elements
    scamap = plt.cm.ScalarMappable(cmap=colormap,norm=norm)


    ###############################################################################
    # Making the 12 meshes corresponding to the 12 elementary surfaces to plot #
    ###############################################################################
    cutpointaxu = cutpoint/axisunituse
    # {X = x0 slice}
    [YmeshYmZm,ZmeshYmZm] = np.meshgrid(np.linspace(simext[2],cutpointaxu[1],num=int(round((cutpoint[1]-ymin)/finecellsize))+1),
                               np.linspace(simext[4],cutpointaxu[2],num=int(round((cutpoint[2]-zmin)/finecellsize))+1))
    XmeshYmZm = np.ones(YmeshYmZm.shape) * cutpointaxu[0]

    [YmeshYpZm,ZmeshYpZm] = np.meshgrid(np.linspace(cutpointaxu[1],simext[3],num=int(round((ymax-cutpoint[1])/finecellsize))+1),
                               np.linspace(simext[4],cutpointaxu[2],num=int(round((cutpoint[2]-zmin)/finecellsize))+1))
    XmeshYpZm = np.ones(YmeshYpZm.shape) * cutpointaxu[0]

    [YmeshYmZp,ZmeshYmZp] = np.meshgrid(np.linspace(simext[2],cutpointaxu[1],num=int(round((cutpoint[1]-ymin)/finecellsize))+1),
                               np.linspace(cutpointaxu[2],simext[5],num=int(round((zmax-cutpoint[2])/finecellsize))+1))
    XmeshYmZp = np.ones(YmeshYmZp.shape) * cutpointaxu[0]

    [YmeshYpZp,ZmeshYpZp] = np.meshgrid(np.linspace(cutpointaxu[1],simext[3],num=int(round((ymax-cutpoint[1])/finecellsize))+1),
                               np.linspace(cutpointaxu[2],simext[5],num=int(round((zmax-cutpoint[2])/finecellsize))+1))
    XmeshYpZp = np.ones(YmeshYpZp.shape) * cutpointaxu[0]

    # {Y = y0 slice}
    [XmeshXmZm,ZmeshXmZm] = np.meshgrid(np.linspace(simext[0],cutpointaxu[0],num=int(round((cutpoint[0]-xmin)/finecellsize))+1),
                               np.linspace(simext[4],cutpointaxu[2],num=int(round((cutpoint[2]-zmin)/finecellsize))+1))
    YmeshXmZm = np.ones(XmeshXmZm.shape) * cutpointaxu[1]

    [XmeshXpZm,ZmeshXpZm] = np.meshgrid(np.linspace(cutpointaxu[0],simext[1],num=int(round((xmax-cutpoint[0])/finecellsize))+1),
                               np.linspace(simext[4],cutpointaxu[2],num=int(round((cutpoint[2]-zmin)/finecellsize))+1))
    YmeshXpZm = np.ones(XmeshXpZm.shape) * cutpointaxu[1]

    [XmeshXmZp,ZmeshXmZp] = np.meshgrid(np.linspace(simext[0],cutpointaxu[0],num=int(round((cutpoint[0]-xmin)/finecellsize))+1),
                               np.linspace(cutpointaxu[2],simext[5],num=int(round((zmax-cutpoint[2])/finecellsize))+1))
    YmeshXmZp = np.ones(XmeshXmZp.shape) * cutpointaxu[1]

    [XmeshXpZp,ZmeshXpZp] = np.meshgrid(np.linspace(cutpointaxu[0],simext[1],num=int(round((xmax-cutpoint[0])/finecellsize))+1),
                               np.linspace(cutpointaxu[2],simext[5],num=int(round((zmax-cutpoint[2])/finecellsize))+1))
    YmeshXpZp = np.ones(XmeshXpZp.shape) * cutpointaxu[1]

    # {Z = z0 slice}
    [XmeshXmYm,YmeshXmYm] = np.meshgrid(np.linspace(simext[0],cutpointaxu[0],num=int(round((cutpoint[0]-xmin)/finecellsize))+1),
                               np.linspace(simext[2],cutpointaxu[1],num=int(round((cutpoint[1]-ymin)/finecellsize))+1))
    ZmeshXmYm = np.ones(XmeshXmYm.shape) * cutpointaxu[2]

    [XmeshXpYm,YmeshXpYm] = np.meshgrid(np.linspace(cutpointaxu[0],simext[1],num=int(round((xmax-cutpoint[0])/finecellsize))+1),
                               np.linspace(simext[2],cutpointaxu[1],num=int(round((cutpoint[1]-ymin)/finecellsize))+1))
    ZmeshXpYm = np.ones(XmeshXpYm.shape) * cutpointaxu[2]

    [XmeshXmYp,YmeshXmYp] = np.meshgrid(np.linspace(simext[0],cutpointaxu[0],num=int(round((cutpoint[0]-xmin)/finecellsize))+1),
                               np.linspace(cutpointaxu[1],simext[3],num=int(round((ymax-cutpoint[1])/finecellsize))+1))
    ZmeshXmYp = np.ones(XmeshXmYp.shape) * cutpointaxu[2]

    [XmeshXpYp,YmeshXpYp] = np.meshgrid(np.linspace(cutpointaxu[0],simext[1],num=int(round((xmax-cutpoint[0])/finecellsize))+1),
                               np.linspace(cutpointaxu[1],simext[3],num=int(round((ymax-cutpoint[1])/finecellsize))+1))
    ZmeshXpYp = np.ones(XmeshXpYp.shape) * cutpointaxu[2]

    # Creating lists of meshes to be called in a for loop
    Xmesh_list = [XmeshYmZm,XmeshYpZm,XmeshYmZp,XmeshYpZp, XmeshXmZm,XmeshXpZm,XmeshXmZp,XmeshXpZp, XmeshXmYm,XmeshXpYm,XmeshXmYp,XmeshXpYp]
    Ymesh_list = [YmeshYmZm,YmeshYpZm,YmeshYmZp,YmeshYpZp, YmeshXmZm,YmeshXpZm,YmeshXmZp,YmeshXpZp, YmeshXmYm,YmeshXpYm,YmeshXmYp,YmeshXpYp]
    Zmesh_list = [ZmeshYmZm,ZmeshYpZm,ZmeshYmZp,ZmeshYpZp, ZmeshXmZm,ZmeshXpZm,ZmeshXmZp,ZmeshXpZp, ZmeshXmYm,ZmeshXpYm,ZmeshXmYp,ZmeshXpYp]

    # coordinates of the point where the all three 2d cut throughs cuts
    xr_coord = -xmin + cutpoint[0]
    yr_coord = -ymin + cutpoint[1]
    zr_coord = -zmin + cutpoint[2]

    # coordinates of the point where the all three 2d cut throughs cut in terms of cells
    xr = int(round((xr_coord/(xmax - xmin))*xsize*2**reflevel))
    yr = int(round((yr_coord/(ymax - ymin))*ysize*2**reflevel))
    zr = int(round((zr_coord/(zmax - zmin))*zsize*2**reflevel))

    # Creating lists of datamap_i to be called in that same for loop
    if 'x' in slices: datamap_x_list = [datamap_x[:zr,:yr],datamap_x[:zr,yr:],datamap_x[zr:,:yr],datamap_x[zr:,yr:]]
    if 'y' in slices: datamap_y_list = [datamap_y[:zr,:xr],datamap_y[:zr,xr:],datamap_y[zr:,:xr],datamap_y[zr:,xr:]]
    if 'z' in slices: datamap_z_list = [datamap_z[:yr,:xr],datamap_z[:yr,xr:],datamap_z[yr:,:xr],datamap_z[yr:,xr:]]

    if not nomask:
        # Creating lists of rhomap_i for the same purpose (especially in case the box does not extend to the full domain)
        if 'x' in slices: rhomap_x_list = [rhomap_x[:zr,:yr],rhomap_x[:zr,yr:],rhomap_x[zr:,:yr],rhomap_x[zr:,yr:]]
        if 'y' in slices: rhomap_y_list = [rhomap_y[:zr,:xr],rhomap_y[:zr,xr:],rhomap_y[zr:,:xr],rhomap_y[zr:,xr:]]
        if 'z' in slices: rhomap_z_list = [rhomap_z[:yr,:xr],rhomap_z[:yr,xr:],rhomap_z[yr:,:xr],rhomap_z[yr:,xr:]]

    # Creating a new figure and a 3d axes with a custom 3d coordinate axes 
    figsize = (6,5)
    if nocb:
        figsize = (5,5)
    fig = plt.figure(figsize=figsize,dpi=150)
    ax1 = axes3d(fig, reflevel, cutpoint, boxcoords, axisunit, axisunituse, tickinterval, fixedticks, scale,
                 thick, axiscolor, viewangle, halfaxes, slices, Earth)

    # Masking and plotting the elementary surfaces one by one (actually three by three)
    for i in range(0,4):
        XmeshYZ = Xmesh_list[i]
        YmeshYZ = Ymesh_list[i]
        ZmeshYZ = Zmesh_list[i]

        XmeshXZ = Xmesh_list[i+4]
        YmeshXZ = Ymesh_list[i+4]
        ZmeshXZ = Zmesh_list[i+4]

        XmeshXY = Xmesh_list[i+8]
        YmeshXY = Ymesh_list[i+8]
        ZmeshXY = Zmesh_list[i+8]

        if 'x' in slices: datamap_x_i = datamap_x_list[i]
        if 'y' in slices: datamap_y_i = datamap_y_list[i]
        if 'z' in slices: datamap_z_i = datamap_z_list[i]

        if not nomask:
            if 'x' in slices: rhomap_x_i = rhomap_x_list[i]
            if 'y' in slices: rhomap_y_i = rhomap_y_list[i]
            if 'z' in slices: rhomap_z_i = rhomap_z_list[i]

        # The grid generated by meshgrid has all four corners for each cell.
        # We mask using only the centre values.
        # Calculate offsets for cell-centre coordinates
        XmeshXYCentres = XmeshXY[:-1,:-1] + 0.5*finecellsize/axisunituse
        YmeshXYCentres = YmeshXY[:-1,:-1] + 0.5*finecellsize/axisunituse

        XmeshXZCentres = XmeshXZ[:-1,:-1] + 0.5*finecellsize/axisunituse
        ZmeshXZCentres = ZmeshXZ[:-1,:-1] + 0.5*finecellsize/axisunituse

        YmeshYZCentres = YmeshYZ[:-1,:-1] + 0.5*finecellsize/axisunituse
        ZmeshYZCentres = ZmeshYZ[:-1,:-1] + 0.5*finecellsize/axisunituse

        maskgrid_XY = np.ma.array(XmeshXYCentres)
        maskgrid_XZ = np.ma.array(XmeshXZCentres)
        maskgrid_YZ = np.ma.array(YmeshYZCentres)
        if not pass_full:
            # If zoomed-in using a defined box, and not specifically asking to pass all values:
            # Generate mask for only visible section (with small buffer for e.g. gradient calculations)
            maskboundarybuffer = 2.*finecellsize/axisunituse

            maskgrid_XY = np.ma.masked_where(XmeshXYCentres<(boxcoords[0]-maskboundarybuffer), maskgrid_XY)
            maskgrid_XY = np.ma.masked_where(XmeshXYCentres>(boxcoords[1]+maskboundarybuffer), maskgrid_XY)
            maskgrid_XY = np.ma.masked_where(YmeshXYCentres<(boxcoords[2]-maskboundarybuffer), maskgrid_XY)
            maskgrid_XY = np.ma.masked_where(YmeshXYCentres>(boxcoords[3]+maskboundarybuffer), maskgrid_XY)

            maskgrid_XZ = np.ma.masked_where(XmeshXZCentres<(boxcoords[0]-maskboundarybuffer), maskgrid_XZ)
            maskgrid_XZ = np.ma.masked_where(XmeshXZCentres>(boxcoords[1]+maskboundarybuffer), maskgrid_XZ)
            maskgrid_XZ = np.ma.masked_where(ZmeshXZCentres<(boxcoords[4]-maskboundarybuffer), maskgrid_XZ)
            maskgrid_XZ = np.ma.masked_where(ZmeshXZCentres>(boxcoords[5]+maskboundarybuffer), maskgrid_XZ)

            maskgrid_YZ = np.ma.masked_where(YmeshYZCentres<(boxcoords[2]-maskboundarybuffer), maskgrid_YZ)
            maskgrid_YZ = np.ma.masked_where(YmeshYZCentres>(boxcoords[3]+maskboundarybuffer), maskgrid_YZ)
            maskgrid_YZ = np.ma.masked_where(ZmeshYZCentres<(boxcoords[4]-maskboundarybuffer), maskgrid_YZ)
            maskgrid_YZ = np.ma.masked_where(ZmeshYZCentres>(boxcoords[5]+maskboundarybuffer), maskgrid_YZ)

        if np.ma.is_masked(maskgrid_XY) and ('z' in slices) and not np.all(maskgrid_XY.mask):
            # Save lists for masking
            MaskXY_X = np.where(~np.all(maskgrid_XY.mask, axis=1))[0] # [0] takes the first element of a tuple
            MaskXY_Y = np.where(~np.all(maskgrid_XY.mask, axis=0))[0]
            XmeshXYPass = XmeshXY[MaskXY_X[0]:MaskXY_X[-1]+2,:]
            XmeshXYPass = XmeshXYPass[:,MaskXY_Y[0]:MaskXY_Y[-1]+2]
            YmeshXYPass = YmeshXY[MaskXY_X[0]:MaskXY_X[-1]+2,:]
            YmeshXYPass = YmeshXYPass[:,MaskXY_Y[0]:MaskXY_Y[-1]+2]
            ZmeshXYPass = ZmeshXY[MaskXY_X[0]:MaskXY_X[-1]+2,:]
            ZmeshXYPass = ZmeshXYPass[:,MaskXY_Y[0]:MaskXY_Y[-1]+2]
        else:
            XmeshXYPass = np.ma.array(XmeshXY)
            YmeshXYPass = np.ma.array(YmeshXY)
            ZmeshXYPass = np.ma.array(ZmeshXY)

        if np.ma.is_masked(maskgrid_XZ) and ('y' in slices) and not np.all(maskgrid_XZ.mask):
            # Save lists for masking
            MaskXZ_X = np.where(~np.all(maskgrid_XZ.mask, axis=1))[0] # [0] takes the first element of a tuple
            MaskXZ_Z = np.where(~np.all(maskgrid_XZ.mask, axis=0))[0]
            XmeshXZPass = XmeshXZ[MaskXZ_X[0]:MaskXZ_X[-1]+2,:]
            XmeshXZPass = XmeshXZPass[:,MaskXZ_Z[0]:MaskXZ_Z[-1]+2]
            YmeshXZPass = YmeshXZ[MaskXZ_X[0]:MaskXZ_X[-1]+2,:]
            YmeshXZPass = YmeshXZPass[:,MaskXZ_Z[0]:MaskXZ_Z[-1]+2]
            ZmeshXZPass = ZmeshXZ[MaskXZ_X[0]:MaskXZ_X[-1]+2,:]
            ZmeshXZPass = ZmeshXZPass[:,MaskXZ_Z[0]:MaskXZ_Z[-1]+2]
        else:
            XmeshXZPass = np.ma.array(XmeshXZ)
            YmeshXZPass = np.ma.array(YmeshXZ)
            ZmeshXZPass = np.ma.array(ZmeshXZ)

        if np.ma.is_masked(maskgrid_YZ) and ('x' in slices) and not np.all(maskgrid_YZ.mask):
            # Save lists for masking
            MaskYZ_Y = np.where(~np.all(maskgrid_YZ.mask, axis=1))[0] # [0] takes the first element of a tuple
            MaskYZ_Z = np.where(~np.all(maskgrid_YZ.mask, axis=0))[0]
            XmeshYZPass = XmeshYZ[MaskYZ_Y[0]:MaskYZ_Y[-1]+2,:]
            XmeshYZPass = XmeshYZPass[:,MaskYZ_Z[0]:MaskYZ_Z[-1]+2]
            YmeshYZPass = YmeshYZ[MaskYZ_Y[0]:MaskYZ_Y[-1]+2,:]
            YmeshYZPass = YmeshYZPass[:,MaskYZ_Z[0]:MaskYZ_Z[-1]+2]
            ZmeshYZPass = ZmeshYZ[MaskYZ_Y[0]:MaskYZ_Y[-1]+2,:]
            ZmeshYZPass = ZmeshYZPass[:,MaskYZ_Z[0]:MaskYZ_Z[-1]+2]
        else:
            XmeshYZPass = np.ma.array(XmeshYZ)
            YmeshYZPass = np.ma.array(YmeshYZ)
            ZmeshYZPass = np.ma.array(ZmeshYZ)

        # Crop both rhomap and datamap to view region
        if np.ma.is_masked(maskgrid_XY) and ('z' in slices) and not np.all(maskgrid_XY.mask):
            if not nomask:
                # Strip away columns and rows which are outside the plot region
                rhomap_z_i = rhomap_z_i[MaskXY_X[0]:MaskXY_X[-1]+1,:]
                rhomap_z_i = rhomap_z_i[:,MaskXY_Y[0]:MaskXY_Y[-1]+1]
            # Also for the datamap, unless it was already provided by an expression
            if not expression:
                datamap_z_i = datamap_z_i[MaskXY_X[0]:MaskXY_X[-1]+1,:]
                datamap_z_i = datamap_z_i[:,MaskXY_Y[0]:MaskXY_Y[-1]+1]

        if np.ma.is_masked(maskgrid_XZ) and ('y' in slices) and not np.all(maskgrid_XZ.mask):
            if not nomask:
                # Strip away columns and rows which are outside the plot region
                rhomap_y_i = rhomap_y_i[MaskXZ_X[0]:MaskXZ_X[-1]+1,:]
                rhomap_y_i = rhomap_y_i[:,MaskXZ_Z[0]:MaskXZ_Z[-1]+1]
            # Also for the datamap, unless it was already provided by an expression
            if not expression:
                datamap_y_i = datamap_y_i[MaskXZ_X[0]:MaskXZ_X[-1]+1,:]
                datamap_y_i = datamap_y_i[:,MaskXZ_Z[0]:MaskXZ_Z[-1]+1]

        if np.ma.is_masked(maskgrid_YZ) and ('x' in slices) and not np.all(maskgrid_YZ.mask):
            if not nomask:
                # Strip away columns and rows which are outside the plot region
                rhomap_x_i = rhomap_x_i[MaskYZ_Y[0]:MaskYZ_Y[-1]+1,:]
                rhomap_x_i = rhomap_x_i[:,MaskYZ_Z[0]:MaskYZ_Z[-1]+1]
            # Also for the datamap, unless it was already provided by an expression
            if not expression:
                datamap_x_i = datamap_x_i[MaskYZ_Y[0]:MaskYZ_Y[-1]+1,:]
                datamap_x_i = datamap_x_i[:,MaskYZ_Z[0]:MaskYZ_Z[-1]+1]

        # Mask region outside ionosphere. Note that for some boundary layer cells, 
        # a density is calculated, but e.g. pressure is not, and these cells aren't
        # excluded by this method. Also mask away regions where datamap is invalid
        if 'z' in slices:
            if not nomask:
                rhomap_z_i = np.ma.masked_less_equal(np.ma.masked_invalid(rhomap_z_i), 0)
                rhomap_z_i = np.ma.masked_where(~np.isfinite(datamap_z_i), rhomap_z_i)
                XYmask_z = rhomap_z_i.mask
                if XYmask_z.any():
                    if XYmask_z.all():
                        # if everything was masked in rhomap, allow plotting
                        XYmask_z[:,:] = False
                    else:
                        # Mask datamap
                        datamap_z_i = np.ma.array(datamap_z_i, mask=XYmask_z)
            else:
                # no masking
                XYmask_z = np.full(datamap_z_i.shape, False)
                        
            # Building the face colour maps for plot_surface()
            if np.ma.isMaskedArray(datamap_z_i):
                fcolor_z_i = scamap.to_rgba(datamap_z_i.data)
                fcolor_z_i[XYmask_z] = np.array([0,0,0,0])
            else:
                fcolor_z_i = scamap.to_rgba(datamap_z_i)

        if 'y' in slices:
            if not nomask:
                rhomap_y_i = np.ma.masked_less_equal(np.ma.masked_invalid(rhomap_y_i), 0)
                rhomap_y_i = np.ma.masked_where(~np.isfinite(datamap_y_i), rhomap_y_i)
                XZmask_y = rhomap_y_i.mask
                if XZmask_y.any():
                    if XZmask_y.all():
                        # if everything was masked in rhomap, allow plotting
                        XZmask_y[:,:] = False
                    else:
                        # Mask datamap
                        datamap_y_i = np.ma.array(datamap_y_i, mask=XZmask_y)
            else:
                # no masking
                XZmask_y = np.full(datamap_y_i.shape, False)                        
            # Building the face colour maps for plot_surface()
            if np.ma.isMaskedArray(datamap_y_i):
                fcolor_y_i = scamap.to_rgba(datamap_y_i.data)
                fcolor_y_i[XZmask_y] = np.array([0,0,0,0])
            else:
                fcolor_y_i = scamap.to_rgba(datamap_y_i)

        if 'x' in slices:
            if not nomask:
                rhomap_x_i = np.ma.masked_less_equal(np.ma.masked_invalid(rhomap_x_i), 0)
                rhomap_x_i = np.ma.masked_where(~np.isfinite(datamap_x_i), rhomap_x_i)
                YZmask_x = rhomap_x_i.mask
                if YZmask_x.any():
                    if YZmask_x.all():
                        # if everything was masked in rhomap, allow plotting
                        YZmask_x[:,:] = False
                    else:
                        # Mask datamap
                        datamap_x_i = np.ma.array(datamap_x_i, mask=YZmask_x)
            else:
                # no masking
                YZmask_x = np.full(datamap_x_i.shape, False)

            # Building the face colour maps for plot_surface()
            if np.ma.isMaskedArray(datamap_x_i):
                fcolor_x_i = scamap.to_rgba(datamap_x_i.data)
                fcolor_x_i[YZmask_x] = np.array([0,0,0,0])
            else:
                fcolor_x_i = scamap.to_rgba(datamap_x_i)

        # Introducing a slight shading to better distiguish the planes
        if shadings is not None:
            (shadx,shady,shadz) = shadings
        else:
            (azi,ele) = viewangle
            deg2rad = np.pi/180.
            shadx = abs(np.cos(ele*deg2rad) * np.cos(azi*deg2rad))
            shady = abs(np.cos(ele*deg2rad) * np.sin(azi*deg2rad))
            shadz = abs(np.sin(ele*deg2rad))
            maxshad = max(shadx,shady,shadz)
            shadx = shadx / maxshad
            shady = shady / maxshad
            shadz = shadz / maxshad

        # Plotting the partial {X = x0} cut
        if 'x' in slices:
            ax1.plot_surface(XmeshYZPass, YmeshYZPass, ZmeshYZPass, rstride=1, cstride=1,
                        facecolors=shadx*fcolor_x_i, shade=False, antialiased=False)

        # Plotting the partial {Y = y0} cut
        if 'y' in slices:
            ax1.plot_surface(XmeshXZPass, YmeshXZPass, ZmeshXZPass, rstride=1, cstride=1,
                        facecolors=shady*fcolor_y_i, shade=False, antialiased=False)

        # Plotting the partial {Z = z0} cut
        if 'z' in slices:
            ax1.plot_surface(XmeshXYPass, YmeshXYPass, ZmeshXYPass, rstride=1, cstride=1,
                        facecolors=shadz*fcolor_z_i, shade=False, antialiased=False)

#******

    # Plot title - adding the cut point information and tick interval length
    # unless title was set manually
    if title is None or title=="msec" or title=="musec":        
        if np.all(np.isclose(cutpoint/axisunituse % 1,0.)):
            plot_title = plot_title + pt.plot.mathmode('-') + ' origin at ({:,.0f}, {:,.0f}, {:,.0f}) [{:s}]'.format(
                         cutpoint[0]/axisunituse,cutpoint[1]/axisunituse,cutpoint[2]/axisunituse,pt.plot.mathmode(pt.plot.bfstring(axisunitstr)))
        else:
            plot_title = plot_title + pt.plot.mathmode('-') + ' origin at ({:,.1f}, {:,.1f}, {:,.1f}) [{:s}]'.format(
                         cutpoint[0]/axisunituse,cutpoint[1]/axisunituse,cutpoint[2]/axisunituse,pt.plot.mathmode(pt.plot.bfstring(axisunitstr)))
        if not fixedticks:
            tickinfostr = 'Tick every {:,.0f} {:s}'.format(tickinterval/axisunituse,pt.plot.mathmode(pt.plot.bfstring(axisunitstr)))
        else:
            tickinfostr = 'Ticks at multiples of {:,.0f} {:s}'.format(tickinterval/axisunituse,pt.plot.mathmode(pt.plot.bfstring(axisunitstr)))

    #    plot_title = pt.plot.mathmode(pt.plot.bfstring(plot_title)) + '\n' + pt.plot.mathmode(pt.plot.bfstring(tickinfostr))
        plot_title = pt.plot.textbfstring(plot_title) + '\n' + pt.plot.textbfstring(tickinfostr)
    ax1.set_title(plot_title,fontsize=fontsize2,fontweight='bold',position=(0.5,0.85))

    # Colourbar title
    if len(cb_title_use)!=0:
        cb_title_use = pt.plot.mathmode(pt.plot.bfstring(cb_title_use))

    # Set flag which affects colorbar decimal precision
    if lin is None:
        pt.plot.cb_linear = False
    else:
        pt.plot.cb_linear = True

    # Creating colorbar axes
    if not nocb:
        cax = fig.add_axes([0.76,0.2,0.03,0.6])
        cbdir="right"; horalign="left"

        # First draw colorbar
        if usesci:
            cb = plt.colorbar(scamap, ticks=ticks, format=mtick.FuncFormatter(pt.plot.cbfmtsci), cax=cax, drawedges=False)
        else:
            cb = plt.colorbar(scamap, ticks=ticks, format=mtick.FuncFormatter(pt.plot.cbfmt), cax=cax, drawedges=False)
        cb.outline.set_linewidth(thick)
        cb.ax.yaxis.set_ticks_position(cbdir)
        # Ensure minor tick marks are off
        if lin is not None:
            cb.minorticks_off()
 
        cbticks = cb.get_ticks()
        cb.set_ticks(cbticks[(cbticks>=vminuse)*(cbticks<=vmaxuse)])

        cb.ax.tick_params(labelsize=fontsize3,width=thick,length=3*thick)
        cb_title = cax.set_title(cb_title_use,fontsize=fontsize3,fontweight='bold', horizontalalignment=horalign)
        cb_title.set_position((0.,1.+0.025*scale)) # avoids having colourbar title too low when fontsize is increased

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
            nlabels = len(cb.ax.yaxis.get_ticklabels()) # TODO implement some kind of ratio like in other scripts, if needed?
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
    if (wmark or wmarkb):
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

    
    # Adjust layout. Uses tight_layout() but in fact this ensures 
    # that long titles and tick labels are still within the plot area.
    plt.tight_layout(pad=0.01) 
    # TODO check: a warning says tight_layout() might not be compatible with those axes. Seems to work though...
    savefig_pad=0.01
    bbox_inches='tight'

    # Save output or draw on-screen
    if not draw:
        logging.info('Saving the figure as {}, Time since start = {:.2f} s'.format(outputfile,time.time()-t0))
        try:
            plt.savefig(outputfile,dpi=300, bbox_inches=bbox_inches, pad_inches=savefig_pad)
        except:
            logging.info("Error with attempting to save figure.")
            logging.info('...Done! Time since start = {:.2f} s'.format(time.time()-t0))
        plt.close()
    else:
        # Draw on-screen
        plt.draw()
        plt.show()
        logging.info('Draw complete! Time since start = {:.2f} s'.format(time.time()-t0))
