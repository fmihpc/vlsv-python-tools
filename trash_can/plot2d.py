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

# 2-D pseudocolor plotter
# USAGE: python plot2d.py NCores vlsvFileNumberStart vlsvFileNumberEnd
#  NCores = number of multithreading cores
#  vlsvFileNumberStart = number of the first plotted VLSV file
#  vlsvFileNumberend   = number of the last plotted VLSV file
# The script assumes the following file name formats (Xs are integers):
#  VLSV files: bulk.XXXXXXX.vlsv (vlsvFolder)
#  Flux files: bulk.XXXXXXX.bin (fluxFunctionFolder)

import sys
import os
import matplotlib
matplotlib.use('Agg') # this is needed for offline plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import analysator as pt
import numpy as np
import operator as oper
import re
from matplotlib.patches import Wedge

# font size
matplotlib.rcParams.update({'font.size': 40})

# unit of length = Earth radius
Re = 6371e3

# VLSV file folder 
vlsvFolder = '/b/vlasiator/2D/BCH/bulk/'

# radius of inner boundary
innerBoundaryR = 30000e3

# plotting variable
vlsvVar = 'rho'
colorMapLog = 1
colorMapLims = (1e5,1e7)
colorMapName = 'seismic'

# axes tick marks
ticksXAxis = np.linspace(-100,100,21)*Re
ticksYAxis = np.linspace(-100,100,21)*Re

# plot area limits
plotLimits = 0
areaXLims = [-20*Re,10*Re]
areaYLims = [0,30*Re]
#ticksXAxis = np.linspace(-20,10,7)*Re
#ticksYAxis = np.linspace(0,30,7)*Re

# output file
outFileSize = (36,22)
outFilePrefix = 'fig2d_'
outFileExtension = '.png'

# plot or remove tail region defined by a conic section model
removeTail = 1
csTail = (3*Re,1.000001,-3*Re,np.pi/1.12)

# plot field lines from flux function files
plotFieldLines = 1
fluxFunctionFolder = '/b/vlasiator/2D/BCH/flux/v1/'
fieldLineWidth = 2.0
fieldLineColor = 'k'
fieldLineStyle = '-'
fieldLineAlpha = 0.5
# correction factors for old flux function files
ffBzIMF = -5e-9
ffVsw = -750e3
# choose plotted field lines as flux function levels
ffLevels = np.arange(-20.0,20.0,0.2)

# plot field lines using pyplot streamplot
plotFieldLinesPyplot = 0
fieldLineDensity = 4
fieldLineArrowSize = 3.0

# plots list of points using their cell ids
plotCids = 0
cids = (4502051,4951951,5551701)
cidsPlotStyle = 'ko'
cidsMarkerSize = 18
cidsAlpha = 0.5

# configure command line arguments
if not (len(sys.argv) == 4):
 print('three command line arguments required')
 quit()
Ncores = -100
vlsvFileNumberStart = -100
vlsvFileNumberEnd = -100
try:
 Ncores = int(sys.argv[1])
 vlsvFileNumberStart = int(sys.argv[2])
 vlsvFileNumberEnd = int(sys.argv[3])
except ValueError:
 print('ERROR: arg1 = ' + str(sys.argv[1]) + ', arg2 = ' + str(sys.argv[2]) + ', arg3 = ' + str(sys.argv[3]))
 quit()
if (Ncores < 1) or (Ncores > 40):
 print('ERROR: negative or otherwise bad number of cores')
 quit() 
if (vlsvFileNumberStart < 0) or (vlsvFileNumberEnd < 0) or (vlsvFileNumberStart > vlsvFileNumberEnd):
 print('ERROR: negative or otherwise bad start or end file numbers')
 quit()
print('running on ' + str(Ncores) + ' cores')
 
# round numbers to string
def round2str(x):
 return str(round(x*10)/10)

# conic section model
def getConicSectionRadius(cs,x,z):
 L = cs[0]
 eps = cs[1]
 Xo = cs[2]
 theta = np.arctan2(np.abs(z),x)
 r = L/( 1 + eps*np.cos(theta) )
 x = Xo + r*np.cos(theta)
 y = r*np.sin(theta)
 r = np.sqrt(x*x + y*y)
 return r

# Sort variable array returned by VlsvReader for 2d plotting
def sortVlsvReaderArray(D,locsSorted,nx,ny):
 DSorted = []
 for i in locsSorted:
  DSorted.append(D[i[1]])
 return np.asarray(DSorted).reshape(ny,nx)

# 2-d pseudocolor plot
def plotAndSave2D(vlsvReader,vlsvVar,nx,ny,nz,locsSorted,xmin,ymin,zmin,xmax,ymax,zmax,tSim,fluxFile,outFileName):
 # load variable and create arrays
 D = vlsvReader.read_variable(vlsvVar)
 # determine 2d simulation plane
 plane = 'xyz'
 if(ny > nz and nz == 1):
  plane = 'xy'
 elif(ny < nz and ny == 1):
  plane = 'xz'
 else:
  print('ERROR: cannot determine simulation plane')
  return
 D = 0
 meshX = 0
 meshY = 0
 if plane == 'xy':
  D = sortVlsvReaderArray(vlsvReader.read_variable(vlsvVar),locsSorted,nx,ny)
  meshX,meshY = np.meshgrid(np.linspace(xmin/Re,xmax/Re,nx),np.linspace(ymin/Re,ymax/Re,ny))
 elif plane == 'xz':
  D = sortVlsvReaderArray(vlsvReader.read_variable(vlsvVar),locsSorted,nx,nz)
  meshX,meshY = np.meshgrid(np.linspace(xmin/Re,xmax/Re,nx),np.linspace(zmin/Re,zmax/Re,nz))
 if removeTail == 1:
  meshR = np.sqrt(meshX*meshX + meshY*meshY)*Re
  meshR_tail = getConicSectionRadius(csTail,meshX*Re,meshY*Re)
  D[(meshR < meshR_tail)] = np.nan
  D = np.ma.masked_invalid(D) # note this will not be needed in matplotlib > 2.1
 plt.figure(figsize=outFileSize)
 if colorMapLog == 1:
  plt.pcolormesh(meshX,meshY,D,cmap=colorMapName,norm=colors.LogNorm(vmin=colorMapLims[0],vmax=colorMapLims[1]))
 else:
  plt.pcolormesh(meshX,meshY,D,cmap=colorMapName,vmin=colorMapLims[0],vmax=colorMapLims[1])
 plt.colorbar()
 if plotFieldLines == 1:
  try:
   with open(fluxFile) as fp:
    fluxFunction = np.fromfile(fluxFile,dtype='double').reshape(nz,nx)
    fluxFunction = fluxFunction - tSim*ffBzIMF*ffVsw
    meshR = np.sqrt(meshX**2 + meshY**2)
    fluxFunction[meshR <= innerBoundaryR/Re] = 0
    if removeTail == 1:
     meshR = np.sqrt(meshX*meshX + meshY*meshY)*Re
     meshR_tail = getConicSectionRadius(csTail,meshX*Re,meshY*Re)
     fluxFunction[(meshR < meshR_tail)] = 0
    plt.contour(meshX,meshY,fluxFunction,ffLevels,colors=(fieldLineColor),linestyles=(fieldLineStyle),linewidths=fieldLineWidth,alpha=fieldLineAlpha)
  except IOError as err:
   print('cannot open and skipping flux file: ' + fluxFile)
   fluxFunction = -1
 if plotFieldLinesPyplot == 1:
  BX = sortVlsvReaderArray(vlsvReader.read_variable('B')[:,0],locsSorted,nx,nz)
  if plane == 'xy':
   BY = sortVlsvReaderArray(vlsvReader.read_variable('B')[:,1],locsSorted,nx,ny)
   plt.streamplot(meshX,meshY,BX,BY,linewidth=fieldLineWidth,density=fieldLineDensity,arrowsize=fieldLineArrowSize,color=fieldLineColor,alpha=fieldLineAlpha)   
  elif plane == 'xz':
   BZ = sortVlsvReaderArray(vlsvReader.read_variable('B')[:,2],locsSorted,nx,nz)
   plt.streamplot(meshX,meshY,BX,BZ,linewidth=fieldLineWidth,density=fieldLineDensity,arrowsize=fieldLineArrowSize,color=fieldLineColor,alpha=fieldLineAlpha)
 # draw inner boundary
 innerBoundary = plt.Circle((0,0), innerBoundaryR/Re, color='w')                         
 plt.gcf().gca().add_artist(innerBoundary)
 # draw Earth with dayside and nightside
 w1 = Wedge((0,0),1.0,90,270,fc='k')
 w2 = Wedge((0,0),1.0,270,90,fc='w')
 c1 = plt.Circle((0,0),1.0,color='k',fill=False,lw=2.0)
 plt.gcf().gca().add_artist(w1)
 plt.gcf().gca().add_artist(w2)
 plt.gcf().gca().add_artist(c1)
 # draw Earth (black filled circle)
 #earth = plt.Circle((0, 0), 1, color='k')                                                
 #plt.gcf().gca().add_artist(earth)
 plt.axis('scaled')
 plt.xlabel('x [Re]')
 plt.xticks(ticksXAxis/Re)
 plt.yticks(ticksYAxis/Re)
 plt.xlim([xmin/Re,xmax/Re])
 if plane == 'xy':
  plt.ylim([zmin/Re,ymax/Re])
  plt.ylabel('y [Re]')
 elif plane == 'xz':
  plt.ylim([zmin/Re,zmax/Re])
  plt.ylabel('z [Re]')
 if plotLimits == 1:
  plt.xlim(np.divide(areaXLims,Re))
  plt.ylim(np.divide(areaYLims,Re))
 plt.title(vlsvVar + ': ' +  't = ' + round2str(tSim) + ' s')
 if plotCids == 1:
  for cid in cids:
   x,y,z = vlsvReader.get_cell_coordinates(cid)
   if plane == 'xy':
    plt.plot(x/Re,y/Re,cidsPlotStyle,ms=cidsMarkerSize,alpha=cidsAlpha)
   elif plane == 'xz':
    plt.plot(x/Re,z/Re,cidsPlotStyle,ms=cidsMarkerSize,alpha=cidsAlpha)
 plt.savefig(outFileName,bbox_inches='tight')
 print('saved: ' + outFileName)
 plt.clf()
 plt.close()

 # plotting function
def doPlot(vlsvFile):
 # check inputs
 if os.path.isfile(vlsvFile) == False:
  print('file not found: ' + vlsvFile)
  return
 print('processing: ' + vlsvFile)
 fluxFile = fluxFunctionFolder + 'bulk.' + vlsvFile[-12:-5] + '.bin'
 # read file header
 vlsvReader = pt.vlsvfile.VlsvReader(vlsvFile)
 [xmin,ymin,zmin,xmax,ymax,zmax] = vlsvReader.get_spatial_mesh_extent()
 [nx,ny,nz] = vlsvReader.get_spatial_mesh_size()
 [sx,sy,sz] = vlsvReader.get_spatial_block_size()
 dx = (xmax - xmin)/(nx*sx)
 dy = (ymax - ymin)/(ny*sy)
 dz = (zmax - zmin)/(nz*sz)
 dV = dx*dy*dz
 [vxmin,vymin,vzmin,vxmax,vymax,vzmax] = vlsvReader.get_velocity_mesh_extent()
 [nvx,nvy,nvz] = vlsvReader.get_velocity_mesh_size()
 [svx,svy,svz] = vlsvReader.get_velocity_block_size()
 dvx = (vxmax - vxmin)/(nvx*svx)
 dvy = (vymax - vymin)/(nvy*svy)
 dvz = (vzmax - vzmin)/(nvz*svz)
 dvV = dvx*dvy*dvz
 # read simulation time
 vlsvFileTime = vlsvReader.read_parameter('t')
 if vlsvFileTime is None:
  vlsvFileTime = vlsvReader.read_parameter('time')
 # (cell id: index) dict
 locs = vlsvReader.get_cellid_locations()
 cellids = list(locs.keys())
 # sort variable array according to cell ids
 locsSorted = sorted(locs.items(), key=oper.itemgetter(0))
 outFileName = outFilePrefix + os.path.basename(vlsvFile) + '_'
 if colorMapLog == 1:
  outFileName = outFileName + 'log_'
 outFileName = outFileName  + vlsvVar + outFileExtension
 outFileName = re.sub('/','_',outFileName)
 plotAndSave2D(vlsvReader,vlsvVar,nx,ny,nz,locsSorted,xmin,ymin,zmin,xmax,ymax,zmax,vlsvFileTime,fluxFile,outFileName)

# find VLSV files
if 'vlsvFiles' not in locals():
 print('processing from: ' + vlsvFolder + ' ' + str(vlsvFileNumberStart) + ' ... ' + str(vlsvFileNumberEnd))
 vlsvFiles = []
 for f in sorted(os.listdir(vlsvFolder)):
  if (f.startswith('bulk.') and f.endswith('.vlsv')):
   try:
    t = int(f[8:12])
   except ValueError:
    print('ERROR: bad VLSV file name: ' + f)
    continue
   if (t >= vlsvFileNumberStart) and (t <= vlsvFileNumberEnd):
    vlsvFiles.append(vlsvFolder + f)
Ntimes = len(vlsvFiles)
print('VLSV files found: ' + str(Ntimes))

# do plotting sequentially or via multithreading
if Ncores <= 1:
 for f in vlsvFiles:
  doPlot(f)
else:
 from multiprocessing import Pool
 if __name__ == '__main__':
  pool = Pool(Ncores)
  pool.map(doPlot,vlsvFiles)

