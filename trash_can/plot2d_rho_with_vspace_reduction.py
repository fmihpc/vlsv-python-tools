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

# 2-D pseudocolor plotter with velocity space reduction
# Plots a figure with the following subplots:
#  1) rho as read directly from a VLSV file (full resolution)
#     lower resolution plots created by reducting the velocity space in each spatial cell
#     where vspace is stored (typically every 50th spatial cell):
#  2) rho as computed summing the velocity space cells with Ekin > EkinThresholds[0]
#  3) rho as computed summing the velocity space cells with Ekin > EkinThresholds[1]
#  ...
#  N) rho as computed summing the velocity space cells with Ekin > EkinThresholds[N-1]
#
# USAGE: python plot2d_rho_with_vspace_reduction.py NCores vlsvFile
#  NCores = number of multithreading cores
#  vlsvFile = VLSV file

import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import analysator as pt
import numpy as np
import operator as oper
from matplotlib.patches import Wedge

# font size
matplotlib.rcParams.update({'font.size': 12})

# constants
Re = 6371e3
mp = 1.6726219e-27
qe = 1.60217662e-19

# radius of inner boundary
innerBoundaryR = 30000e3

# plotting variable
vlsvVar = 'rho'
colorMapLog = 1
colorMapLims1 = (1e5,1e7)
colorMapLims2 = (1e-2,1e7)
colorMapName = 'seismic'

# axes tick marks
ticksXAxis = np.linspace(-100,100,21)*Re
ticksYAxis = np.linspace(-100,100,21)*Re

# threholds limits of kinetic energy in keV for vspace reducing: each subplot will include ion density only above these values
EkinThresholds = (0,5,10,20,30)

# striding of cells when reducing vspaces
# use Nstride >> 1 to make script run faster for debugging etc purposes (every Nstride'th cell reduced)
# use Nstride = 1 for a proper plot (all cells with vspace reduced)
Nstride = 137

# output file
outFileSize = (30,12)
outFilePrefix = 'fig2d_vspace_reduction_rho_'
outFileExtension = '.png'
# number of subplots NxSubPlots*NySubPlots should be larger than len(EkinThresholds) + 1
NxSubPlots = 3
NySubPlots = 2

# plot or remove tail region defined by a conic section model
removeTail = 1
csTail = (3*Re,1.000001,-3*Re,np.pi/1.12)

# configure command line arguments
if not (len(sys.argv) == 3):
 print('two command line arguments required')
 quit()
Ncores = -100
vlsvFile = ''
try:
 Ncores = int(sys.argv[1])
except ValueError:
 print('ERROR: arg1 = ' + str(sys.argv[1]))
 quit()
if (Ncores < 1) or (Ncores > 40):
 print('ERROR: negative or otherwise bad number of cores')
 quit()
vlsvFile = sys.argv[2]
if os.path.isfile(vlsvFile) == False:
 print('ERROR: file not found: ' + vlsvFile)
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
def plot2D(vlsvReader,vlsvVar,plane,nx,ny,nz,locsSorted,xmin,ymin,zmin,xmax,ymax,zmax,cmin,cmax,tSim):
 # load variable and create arrays
 D = 0
 meshX = 0
 meshY = 0
 if plane == 'xy':
  D = sortVlsvReaderArray(vlsvReader.read_variable(vlsvVar),locsSorted,nx,ny)
  meshX,meshY = np.meshgrid(np.linspace(xmin/Re,xmax/Re,nx),np.linspace(ymin/Re,ymax/Re,ny))
 elif plane == 'xz':
  D = sortVlsvReaderArray(vlsvReader.read_variable(vlsvVar),locsSorted,nx,nz)
  meshX,meshY = np.meshgrid(np.linspace(xmin/Re,xmax/Re,nx),np.linspace(zmin/Re,zmax/Re,nz))
 else:
  print('ERROR: unknown plane')
  return
 if removeTail == 1:
  meshR = np.sqrt(meshX*meshX + meshY*meshY)*Re
  meshR_tail = getConicSectionRadius(csTail,meshX*Re,meshY*Re)
  D[(meshR < meshR_tail)] = np.nan
  D = np.ma.masked_invalid(D) # note this will not be needed in matplotlib > 2.1
 if colorMapLog == 1:
  plt.pcolormesh(meshX,meshY,D,cmap=colorMapName,norm=colors.LogNorm(vmin=cmin,vmax=cmax))
 else:
  plt.pcolormesh(meshX,meshY,D,cmap=colorMapName,vmin=cmin,vmax=cmax)
 plt.colorbar()
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
 plt.title(vlsvVar + ': t = ' + round2str(tSim) + ' s')
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
 else:
  print('ERROR: unknown plane')
  return
 
# 2-d pseudocolor plot for vspace reduced grid
def plot2DReduced(varStr,D,meshX,meshY,plane,xmin,ymin,zmin,xmax,ymax,zmax,cmin,cmax,tSim):
 if removeTail == 1:
  meshR = np.sqrt(meshX*meshX + meshY*meshY)*Re
  meshR_tail = getConicSectionRadius(csTail,meshX*Re,meshY*Re)
  D[(meshR < meshR_tail)] = np.nan
  D = np.ma.masked_invalid(D) # note this will not be needed in matplotlib > 2.1
 if colorMapLog == 1:
  plt.pcolormesh(meshX,meshY,D,cmap=colorMapName,norm=colors.LogNorm(vmin=cmin,vmax=cmax))
 else:
  plt.pcolormesh(meshX,meshY,D,cmap=colorMapName,vmin=cmax,vmax=cmin)
 plt.colorbar()
 # draw inner boundary
 innerBoundary = plt.Circle((0,0), innerBoundaryR/Re, color='w')                         
 plt.gcf().gca().add_artist(innerBoundary)
 # draw Earth with dayside and nightside
 w1 = Wedge((0,0),1.0,90,270,fc='k')
 w2 = Wedge((0,0),1.0,270,90,fc='w')
 plt.gcf().gca().add_artist(w1)
 plt.gcf().gca().add_artist(w2)
 plt.title(varStr + ': ' +  't = ' + round2str(tSim) + ' s')
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
 else:
  print('ERROR: unknown plane')
  return

# check enough subplot defined
if len(EkinThresholds)+1 > NxSubPlots*NySubPlots:
 print('ERROR: too few subplots defined, need at least ' + str(len(EkinThresholds)+1))
print('processing: ' + vlsvFile)
print('reducing every ' + str(Nstride) + 'th cell with vspace')
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
# determine 2d simulation plane
plane = 'xyz'
if(ny > nz and nz == 1):
 plane = 'xy'
elif(ny < nz and ny == 1):
 plane = 'xz'
else:
 print('ERROR: cannot determine simulation plane')
# read simulation time
vlsvFileTime = vlsvReader.read_parameter('t')
if vlsvFileTime is None:
 vlsvFileTime = vlsvReader.read_parameter('time')
# check variables exist
if vlsvReader.check_variable(vlsvVar) == False:
 print('ERROR: variable ' + vlsvVar + ' not found')
 quit()
if vlsvReader.check_variable('fSaved') == False:
 print('ERROR: variable fSaved not found')
 quit()

# determine distance between cells with vspaces assuming it is constant in the simulation box
found1 = False
found2 = False
dxVspaces = -1.0
x1=y1=z1=x2=y2=z2=0.0
for cid in range(1,1000):
 if vlsvReader.read_variable('fSaved',cid) == 1.0:
  if found1 == False:
   x1,y1,z1 = vlsvReader.get_cell_coordinates(cid)
   found1 = True
  else:
   if found2 == False:
    x2,y2,z2 = vlsvReader.get_cell_coordinates(cid)
    found2 = True
    break
   else:
    break
if x1 != x2:
 if z1 == z2:
  dxVspaces = abs(x2-x1)
 else:
  print('ERROR: cannot determine dx between velocity spaces')
  quit()
else:
 print('ERROR: cannot determine dx between velocity spaces')
 quit()

# create mesh for plotting reduced rhos
nxVspaces = (int)(nx*dx/dxVspaces)
nyVspaces = (int)(ny*dy/dxVspaces)
nzVspaces = (int)(nz*dz/dxVspaces)
# coordinate mesh variables for plotting
meshXVspaces = 0
meshYVspaces = 0
if plane == 'xy':
 meshXVspaces,meshYVspaces = np.meshgrid(np.linspace(x1/Re,(x1+nxVspaces*dxVspaces)/Re,nxVspaces),np.linspace(y1/Re,(y1+nyVspaces*dxVspaces)/Re,nyVspaces))
elif plane == 'xz':
 meshXVspaces,meshYVspaces = np.meshgrid(np.linspace(x1/Re,(x1+nxVspaces*dxVspaces)/Re,nxVspaces),np.linspace(z1/Re,(z1+nzVspaces*dxVspaces)/Re,nzVspaces))
else:
 print('ERROR: unknown plane')

# (cell id: index) dict
locs = vlsvReader.get_cellid_locations()
cellids = list(locs.keys())
# (cell id: index) dict with only vspace locations
cellidsWithVspace = np.sort(vlsvReader.read(mesh='SpatialGrid',tag='CELLSWITHBLOCKS'))
# sort variable array according to cell ids
locsSorted = sorted(locs.items(), key=oper.itemgetter(0))

# initialize arrays where rho with different energy limits is stored
if Ncores > 1:
 from multiprocessing import Pool,Array
 rhoReducedParallel = []
 for i in range(0,len(EkinThresholds)):
  # when multithreading is used, initialize a shared memory map with Array
  rhoReducedParallel.append( Array('d',nxVspaces*nzVspaces) )
else:
 # otherwise use a normal numpy array
 rhoReduced = np.empty((nxVspaces*nzVspaces,len(EkinThresholds)))
 rhoReduced.fill(np.nan)

# analyze velocity space in a spatial cell:
# determine density (rho) of ions with energies exceeding EkinThresholds values separately for each value
def vSpaceReducer(ii):
 cid = cellidsWithVspace[ii]
 # check if velocity space exists in this cell
 if vlsvReader.read_variable('fSaved',cid) != 1.0:
  return
 fMin = 1e-15 # default
 if vlsvReader.check_variable('MinValue') == True:
  fMin = vlsvReader.read_variable('MinValue',cid)
 velcells = vlsvReader.read_velocity_cells(cid)
 V = vlsvReader.get_velocity_cell_coordinates(list(velcells.keys()))
 Ekin = 0.5*mp*(np.sum(np.square(V),1))/qe/1e3 # keV
 f = list(zip(*velcells.items()))
 # check that velocity space has cells
 if(len(f) > 0):
  f = np.asarray(f[1])
 else:
  return
 #if len(f[f < 0]) > 10:
 # print('Warning: many f < 0 values')
 # remove illegal f values
 ii_f = np.where(f >= fMin)
 if len(ii_f) < 1:
  return
 f = f[ii_f]
 V = V[ii_f]
 Ekin = Ekin[ii_f]
 # energy filtering
 for jj in range(0,len(EkinThresholds),1):
  f_filtered = f[Ekin >= EkinThresholds[jj]]
  if(len(f_filtered) > 0):
   sum_f_filtered = np.sum(f_filtered)*dvV
   if Ncores > 1:
    rhoReducedParallel[jj][ii] = sum_f_filtered
   else:
    rhoReduced[ii,jj] = sum_f_filtered

# computed reduction sequentially or via multithreading
inds = list(range(0,len(cellidsWithVspace),Nstride))
print('reducing velocity spaces in ' + str(int(len(inds))) + ' cells')
if Ncores > 1:
 if __name__ == '__main__':
  p = Pool(Ncores)
  p.map(vSpaceReducer,inds)
  p.close()
else:
 for ii in inds:
  vSpaceReducer(ii)

# do plot
plt.figure(figsize=outFileSize)
# first subplot which is a bulk variable (e.g. rho) directly from VLSV file
plt.subplot(NySubPlots,NxSubPlots,1)
plot2D(vlsvReader,vlsvVar,plane,nx,ny,nz,locsSorted,xmin,ymin,zmin,xmax,ymax,zmax,colorMapLims1[0],colorMapLims1[1],vlsvFileTime)
# further subplots which are created by the vspace reducer
for ii in range(0,len(EkinThresholds),1):
 if Ncores > 1:
  rhoPlotter = np.array(rhoReducedParallel[ii])
  rhoPlotter[rhoPlotter == 0] = np.nan
 else:
  rhoPlotter = rhoReduced[:,ii]
 rhoPlotter = np.asarray(rhoPlotter).reshape(meshXVspaces.shape[0],meshXVspaces.shape[1])
 plt.subplot(NySubPlots,NxSubPlots,ii+2)
 cmin = colorMapLims1[0]
 cmax = colorMapLims1[1]
 # use color scale #2 for second and further reduced rho plots
 if ii > 0:
  cmin = colorMapLims2[0]
  cmax = colorMapLims2[1]
 plot2DReduced('rho (Ekin > ' + str(EkinThresholds[ii]) + ' keV)',rhoPlotter,meshXVspaces,meshYVspaces,plane,xmin,ymin,zmin,xmax,ymax,zmax,cmin,cmax,vlsvFileTime)
outFileName = outFilePrefix + os.path.basename(vlsvFile) + outFileExtension
plt.savefig(outFileName,bbox_inches='tight')


