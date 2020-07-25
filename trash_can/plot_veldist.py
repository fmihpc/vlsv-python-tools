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

# Velocity distribution plotter (magnetic field coordinate system)
# USAGE: python plot_veldist.py NCores vlsvFileNumberStart vlsvFileNumberEnd
#  NCores = number of multithreading cores
#  vlsvFileNumberStart = number of the first plotted VLSV file
#  vlsvFileNumberend   = number of the last plotted VLSV file
# The script assumes the following file name formats (Xs are integers):
#  VLSV files: bulk.XXXXXXX.vlsv (vlsvFolder)

import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import analysator as pt
import numpy as np
import operator as oper
from rotation import *

# font size
matplotlib.rcParams.update({'font.size': 34})

# constants
Re = 6371e3
mp = 1.6726219e-27
qe = 1.60217662e-19

# VLSV file folder 
vlsvFolder = '/b/vlasiator/2D/BCH/bulk/'

# output file
outFileSize = (20,30)
outFilePrefix = 'fig_veldist'
outFileExtension = '.png'

# velocity plot color map
colorMapLims = (1e-14,1e-11)
colorMapVelSpace = 'seismic'

# unit of velocity
velUnit = 1000e3
velUnitStr = '1000 km/s'

# velocity space cut definitions
# first plot panel: (Vperp1,Vpar)
# second plot panel: (Vperp2,Vperp1)
# Vpar = velocity along magnetic field
# Vperp1 = first velocity component perpendicular to magnetic field
# Vperp2 = second velocity component perpendicular to magnetic field
# velocity space binning
vParBinEdges = np.linspace(-3000e3,3000e3,201)
vPerpBinEdges = np.linspace(-3000e3,3000e3,201)
vxBinEdges = np.linspace(-3000e3,3000e3,201)
vyBinEdges = np.linspace(-3000e3,3000e3,201)
# (Vperp1,Vpar) plot includes Vperp2 values from this range
Vperp2Range = (-100e3,100e3)
# (Vperp2,Vperp1) plot includes Vpar values from this range
VparRange = (-1100e3,-900e3)
# plot limits
vParLims = np.array([min(vParBinEdges),max(vParBinEdges)])
vPerpLims = np.array([min(vPerpBinEdges),max(vPerpBinEdges)])
# plot ticks
vParTicks = np.linspace(min(vParBinEdges),max(vParBinEdges),7)
vPerpTicks = np.linspace(min(vPerpBinEdges),max(vPerpBinEdges),7)

# plot circles of constant kinetic energy at these eV values
showEnergyLevels = (10e3,20e3,30e3,40e3)
energyLevelsAlpha = 0.5
energyLevelsLineWidth = 2.0
energyLevelsColor = 'k'
energyLevelsLineType = 'dashed'
energyLevelsFontSize = 18
energyLevelsTextPosYOffset = 0.02

# give a list of cids
cids = (4502051,4951951,5551701)

# or give a list of requested coordinates
#xReq = np.array([9.4,0,2.4,-2.3,-14.1,-28-2,-40,14.2,9.4,0,-11.7,-23.5])*Re
#zReq = np.array([0,9.4,14.2,21.2,30.6,42.4,51.8,0,16.5,28.3,40.0,49.5])*Re
#yReq = np.zeros(xReq.shape)

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

# find nearest spatial cell with vspace to cid
def getNearestCellWithVspace(vlsvReader,cid):
 cell_candidates = vlsvReader.read(mesh='SpatialGrid',tag='CELLSWITHBLOCKS')
 cell_candidate_coordinates = [vlsvReader.get_cell_coordinates(cell_candidate) for cell_candidate in cell_candidates]
 cell_coordinates = vlsvReader.get_cell_coordinates(cid)
 norms = np.sum((cell_candidate_coordinates - cell_coordinates)**2, axis=-1)**(1./2)
 norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
 return cell_candidates[i]

# find cell ids with vspace to be analyzed if not given
vlsvReader = pt.vlsvfile.VlsvReader(vlsvFiles[0])
cidsTemp = []
if 'cids' not in locals():
 print('Finding nearest cells with vspace from given coordinates')
 if ('xReq' not in locals()) or ('yReq' not in locals()) or ('zReq' not in locals()):
  print('ERROR: cids or (xReq,yReq,zReq) coordinates must be given')
  quit()
 if xReq.shape == yReq.shape == zReq.shape:
  print('Number of points: ' + str(xReq.shape[0]))
 else:
  print('ERROR: bad coordinate variables given')
  quit()
 cidsTemp = []
 for ii in range(xReq.shape[0]):
  cidRequest = (np.int64)(vlsvReader.get_cellid(np.array([xReq[ii],yReq[ii],zReq[ii]])))
  cidNearestVspace = -1
  if cidRequest > 0:
   cidNearestVspace = getNearestCellWithVspace(vlsvReader,cidRequest)
  else:
   print('ERROR: cell not found')
   quit()
  if(cidNearestVspace <= 0):
   print('ERROR: cell with vspace not found')
   quit()
  xCid,yCid,zCid = vlsvReader.get_cell_coordinates(cidRequest)
  xVCid,yVCid,zVCid = vlsvReader.get_cell_coordinates(cidNearestVspace)
  print('Point: ' + str(ii+1) + '/' + str(xReq.shape[0]))
  print('Requested coordinates : ' + str(xReq[ii]/Re) + ', ' + str(yReq[ii]/Re) + ', ' + str(zReq[ii]/Re))
  print('Nearest spatial cell  : ' + str(xCid/Re)    + ', ' + str(yCid/Re)    + ', ' + str(zCid/Re))
  print('Nearest vspace        : ' + str(xVCid/Re)   + ', ' + str(yVCid/Re)   + ', ' + str(zVCid/Re))
  cidsTemp.append(cidNearestVspace)
 cidsTemp = np.unique(cidsTemp)
 print('Unique cells with vspace found: ' + str(len(cidsTemp)))
else:
 print('Using given cell ids and assuming vspace is stored in them')
 cidsTemp = cids

if len(cidsTemp) < 1:
 print('ERROR: no cells found')
 quit()
 
# sort cell id array
cids = np.sort(cidsTemp)

# round number to string
def round2str(x):
 return str(round(x*10)/10)

# list points
for ii in range(len(cids)):
 cc = cids[ii]
 x,y,z = vlsvReader.get_cell_coordinates(cc)
 print('point ' + str(ii+1).zfill(2) + ': ' + str(cc) + ', x = ' + round2str(x/Re) + ', y = ' + round2str(y/Re)  + ', z = ' + round2str(z/Re))

# plot circles of constant kinetic energy in the velocity figure
def plotEnergyCircles():
 for Ekin in showEnergyLevels:
  EkinStr = str(int(round(Ekin/1e3))) + ' keV'
  Rkin = np.sqrt(2*Ekin*qe/mp)/velUnit # convert eVs to velocity units assuming ions are protons (will be used as circle radius)
  c1 = plt.Circle((0,0),Rkin,ls=energyLevelsLineType,color=energyLevelsColor,fill=False,lw=energyLevelsLineWidth,alpha=energyLevelsAlpha)
  plt.gcf().gca().add_artist(c1)
  plt.gcf().gca().text(0,Rkin + energyLevelsTextPosYOffset,EkinStr,fontsize=energyLevelsFontSize,horizontalalignment='center',color=energyLevelsColor,alpha=energyLevelsAlpha)

# create a 2-dimensional histogram
def doHistogram(f,Vx,Vy,V,vxBinEdges,vyBinEdges):
 Vx[Vx < min(vxBinEdges)] = min(vxBinEdges)
 Vy[Vy < min(vyBinEdges)] = min(vyBinEdges)
 Vx[Vx > max(vxBinEdges)] = max(vxBinEdges)
 Vy[Vy > max(vyBinEdges)] = max(vyBinEdges)
 fv = f*np.sqrt( np.sum(np.square(V),1) ) # use particle flux as weighting in the histogram
 (nVhist,VxEdges,VyEdges) = np.histogram2d(Vx,Vy,bins=(vxBinEdges,vyBinEdges),weights=fv,normed=0)
 nVhist = nVhist.transpose()
 dV = np.abs(vxBinEdges[-1] - vxBinEdges[-2]) # assumes constant bin size
 nVhist = np.divide(nVhist,(dV*4*np.pi)) # normalization
 return (nVhist,VxEdges,VyEdges)
  
# analyze velocity space in a spatial cell (velocity space reducer)
def vSpaceReducer(vlsvReader,cid,Bx,By,Bz):
 # check if velocity space exists in this cell
 if vlsvReader.read_variable('fSaved',cid) != 1.0:
  return (False,0,0)
 fMin = 1e-15 # default
 if vlsvReader.check_variable('MinValue') == True:
  fMin = vlsvReader.read_variable('MinValue',cid)
 print('Cell ' + str(cid).zfill(9))
 velcells = vlsvReader.read_velocity_cells(cid)
 V = vlsvReader.get_velocity_cell_coordinates(list(velcells.keys()))
 f = list(zip(*velcells.items()))
 # check that velocity space has cells
 if(len(f) > 0):
  f = np.asarray(f[1])
 else:
  return (False,0,0)
 ii_f = np.where(f >= fMin)
 if len(ii_f) < 1:
  return (False,0,0)
 f = f[ii_f]
 V = V[ii_f,:][0,:,:]
 # rotate velocities to B frame: (vx,vy,vz) -> (vperp2,vperp1,vpar)
 Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
 Bx = Bx/Btot
 By = By/Btot
 Bz = Bz/Btot
 Vrot = rotateVectorToVector(V,np.array([Bx,By,Bz]))
 Vperp2 = Vrot[:,0]
 Vpar = Vrot[:,2]
 # create 2-dimensional histogram of (Vperp1,Vpar) with selected Vperp2 range
 ii1 = np.where((Vperp2 >= Vperp2Range[0]) & (Vperp2 <= Vperp2Range[1]) == True) # select Vperp2 range
 f1 = f[ii1]
 VrotWithVperp2Range = Vrot[ii1,:][0,:,:]
 Vperp1WithVperp2Range = VrotWithVperp2Range[:,1]
 VparWithVperp2Range  = VrotWithVperp2Range[:,2]
 (binsVperp1Vpar,edgesVperp1,edgesVpar) = doHistogram(f1,Vperp1WithVperp2Range,VparWithVperp2Range,VrotWithVperp2Range,vPerpBinEdges,vParBinEdges)
 # create 2-dimensional histogram of (Vperp2,Vperp1) with selected Vpar range
 ii2 = np.where((Vpar >= VparRange[0]) & (Vpar <= VparRange[1]) == True) # select Vpar range
 f2 = f[ii2]
 VrotWithVparRange = Vrot[ii2,:][0,:,:]
 Vperp2WithVparRange = VrotWithVparRange[:,0]
 Vperp1WithVparRange = VrotWithVparRange[:,1]
 (binsVperp2Vperp1_,edgesVperp2_,edgesVperp1_) = doHistogram(f2,Vperp2WithVparRange,Vperp1WithVparRange,VrotWithVparRange,vPerpBinEdges,vPerpBinEdges)
 return (True,binsVperp1Vpar,edgesVperp1,edgesVpar,binsVperp2Vperp1_,edgesVperp2_,edgesVperp1_)

# plotting function
def doPlot(vlsvFile):
 # check inputs
 if os.path.isfile(vlsvFile) == False:
  print('ERROR: file not found: ' + vlsvFile)
  return
 print('processing: ' + vlsvFile)
 vlsvReader = pt.vlsvfile.VlsvReader(vlsvFile)
 # read simulation time
 vlsvFileTime = vlsvReader.read_parameter('t')
 if vlsvFileTime is None:
  vlsvFileTime = vlsvReader.read_parameter('time')
 if vlsvReader.check_variable('fSaved') == False:
  print('ERROR: variable fSaved not found')
  return
 # cell id - index  dict
 locs = vlsvReader.get_cellid_locations()
 cellids = list(locs.keys())
 # sort variable array according to cell ids
 locs_sorted = sorted(locs.items(), key=oper.itemgetter(0))
 fileNameStr = os.path.basename(vlsvFile)
 titleStr = ' t = ' + round2str(vlsvFileTime) + ' s'
 for ii in range(len(cids)):
  cid = cids[ii]
  Bx,By,Bz = vlsvReader.read_variable('B',cid)
  (checkOk,binsVperp1Vpar,edgesVperp1,edgesVpar,binsVperp2Vperp1_,edgesVperp2_,edgesVperp1_) = vSpaceReducer(vlsvReader,cid,Bx,By,Bz)
  # this should never happen
  if checkOk == False:
   print('ERROR: error from velocity space reducer')
   continue
  # velocity distribution
  plt.figure(figsize=outFileSize)
  plt.subplot(2,1,1)
  X,Y = np.meshgrid(edgesVperp1/velUnit,edgesVpar/velUnit)
  plt.pcolormesh(X,Y,binsVperp1Vpar,cmap=colorMapVelSpace,norm=colors.LogNorm(vmin=colorMapLims[0],vmax=colorMapLims[1]))
  plt.grid()
  plt.xlabel('Vperp1 [' + velUnitStr + ']')
  plt.ylabel('Vpar [' + velUnitStr + ']')
  plt.colorbar()
  plotEnergyCircles()
  plt.title(titleStr + '| P' + str(ii+1).zfill(2) + ' | CID = ' + str(cid) + '\n' + 'Vperp2 = ' + round2str(Vperp2Range[0]/velUnit) + ' ... ' + round2str(Vperp2Range[1]/velUnit))
  plt.axis('equal')
  plt.xlim(vPerpLims/velUnit)
  plt.ylim(vParLims/velUnit)
  plt.xticks(vPerpTicks/velUnit)
  plt.yticks(vParTicks/velUnit)
  plt.subplot(2,1,2)
  X,Y = np.meshgrid(edgesVperp2_/velUnit,edgesVperp1_/velUnit)
  plt.pcolormesh(X,Y,binsVperp2Vperp1_,cmap=colorMapVelSpace,norm=colors.LogNorm(vmin=colorMapLims[0],vmax=colorMapLims[1]))
  plt.grid()
  plt.xlabel('Vperp2 [' + velUnitStr + ']')
  plt.ylabel('Vperp1 [' + velUnitStr + ']')
  plt.colorbar()
  plotEnergyCircles()
  plt.title('Vpar = ' + round2str(VparRange[0]/velUnit) + ' ... ' + round2str(VparRange[1]/velUnit))
  plt.axis('equal')
  plt.xlim(vPerpLims/velUnit)
  plt.ylim(vPerpLims/velUnit)
  plt.xticks(vPerpTicks/velUnit)
  plt.yticks(vPerpTicks/velUnit)
  outFileName = outFilePrefix + '_P' + str(ii+1).zfill(2) + '_CID' + str(cid) + '_' + os.path.basename(vlsvFile) + outFileExtension
  plt.savefig(outFileName,bbox_inches='tight')
  print('saved: ' + outFileName)
  plt.clf()
  plt.close()
  
# do plotting sequentially or via multithreading
if Ncores <= 1:
 for f in vlsvFiles:
  doPlot(f)
else:
 from multiprocessing import Pool
 if __name__ == '__main__':
  pool = Pool(Ncores)
  pool.map(doPlot,vlsvFiles)

