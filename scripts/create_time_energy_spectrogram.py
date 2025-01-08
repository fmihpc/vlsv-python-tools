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
'''
Omnidirectional time-energy spectrogram
USAGE: python create_time_energy_spectrogram.py NCores vlsvFileNumberStart vlsvFileNumberEnd
 NCores = number of multithreading cores
 vlsvFileNumberStart = number of the first plotted VLSV file
 vlsvFileNumberend   = number of the last plotted VLSV file
The script assumes the following file name formats (Xs are integers):
 VLSV files: bulk.XXXXXXX.vlsv (vlsvFolder)
ASCII output files:
 energy spectra : outSpectraFilePrefix_bulk.XXXXXXX.vlsv.dat
 bulk parameters: outBulkFilePrefix_bulk.XXXXXXX.vlsv.dat

Energy spectra file format:
 CID1       CID2       CIDN
 Time       Time       ... 
 NbinEdges  Nbins      ... 
 EbinEdge1  EbinEdge1  ... 
 EbinEdge2  EbinEdge2  ... 
 ...        ...        ... 
 EbinEdgeN  EbinEdgeN  ... 
 Nbins      Nbins      ... 
 bin1       bin1       ... 
 bin2       bin2       ... 
 ...        ...        ... 
 binN       binN       ... 
 where each column includes one spatial cell
 CIDX = cell id
 Time = simulation time
 NbinEdges = number of bin edges
 EbinEdgeX = energy value (in electron volts) of edge X
 Nbins = number of bins (should be NbinEdges - 1)
 binX = histogram value of energy bin X

Bulk parameter file format:
 CID1    CID2    CIDN
 Time    Time    ... 
 Param1  Param1  ... 
 Param2  Param2  ... 
 ...     ...     ... 
 ParamN  ParamN  ... 
 where each column includes one spatial cell
 CIDX = cell id
 Time = simulation time
 Param1...N = rho,rhovx,rhovy,rhovz,Bx,By,Bz,Ex,Ey,Ez,Pxx,Pyy,Pzz,POff1,POff2,POff3,T,Tpar,Tperp1,Tperp2,Tperp

'''

import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import analysator as pt
import numpy as np
import operator as oper
import logging

# constants
Re = 6371e3
mp = 1.6726219e-27
qe = 1.60217662e-19

# VLSV file folder
vlsvFolder = '/b/vlasiator/2D/BCH/bulk/'

# output files
outSpectraFilePrefix = 'spectra'
outBulkFilePrefix = 'bulk_parameters'

# bin edges of kinetic energy in electron volts (energies below and above the last and first and )
EkinBinEdges = np.logspace(np.log10(100),np.log10(80e3),66)

# Initialize as none
xReq = None
yReq = None
zReq = None

# give a list of cids
cids = (4502051,4951951,5551701)

# or give a list of requested coordinates
#xReq = np.array([9.4,0,2.4,-2.3,-14.1,-28-2,-40,14.2,9.4,0,-11.7,-23.5])*Re
#zReq = np.array([0,9.4,14.2,21.2,30.6,42.4,51.8,0,16.5,28.3,40.0,49.5])*Re
#xReq = np.array([-5.1 , -7.0])*Re
#zReq = np.array([16.5 , 18.0])*Re
#yReq = np.zeros(xReq.shape)

# configure command line arguments
if not (len(sys.argv) == 4):
 logging.info('three command line arguments required')
 quit()
Ncores = -100
vlsvFileNumberStart = -100
vlsvFileNumberEnd = -100
try:
 Ncores = int(sys.argv[1])
 vlsvFileNumberStart = int(sys.argv[2])
 vlsvFileNumberEnd = int(sys.argv[3])
except ValueError:
 logging.info('ERROR: arg1 = ' + str(sys.argv[1]) + ', arg2 = ' + str(sys.argv[2]) + ', arg3 = ' + str(sys.argv[3]))
 quit()
if (Ncores < 1) or (Ncores > 40):
 logging.info('ERROR: negative or otherwise bad number of cores')
 quit() 
if (vlsvFileNumberStart < 0) or (vlsvFileNumberEnd < 0) or (vlsvFileNumberStart > vlsvFileNumberEnd):
 logging.info('ERROR: negative or otherwise bad start or end file numbers')
 quit()
logging.info('running on ' + str(Ncores) + ' cores')

# find VLSV files
if 'vlsvFiles' not in locals():
 logging.info('processing from: ' + vlsvFolder + ' ' + str(vlsvFileNumberStart) + ' ... ' + str(vlsvFileNumberEnd))
 vlsvFiles = []
 for f in sorted(os.listdir(vlsvFolder)):
  if (f.startswith('bulk.') and f.endswith('.vlsv')):
   try:
    t = int(f[8:12])
   except ValueError:
    logging.info('ERROR: bad VLSV file name: ' + f)
    continue
   if (t >= vlsvFileNumberStart) and (t <= vlsvFileNumberEnd):
    vlsvFiles.append(vlsvFolder + f)
Ntimes = len(vlsvFiles)
logging.info('VLSV files found: ' + str(Ntimes))

#vlsvFileNumberStart = 1000; vlsvFileNumberEnd =3000;
#vlsvFolder = '/b/vlasiator/2D/BCQ/bulk/'
#vlsvFolder ='/lustre/tmp/vlasiator/2D/BCQ/bulk/'
#xReq = 1*Re; yReq = 0; zReq = -19*Re;
#xReq = 1*Re; yReq = 0; zReq = -17*Re;
#xReq = 1*Re; yReq = 0; zReq = -13*Re;
#xReq = 1*Re; yReq = 0; zReq = -7*Re;

# find cell ids with vspace to be analyzed if not given
vlsvReader = pt.vlsvfile.VlsvReader(vlsvFiles[0])
cidsTemp = []
if 'cids' not in locals():
 logging.info('Finding nearest cells with vspace from given coordinates')
 if (xReq is None) or (yReq is None) or (zReq is None):
  logging.info('ERROR: cids or (xReq,yReq,zReq) coordinates must be given')
  quit()
 if xReq.shape == yReq.shape == zReq.shape:
  logging.info('Number of points: ' + str(xReq.shape[0]))
 else:
  logging.info('ERROR: bad coordinate variables given')
  quit()
 cidsTemp = []
 for ii in range(xReq.shape[0]):
  cidRequest = (np.int64)(vlsvReader.get_cellid(np.array([xReq[ii],yReq[ii],zReq[ii]])))
  cidNearestVspace = -1
  if cidRequest > 0:
   cidNearestVspace = vlsvReader.get_cellid_with_vdf(np.array([xReq[ii],yReq[ii],zReq[ii]]), pop='proton')   # deprecated getNearestCellWithVspace(). needs testing
  else:
   logging.info('ERROR: cell not found')
   quit()
  if(cidNearestVspace <= 0):
   logging.info('ERROR: cell with vspace not found')
   quit()
  xCid,yCid,zCid = vlsvReader.get_cell_coordinates(cidRequest)
  xVCid,yVCid,zVCid = vlsvReader.get_cell_coordinates(cidNearestVspace)
  logging.info('Point: ' + str(ii+1) + '/' + str(xReq.shape[0]))
  logging.info('Requested coordinates : ' + str(xReq[ii]/Re) + ', ' + str(yReq[ii]/Re) + ', ' + str(zReq[ii]/Re))
  logging.info('Nearest spatial cell  : ' + str(xCid/Re)    + ', ' + str(yCid/Re)    + ', ' + str(zCid/Re))
  logging.info('Nearest vspace        : ' + str(xVCid/Re)   + ', ' + str(yVCid/Re)   + ', ' + str(zVCid/Re))
  cidsTemp.append(cidNearestVspace)
 cidsTemp = np.unique(cidsTemp)
 logging.info('Unique cells with vspace found: ' + str(len(cidsTemp)))
else:
 logging.info('Using given cell ids and assuming vspace is stored in them')
 cidsTemp = cids

if len(cidsTemp) < 1:
 logging.info('ERROR: no cells found')
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
 logging.info('point ' + str(ii+1).zfill(2) + ': ' + str(cc) + ', x = ' + round2str(x/Re) + ', y = ' + round2str(y/Re)  + ', z = ' + round2str(z/Re))

# analyze velocity space in a spatial cell
def vSpaceReducer(vlsvReader,cid):
 # check if velocity space exists in this cell
 if vlsvReader.read_variable('fSaved',cid) != 1.0:
  return (False,0,0)
 fMin = 1e-15 # default
 if vlsvReader.check_variable('MinValue') == True:
  fMin = vlsvReader.read_variable('MinValue',cid)
 logging.info('Cell ' + str(cid).zfill(9))
 velcells = vlsvReader.read_velocity_cells(cid)
 V = vlsvReader.get_velocity_cell_coordinates(list(velcells.keys()))
 V2 = np.sum(np.square(V),1)
 Ekin = 0.5*mp*V2/qe
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
 Ekin = Ekin[ii_f]
 V2 = V2[ii_f]
 Ekin[Ekin < min(EkinBinEdges)] = min(EkinBinEdges)
 Ekin[Ekin > max(EkinBinEdges)] = max(EkinBinEdges)
 # normalization
 fv = f*np.sqrt(V2) # use particle flux as weighting
 # compute histogram
 (nhist,edges) = np.histogram(Ekin,bins=EkinBinEdges,weights=fv,normed=0)
 # normalization
 dE = EkinBinEdges[1:] - EkinBinEdges[0:-1]
 nhist = np.divide(nhist,(dE*4*np.pi))
 return (True,nhist,edges)

# spectra creation function
def doSpectra(vlsvFile):
 # check inputs
 if os.path.isfile(vlsvFile) == False:
  logging.info('ERROR: file not found: ' + vlsvFile)
  return
 logging.info('processing: ' + vlsvFile)
 vlsvReader = pt.vlsvfile.VlsvReader(vlsvFile)
 # read simulation time
 vlsvFileTime = vlsvReader.read_parameter('t')
 if vlsvFileTime is None:
  vlsvFileTime = vlsvReader.read_parameter('time')
 if vlsvReader.check_variable('fSaved') == False:
  logging.info('ERROR: variable fSaved not found')
  return
 # cell id - index  dict
 locs = vlsvReader.get_cellid_locations()
 cellids = list(locs.keys())
 # sort variable array according to cell ids
 locs_sorted = sorted(locs.items(), key=oper.itemgetter(0))
 fileNameStr = os.path.basename(vlsvFile)
 spectraStr = [] # spectra file contents
 bulkStr = [] # bulk parameter file contents
 for ii in range(len(cids)):
  cid = cids[ii]
  (checkOk,nhist,edges) = vSpaceReducer(vlsvReader,cid)
  # this should never happen
  if checkOk == False:
   logging.info('ERROR: error from velocity space reducer')
   continue
  params = (cid,vlsvFileTime)
  paramsStr = [str(cid).zfill(17),str('%+.10e' % vlsvFileTime),str(len(edges)).zfill(17)]
  for jj in range(len(edges)):
   paramsStr.append(str('%+.10e' % edges[jj]))
  paramsStr.append(str(len(nhist)).zfill(17))
  for jj in range(len(nhist)):
   paramsStr.append(str('%+.10e' % nhist[jj]))
  if ii == 0:
   spectraStr = [paramsStr]
  else:
   spectraStr += [paramsStr]
  # bulk parameters
  rho = vlsvReader.read_variable('rho',cid)
  Bx,By,Bz = vlsvReader.read_variable('B',cid)
  Ex,Ey,Ez = vlsvReader.read_variable('E',cid)
  rhovx,rhovy,rhovz = vlsvReader.read_variable('rho_v',cid)
  Pxx,Pyy,Pzz = vlsvReader.read_variable('PTensorDiagonal',cid)
  POff1,POff2,POff3 = vlsvReader.read_variable('PTensorOffDiagonal',cid)
  T = vlsvReader.read_variable('Temperature',cid)
  Tpar = vlsvReader.read_variable('TParallel',cid)
  Tperp1 = vlsvReader.read_variable('TTensorRotated',cid)[0,0]
  Tperp2 = vlsvReader.read_variable('TTensorRotated',cid)[1,1]
  Tperp = vlsvReader.read_variable('TPerpendicular',cid)
  # construct output parameter columns as strings
  params = (cid,vlsvFileTime,rho,rhovx,rhovy,rhovz,Bx,By,Bz,Ex,Ey,Ez,Pxx,Pyy,Pzz,POff1,POff2,POff3,T,Tpar,Tperp1,Tperp2,Tperp)
  paramsStr = []
  for jj in range(len(params)):
   if jj < 1:
    paramsStr.append(str(params[jj]).zfill(17))
   else:
    paramsStr.append(str('%+.10e' % params[jj]))
  # concatenate columns
  if ii == 0:
   bulkStr = [paramsStr]
  else:
   bulkStr += [paramsStr]
 # write files
 if len(spectraStr) > 0:
  outFileName = outSpectraFilePrefix + '_' + fileNameStr + '.dat'
  logging.info('writing: ' + outFileName)
  with open(outFileName,'a') as f_handle:
   np.savetxt(f_handle,np.column_stack(spectraStr),fmt='%s')
 if (len(bulkStr) > 0):
  outFileName = outBulkFilePrefix + '_' + fileNameStr + '.dat'
  logging.info('writing: ' + outFileName)
  with open(outFileName,'a') as f_handle:
   np.savetxt(f_handle,np.column_stack(bulkStr),fmt='%s')

# run sequentially or via multithreading
if Ncores <= 1:
 for f in vlsvFiles:
  doSpectra(f)
else:
 from multiprocessing import Pool
 if __name__ == '__main__':
  pool = Pool(Ncores)
  pool.map(doSpectra,vlsvFiles)

