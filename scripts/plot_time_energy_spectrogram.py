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

# Time-energy spectrogram and bulk parameter time series plotter
# USAGE: python plot_time_energy_spectrogram.py
# Constructs and plots time seriers from ASCII files created by
# the script create_time_energy_spectrogram.py.
# The script assumes the following file name formats (Xs are integers):
#  spectra files: inputSpectraFilePrefix*inputFileExtension
#  bulk parameters files: inputBulkFilePrefix*inputFileExtension

import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import analysator as pt
import numpy as np

# font size and linewidth
matplotlib.rcParams.update({'font.size': 26})
matplotlib.rcParams['lines.linewidth'] = 2

# constants
Re = 6371e3
mp = 1.6726219e-27
qe = 1.60217662e-19
K2eV = 11604.505

# VLSV file
vlsvFile = '/b/vlasiator/2D/BCH/bulk/bulk.0003000.vlsv'

# spectra and bulk parameter file folder
folder = './'
inputSpectraFilePrefix = 'spectra'
inputBulkFilePrefix = 'bulk_parameters'
inputFileExtension = '.dat'

# choose cell ids to be plotted (must be found in input spectra and bulk files)
plotterCids = (4502051,4951951,5551701)

# output file
outFileSizeAllSpectra = (15,20)
outFileSizeSpectraAndBulk = (15,20)
outFilePrefix = 'fig'
outFileExtension = '.png'

# color map for energy spectrogram
colorMapLims = (1e-9,2e-6)
colorMapName = 'seismic'

# limits and ticks
timeLims = [1200,2200]
EkinLims = [1e2,1e5]
EkinTicks = np.array([1e2,1e3,1e4,1e5])

# round number to string
def round2str(x):
 return str(round(x*10)/10)

# plotting function
def doPlots(folder):
 # find spectra and bulk parameter files
 spectraFiles = []
 bulkFiles = []
 for f in sorted(os.listdir(folder)):
  if f.startswith(inputSpectraFilePrefix) and f.endswith(inputFileExtension):
   spectraFiles.append(f)
  if f.startswith(inputBulkFilePrefix) and f.endswith(inputFileExtension):
   bulkFiles.append(f)
 # check
 if not len(bulkFiles) == len(spectraFiles):
  print('ERROR: different number of spectra and bulk files')
  return
 Nt = len(bulkFiles)
 # read bulk files
 bb = np.loadtxt(folder + bulkFiles[0],unpack=True)
 ss = np.loadtxt(folder + spectraFiles[0],unpack=True)
 bulk = np.nan*np.zeros(bb.shape + (Nt,))
 for ii in range(Nt):
  bulk[:,:,ii] = np.loadtxt(folder + bulkFiles[ii],unpack=True)
 spectra = np.nan*np.zeros(ss.shape + (Nt,))
 for ii in range(Nt):
  spectra[:,:,ii] = np.loadtxt(folder + spectraFiles[ii],unpack=True)
 cids = np.int64(bulk[:,0,0])
 Ncids = len(cids)
 t = bulk[0,1,:]
 # sanity checks
 for ii in range(Nt):
  cids_b = np.int64(bulk[:,0,ii])
  cids_s = np.int64(spectra[:,0,ii])
  if not np.all(cids == cids_b) and np.all(cids == cids_s):
   print('ERROR: different cells in spectra and bulk files')
   return
  t_b = bulk[:,1,ii]
  t_s = spectra[:,1,ii]
  if not np.all(t == t_b) and np.all(t == t_s):
   print('ERROR: different simulation times in spectra and bulk files')
   return
 # indices of bulk parameters
 #i_rho,i_rhovx,i_rho_vy,i_rho_vz,i_Bx,i_By,i_Bz,i_Ex,i_Ey,i_Ez,i_Pxx,i_Pyy,i_Pzz,i_POff1,i_POff2,i_POff3,i_Ttot,i_Tpar,i_Tperp1,i_Tper2 = np.array(range(20))+2
 rho = bulk[:,2,:]
 rhovx = bulk[:,3,:]
 rhovy = bulk[:,4,:]
 rhovz = bulk[:,5,:]
 Bx = bulk[:,6,:]
 By = bulk[:,7,:]
 Bz = bulk[:,8,:]
 Ex = bulk[:,9,:]
 Ey = bulk[:,10,:]
 Ez = bulk[:,11,:]
 Pxx = bulk[:,12,:]
 Pyy = bulk[:,13,:]
 Pzz = bulk[:,14,:]
 POff1 = bulk[:,15,:]
 POff2 = bulk[:,16,:]
 POff3 = bulk[:,17,:]
 Ttot = bulk[:,18,:]
 Tpar = bulk[:,19,:]
 Tperp1 = bulk[:,20,:]
 Tperp2 = bulk[:,21,:]
 Tperp = bulk[:,22,:]
 Vx = rhovx/rho
 Vy = rhovy/rho
 Vz = rhovz/rho
 Vtot = np.sqrt(np.square(Vx) + np.square(Vy) + np.square(Vz))
 Btot = np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz))
 Etot = np.sqrt(np.square(Ex) + np.square(Ey) + np.square(Ez))
 
 # spectra
 NbinEdges = np.int64(spectra[0,2,0])
 Nbins = np.int64(spectra[0,2+NbinEdges+1,0])
 EkinBinEdges = spectra[0,3:3+NbinEdges,0]
 EkinBins = spectra[:,3+NbinEdges+1:,:]
 EkinCenters = 0.5*(EkinBinEdges[0:-1] + EkinBinEdges[1:])
 
 # output variable
 result = dict()
 for ii in range(Ncids):
  cid = cids[ii]
  result[cid] = dict()
  result[cid]['EkinBins'] = EkinBins[ii,:,:]
  result[cid]['t'] = t
  result[cid]['EkinCenters'] = EkinCenters
  result[cid]['rho'] = rho[ii,:]
  result[cid]['Vx'] = Vx[ii,:]
  result[cid]['Vy'] = Vy[ii,:]
  result[cid]['Vz'] = Vz[ii,:]
  result[cid]['Vtot'] = Vtot[ii,:]
  result[cid]['Tpar'] = Tpar[ii,:]
  result[cid]['Tperp1'] = Tperp1[ii,:]
  result[cid]['Tperp2'] = Tperp2[ii,:]
  result[cid]['Tperp'] = Tperp[ii,:]
  result[cid]['Ttot'] = Ttot[ii,:]
  result[cid]['Bx'] = Bx[ii,:]
  result[cid]['By'] = By[ii,:]
  result[cid]['Bz'] = Bz[ii,:]
  result[cid]['Btot'] = Btot[ii,:]
 return result

params = doPlots(folder)
vlsvReader = pt.vlsvfile.VlsvReader(vlsvFile)

# plotter : time-energy spectrograms from all points in one plot
plt.figure(figsize=outFileSizeAllSpectra)
Ncids = len(plotterCids)
for ii in range(Ncids):
 cid = plotterCids[ii]
 xp,yp,zp = vlsvReader.get_cell_coordinates(cid)
 X,Y = np.meshgrid(params[cid]['t'],params[cid]['EkinCenters'])
 EkinBins = params[cid]['EkinBins']
 axes = plt.subplot(Ncids,1,Ncids-ii)
 plt.pcolormesh(X,Y,EkinBins,cmap=colorMapName,norm=colors.LogNorm(vmin=colorMapLims[0],vmax=colorMapLims[1]))
 plt.xlim(timeLims)
 plt.ylim(EkinLims)
 plt.yscale('log')
 plt.yticks(EkinTicks)
 plt.ylabel('Ekin(P' + str(ii+1) + ') [eV]')
 plt.colorbar(orientation='vertical')
 plt.title('P' + str(ii+1)  + ': (x = ' + round2str(xp/Re) + ', y = '+ round2str(yp/Re) + ', z = '+ round2str(zp/Re) + ')')
 plt.gcf().gca().xaxis.set_tick_params(which='both', direction='out')
 if ii <= 0:
  plt.xlabel('Time [s]')
plt.savefig(outFilePrefix + '_spectra_all' + outFileExtension,bbox_inches='tight')
plt.clf()
plt.close()

# plotter: time-energy spectrogram and bulk parameters in each point in seprate plots
Ncids = len(plotterCids)
for ii in range(Ncids):
 cid = plotterCids[ii]
 xp,yp,zp = vlsvReader.get_cell_coordinates(cid)
 t = params[cid]['t']
 X,Y = np.meshgrid(t,params[cid]['EkinCenters'])

 plt.figure(figsize=outFileSizeSpectraAndBulk)
 
 plt.subplot(511)
 plt.pcolormesh(X,Y,params[cid]['EkinBins'],cmap=colorMapName,norm=colors.LogNorm(vmin=colorMapLims[0],vmax=colorMapLims[1]))
 plt.xlim(timeLims)
 plt.ylim(EkinLims)
 plt.yscale('log')
 plt.ylabel('Ekin [eV]')
 plt.colorbar(orientation='horizontal') 
 plt.title('P' + str(ii+1)  + ': (x = ' + round2str(xp/Re) + ', y = '+ round2str(yp/Re) + ', z = '+ round2str(zp/Re) + ')')
 plt.gcf().gca().xaxis.set_tick_params(which='both', direction='out')
 plt.gcf().gca().xaxis.set_ticklabels([])
 
 plt.subplot(512)
 plt.plot(t,params[cid]['rho']/1e6)
 plt.xlim(timeLims)
 plt.gcf().gca().xaxis.set_ticklabels([])
 plt.ylabel('n [cm$^{-3}$]');
 plt.grid()
 
 plt.subplot(513)
 plt.plot(t,params[cid]['Vx']/1e3,'-r',label='Vx')
 plt.plot(t,params[cid]['Vy']/1e3,'-g',label='Vy')
 plt.plot(t,params[cid]['Vz']/1e3,'-b',label='Vz')
 plt.plot(t,params[cid]['Vtot']/1e3,'-k',label='Vtot')
 plt.xlim(timeLims)
 plt.gcf().gca().xaxis.set_ticklabels([])
 plt.ylabel('V [km/s]');
 plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
 plt.grid()
 
 plt.subplot(514)
 plt.plot(t,params[cid]['Tpar']/K2eV,'-r',label='Tpar')
 #plt.plot(t,params[cid]['Tperp']/K2eV,'-b',label='Tperp')
 plt.plot(t,params[cid]['Tperp1']/K2eV,'-g',label='Tperp1')
 plt.plot(t,params[cid]['Tperp2']/K2eV,'-b',label='Tperp2')
 plt.plot(t,params[cid]['Ttot']/K2eV,'-k',label='Ttot')
 plt.xlim(timeLims)
 plt.gcf().gca().xaxis.set_ticklabels([])
 plt.ylabel('T [eV]');
 plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
 plt.grid()

 plt.subplot(515)
 plt.plot(t,params[cid]['Bx']/1e-9,'-r',label='Bx')
 plt.plot(t,params[cid]['By']/1e-9,'-g',label='By')
 plt.plot(t,params[cid]['Bz']/1e-9,'-b',label='Bz')
 plt.plot(t,params[cid]['Btot']/1e-9,'-k',label='Btot')
 plt.xlim(timeLims)
 plt.xlabel('Time [s]')
 plt.ylabel('B [nT]');
 plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
 plt.grid() 
 
 plt.savefig(outFilePrefix + '_spectra_and_bulk_P' + str(ii+1).zfill(2) + outFileExtension,bbox_inches='tight')

