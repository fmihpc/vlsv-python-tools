#
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2021 University of Helsinki
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
import numpy as np
import analysator
import logging
# Function to reduce the velocity space in a spatial cell to an omnidirectional energy spectrum
# Weighted by particle flux/none
def get_spectrum_energy(vlsvReader,
                        cid,
                        population="proton",
                        fMin = 1e-21,
                        EMin=100,
                        EMax=80e3,
                        nBins=66,
                        mass=1.6726219e-27, # default: mp
                        weight='flux',
                        differential=True, # weigh by eV
                        bindifferential=False, # weigh by d(eV)
                        restart=True):

   EkinBinEdges = np.logspace(np.log10(EMin),np.log10(EMax),nBins+1, endpoint=True)
   dE = EkinBinEdges[1:] - EkinBinEdges[:-1]
   vlsvReader = analysator.vlsvfile.VlsvReader(vlsvReader)
   # check if velocity space exists in this cell
   if not vlsvReader.check_variable("moments"): # restart files have VDFs everywhere
      if vlsvReader.check_variable("fSaved"):
         if not vlsvReader.read_variable('fSaved',cid):
            return (False,np.zeros(nBins), EkinBinEdges)
      elif vlsvReader.check_variable("vg_f_saved"):
         if not vlsvReader.read_variable('vg_f_saved',cid):
            return (False,np.zeros(nBins), EkinBinEdges)
      else:
         logging.info("Error finding cells with VDFs!")

   if vlsvReader.check_variable('MinValue'):
      fMin = vlsvReader.read_variable('MinValue',cid)
   elif vlsvReader.check_variable(population+'/effectivesparsitythreshold'):
      fMin = vlsvReader.read_variable(population+'/effectivesparsitythreshold',cid)
   elif vlsvReader.check_variable(population+'/vg_effectivesparsitythreshold'):
      fMin = vlsvReader.read_variable(population+'/vg_effectivesparsitythreshold',cid)

   #logging.info('Cell ' + str(cid).zfill(9))
   velcells = vlsvReader.read_velocity_cells(cid, population)
   V = vlsvReader.get_velocity_cell_coordinates(list(velcells.keys()), pop=population)
   V2 = np.sum(np.square(V),1)
   JtoeV = 1/1.60217662e-19
   Ekin = 0.5*mass*V2*JtoeV
   f = list(zip(*velcells.items()))


   # check that velocity space has cells - still return a zero histogram in the same shape
   if(len(f) > 0):
      f = np.asarray(f[1])
   else:
      return (False,np.zeros(nBins), EkinBinEdges)
   ii_f = np.where(np.logical_and(f >= fMin, Ekin > 0))
   if len(ii_f) < 1:
      return (False,np.zeros(nBins), EkinBinEdges)
   f = f[ii_f]
   Ekin = Ekin[ii_f]
   V2 = V2[ii_f]

   dV3 = np.prod(vlsvReader.get_velocity_mesh_dv(population))
   # normalization
   if (differential): # differential flux per eV
      if(weight == 'flux'):
         fw = f * np.sqrt(V2) * dV3 / (4*np.pi * Ekin) # use particle flux as weighting
         units = "1/(s m^2 sr eV)"
         latexunits = r'$\mathrm{s}^-1\,\mathrm{m}^{-2}\,\mathrm{sr}^{-1}\,\mathrm{eV}^{-1}$'
         latex=r'$\mathcal{F}(\vec{r},E)\,E^{-1}$'
      else:
         fw = f * dV3 / Ekin
         units = "1/(m^3 eV)"
         latexunits = r'$\mathrm{m}^{-3}\,\mathrm{eV}^{-1}$'
         latex=r'$f(\vec{r},E)\,E^{-1}$'
         weight = 'particles'
   elif (bindifferential): # differential flux per d(eV)
      if(weight == 'flux'):
         fw = f * np.sqrt(V2) * dV3 / (4*np.pi) # use particle flux as weighting
         units = "1/(s m^2 sr eV)"
         latexunits = r'$\mathrm{s}^-1\,\mathrm{m}^{-2}\,\mathrm{sr}^{-1}\,\delta\mathrm{eV}^{-1}$'
         latex=r'$\mathcal{F}(\vec{r},E)\,\Delta{E^{-1}}$'
      else:
         fw = f * dV3
         units = "1/(m^3 eV)"
         latexunits = r'$\mathrm{m}^{-3}\,\delta\mathrm{eV}^{-1}$'
         latex=r'$f(\vec{r},E)\,\Delta{E^{-1}}$'
         weight = 'particles'
   else:
      if(weight == 'flux'):
         fw = f * np.sqrt(V2) * dV3 / (4*np.pi) # use particle flux as weighting
         units = "1/(s m^2 sr)"
         latexunits = r'$\mathrm{s}^-1\,\mathrm{m}^{-2}\,\mathrm{sr}^{-1}$'
         latex=r'$\mathcal{F}(\vec{r},E)$'
      else:
         fw = f * dV3
         units = "1/(m^3)"
         latexunits = r'$\mathrm{m}^{-3}$'
         latex=r'$f(\vec{r},E)$'
         weight = 'particles'

   #Ekin[Ekin < min(EkinBinEdges)] = min(EkinBinEdges)
   #Ekin[Ekin > max(EkinBinEdges)] = max(EkinBinEdges)

   # compute histogram
   (nhist,edges) = np.histogram(Ekin,bins=EkinBinEdges,weights=fw,normed=0)

   if (bindifferential): # finish differential flux per d(eV)
      nhist = np.divide(nhist,dE)

   vari = analysator.calculations.VariableInfo(nhist,
                                       name="Omnidirectional differential energy spectrum "+population+' ('+weight+')',
                                       units=units,
                                       latex=latex,
                                       latexunits=latexunits)
   return (True,vari,edges)


# Function to reduce the velocity space in a spatial cell to a distribution in velocity along a vector
def get_spectrum_alongaxis_vel(vlsvReader,
                               cid,
                               population="proton",
                               vector=None,
                               vectorVar="vg_b_vol",
                               fMin=1e-21,
                               VMin=-2e6,
                               VMax=2e6,
                               nBins=200,
                               differential=True, # weigh by velocity
                               bindifferential=False, # weigh by d(velocity)
                               restart=True):

   vlsvReader = analysator.vlsvfile.VlsvReader(vlsvReader)

   if vectorVar is not None and vector is None:
      if vlsvReader.check_variable(vectorVar):
         vector=vlsvReader.read_variable(vectorVar, cid)
      else:
         # failsafe
         if vlsvReader.check_variable("B"):
            vector = vlsvReader.read_variable('B', cid)
         elif vlsvReader.check_variable("vg_b_vol"):
            vector = vlsvReader.read_variable('vg_b_vol', cid)
         elif vlsvReader.check_variable("fg_b"):
            coordinates = vlsvReader.get_cell_coordinates(cid)
            vector = vlsvReader.read_interpolated_fsgrid_variable('fg_b', coordinates)

   vector = vector/np.linalg.norm(vector)
   VBinEdges = np.linspace(VMin, VMax, nBins+1, endpoint=True)

   # check if velocity space exists in this cell
   if not vlsvReader.check_variable("moments"): # restart files have VDFs everywhere
      if vlsvReader.check_variable("fSaved"):
         if not vlsvReader.read_variable('fSaved',cid):
            return (False,np.zeros(nBins), VBinEdges)
      elif vlsvReader.check_variable("vg_f_saved"):
         if not vlsvReader.read_variable('vg_f_saved',cid):
            return (False,np.zeros(nBins), VBinEdges)
      else:
         logging.info("Error finding cells with VDFs!")

   if vlsvReader.check_variable('MinValue'):
      fMin = vlsvReader.read_variable('MinValue',cid)
   elif vlsvReader.check_variable(population+'/effectivesparsitythreshold'):
      fMin = vlsvReader.read_variable(population+'/effectivesparsitythreshold',cid)
   elif vlsvReader.check_variable(population+'/vg_effectivesparsitythreshold'):
      fMin = vlsvReader.read_variable(population+'/vg_effectivesparsitythreshold',cid)

   #logging.info('Cell ' + str(cid).zfill(9))
   velcells = vlsvReader.read_velocity_cells(cid, population)
   V = vlsvReader.get_velocity_cell_coordinates(list(velcells.keys()), pop=population)
   V2 = np.sum(np.square(V),1)
   Vproj = np.dot(V,vector) # safe because "vector" is a 1-D array
   f = list(velcells.values())

   # check that velocity space has cells - still return a zero histogram in the same shape
   if(len(f) > 0):
      f = np.asarray(f)
   else:
      return (False,np.zeros(nBins), VBinEdges)
   ii_f = np.where(f >= fMin)
   #ii_f = np.where(f >= fMin or True)
   if len(ii_f) < 1:
      return (False,np.zeros(nBins), VBinEdges)
   #f = f[ii_f]
   #Vproj = Vproj[ii_f]
   #V2 = V2[ii_f]

   # truncate
   Vproj[Vproj < min(VBinEdges)] = min(VBinEdges)
   Vproj[Vproj > max(VBinEdges)] = max(VBinEdges)

   dV3 = np.prod(vlsvReader.get_velocity_mesh_dv(population))
   fw = f*dV3
   # normalization
   if (differential): # differential flux per [m/s]
      units = "s/m^4"
      latexunits = r'$\mathrm{s}\mathrm{m}^{-4}$'
      latex=r'$f(\vec{r},v)\,v^{-1}$'
      weight = 'particles'
   elif (bindifferential): # differential flux per d[m/s]
      units = "s/m^4"
      latexunits = r'$\mathrm{s}\mathrm{m}^{-4}$'
      latex=r'$f(\vec{r},v)\,\Delta{v^{-1}}$'
      weight = 'particles'
   else:
      units = "1/m^3"
      latexunits = r'$\mathrm{m}^{-3}$'
      latex=r'$f(\vec{r},v)$'
      weight = 'particles'

   (nhist,edges) = np.histogram(Vproj,bins=VBinEdges,weights=fw,normed=0)
   # normalization
   dv = abs(VBinEdges[1:] - VBinEdges[:-1])
   if (differential): # differential flux per [m/s]
      midv = VBinEdges[1:] + 0.5 * dv
      nhist = np.divide(nhist,midv)
   elif (bindifferential): # differential flux per d[m/s]
      nhist = np.divide(nhist,dv)

   vari = analysator.calculations.VariableInfo(nhist,
                                    name="Parallel distribution of "+population+' ('+weight+')',
                                    units=units,
                                    latex=latex,
                                    latexunits=latexunits)

   return (True,vari,edges)
