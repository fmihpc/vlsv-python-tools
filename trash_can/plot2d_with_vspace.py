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

# 2D plotting

def vlsv_plot2d_with_vspace(vlsvReader,varName="rho",withDistr=0,Nstride=97,Nbins=10,distrMinMax=[0,10],lengthUnit=1,xFigSize=15,yFigSize=14,xFigSizeDistr=0.015,yFigSizeDistr=0.015):
 ''' 2-dimensional pseudocolor plot (with velocity space histogram subplots)
 :param vlsvReader:    VLSV reader interface
 :param varName:       Variable name if VLSV reader
 :param withDistr:     Plot distribution sub plots on top of pseudocolor (0/1)
 :param Nstride:       Check every Nstride\'th spatial cell for a distribution
 :param Nbins:         Number of distribution histogram bins
 :param lengthUnit:    Length unit of axes
 :param xFigSize:      Figure size in horizontal direction
 :param yFigSize:      Figure size in vertical direction
 :param xFigSizeDistr: Distribution plot size in horizontal direction
 :param yFigSizeDistr: Distribution plot size in vertical direction
 :returns: nothing
 .. code-block:: python
 # Example usage:
   import analysator as pt
   fvlsv = pt.vlsvfile.VlsvReader("/path/to/run/bulk.0003000.vlsv")
   pt.plot.vlsv_plot2d_with_vspace(fvlsv,varName="rho",lengthUnit=6371e3)
   pt.plot.vlsv_plot2d_with_vspace(fvlsv,varName="rho",lengthUnit=6371e3,withDistr=1,Nstride=97,xFigSize=15,yFigSize=14)
  '''
 import numpy as np
 import matplotlib.pyplot as plt
 import operator as oper
 # read file header
 xmin = vlsvReader.read_parameter("xmin")
 xmax = vlsvReader.read_parameter("xmax")
 ymin = vlsvReader.read_parameter("ymin")
 ymax = vlsvReader.read_parameter("ymax")
 zmin = vlsvReader.read_parameter("zmin")
 zmax = vlsvReader.read_parameter("zmax")
 xsize = xmax - xmin
 ysize = ymax - ymin
 zsize = zmax - zmin
 nx = vlsvReader.read_parameter("xcells_ini")
 ny = vlsvReader.read_parameter("ycells_ini")
 nz = vlsvReader.read_parameter("zcells_ini")
 dx = xsize/nx
 dy = ysize/ny
 dz = zsize/nz
 
 # check if the file has xy or xz plane
 if(ny > nz and nz == 1):
  plane = "xy"
  meshX,meshY = np.meshgrid(np.linspace(xmin/lengthUnit,xmax/lengthUnit,nx),np.linspace(ymin/lengthUnit,ymax/lengthUnit,ny))
 elif(ny < nz and ny == 1):
  plane = "xz"
  meshX,meshY = np.meshgrid(np.linspace(xmin/lengthUnit,xmax/lengthUnit,nx),np.linspace(zmin/lengthUnit,zmax/lengthUnit,nz))
 else:
  print("Error: cannot determine simulation plane")
  return
 # cell id - index  dict
 locs = vlsvReader.get_cellid_locations()
 cellids = list(locs.keys())
 # open variable
 if varName in vlsvReader.get_all_variables():
  var = vlsvReader.read_variable(varName)
 else:
  print("Variable " + varName + " not found")
  return
 # sort variable array according to cell ids
 locs_sorted = sorted(locs.items(), key=oper.itemgetter(0))
 var_sorted = []
 for i in locs_sorted:
  var_sorted.append(var[i[1]])
 # Shape variable array as 2-dimensional matrix
 if(plane == "xz"):
  var = np.asarray(var_sorted).reshape(nx,nz) 
 if(plane == "xy"):
  var = np.asarray(var_sorted).reshape(nx,ny)
 # do plotting
 plt.figure(figsize=(xFigSize,yFigSize))
 plt.pcolormesh(meshX,meshY,var)
 plt.axis('equal')
 plt.xlabel("x [" + str(lengthUnit/1e3) + " km]")
 plt.xlim([xmin/lengthUnit,xmax/lengthUnit])
 if(plane == "xz"):
  plt.ylim([zmin/lengthUnit,zmax/lengthUnit])
  plt.ylabel("z [" + str(lengthUnit/1e3) + " km]")
 if(plane == "xy"):
  plt.ylim([ymin/lengthUnit,ymax/lengthUnit])
  plt.xlabel("y [" + str(lengthUnit/1e3) + " km]")
 plt.title(varName)
 if withDistr == 0:
  plt.colorbar()
  plt.show()
  return
 # distribution plotting
 # coordinates and size of current axis
 xfig0 = plt.gca().get_position().get_points()[0,0]
 yfig0 = plt.gca().get_position().get_points()[0,1]
 xfig1 = plt.gca().get_position().get_points()[1,0]
 yfig1 = plt.gca().get_position().get_points()[1,1]
 xfigsize = xfig1-xfig0
 yfigsize = yfig1-yfig0
 if vlsvReader.check_variable("fSaved") == False:
  print("Error: Variable fSaved not found")
  return
 cids_with_vel = ()
 n=0
 Cloop = range(0,len(cellids),Nstride)
 for i in Cloop:
  cid = cellids[i]
  n=n+1
  # check if velocity space exists in this cell
  if vlsvReader.read_variable("fSaved",cid) == 1.0:
   x,y,z = vlsvReader.get_cell_coordinates(cid)
   cids_with_vel = np.append(cids_with_vel,cid)
   velcells = vlsvReader.read_velocity_cells(cid)
   jj = list(velcells.keys())
   # default subplots: histogram of kinetic energy
   mp = 1.6726219e-27
   qe = 1.60217662e-19
   V = vlsvReader.get_velocity_cell_coordinates(jj)
   Vtot2 = (np.sum(np.square(V),1))
   Ekin = 0.5*mp*Vtot2/qe/1e3 # keV
   f = list(zip(*velcells.items()))
   # check that velocity space has cells
   if(len(f) > 0):
    f = np.asarray(f[1])
   else:
    continue
   xsub = xfig0 + xfigsize*(x-xmin)/xsize
   ysub = yfig0 + yfigsize*(z-zmin)/zsize
   print("Subplot " +  str(n) + "/" + str(len(Cloop)) + ": fig crds = " + str(xsub) + ", " + str(ysub))
   ax = plt.axes([xsub,ysub,xFigSizeDistr,yFigSizeDistr])
   ax.hist(Ekin,bins=Nbins,weights=f,normed=1,log=1,color='r',edgecolor='none',range=distrMinMax)
   ax.xaxis.set_major_locator(plt.NullLocator())
   ax.yaxis.set_major_locator(plt.NullLocator())
   plt.xlim(distrMinMax)
 plt.show()

