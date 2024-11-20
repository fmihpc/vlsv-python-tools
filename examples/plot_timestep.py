#!/usr/bin/python
# Plot a single vlasiator bulk file, can be used for continuous job monitoring.
# 
# Note: this requires python with matplotlib!
# On hornet, do:
#  module load tools/python/2.7.8
# before running this.

import matplotlib;
matplotlib.use('Agg')

from matplotlib.colors import LogNorm
import numpy as np;
import pytools as pt;
import pylab as pl;
import numpy.ma as ma;
import sys

if len(sys.argv) < 2:
	print("Syntax: python plot_timestep.py bulkfile.vlsv output.png")
	sys.exit()

# Open file
filename = sys.argv[1] 
outputname = sys.argv[2]
f = pt.vlsvfile.VlasiatorReader(filename)

# Themis colormap, as extracted from the themis tools' IDL file
hot_desaturated_colors=[(71./255.,71./255.,219./255.),(0,0,91./255.),(0,1,1),(0,.5,0),(1,1,0),(1,96./255,0),(107./255,0,0),(224./255,76./255,76./255)]
hot_desaturated_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("hot_desaturated",hot_desaturated_colors)

# Determine sizes
xsize = f.read_parameter("xcells_ini")
ysize = f.read_parameter("ycells_ini")
zsize = f.read_parameter("zcells_ini")
cellids = f.read_variable("CellID")

# Juggle fields into correct order
rho = f.read_variable("rho")
rho = rho[cellids.argsort()].reshape([ysize,xsize])

B = f.read_variable("B")
#B = f.read_variable("perturbed_B") + f.read_variable("B")
B_mag = np.array([np.linalg.norm(v) for v in B])
B_mag = B_mag[cellids.argsort()].reshape([ysize,xsize])

boundary_type = f.read_variable("Boundary_type");
boundary_type = boundary_type[cellids.argsort()].reshape([ysize,xsize])
boundary_type = (boundary_type != 1)

maskedrho = ma.masked_array(rho,mask=boundary_type)
maskedB = ma.masked_array(B_mag,mask=boundary_type)

rhomin = np.min(maskedrho)
rhomax = np.max(maskedrho)
Bmin = np.min(maskedB)
Bmax = np.max(maskedB)

# Plot rho and B
pl.rcParams['figure.figsize']= 30,38
#pl.rcParams['figure.DPI']= xcells_ini/30

ax1=pl.plt.subplot(2,1,1)
ax1.set_aspect('equal','datalim')
ax1.set_title("Rho (min = " + ("%.3e" % rhomin)+ ", max = " +("%.3e" % rhomax)+")")
fig1=ax1.pcolormesh(maskedrho, norm=LogNorm(vmin=1e4, vmax=1e7),cmap=hot_desaturated_colormap);
pl.plt.colorbar(fig1)

ax2=pl.plt.subplot(2,1,2,sharex=ax1, sharey=ax1)
ax2.set_title("B(min = " +("%.3e" % Bmin)+ ", max = " +("%.3e" % Bmax)+")")
fig2=ax2.pcolormesh(maskedB, norm=LogNorm(vmin=1e-9, vmax=1e-7),cmap=hot_desaturated_colormap);
pl.plt.colorbar(fig2)

#fig1.tight_layout()
#fig2.tight_layout()
pl.plt.savefig(outputname,bbox_inches='tight',dpi=xsize/30)

