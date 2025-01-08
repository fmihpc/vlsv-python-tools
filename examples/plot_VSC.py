#!/usr/bin/env python                                                                                                                              
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
import sys
import analysator as pt
from variable import get_data, get_name, get_units

# initialize lists for gathering values
profiles_B = []
profiles_Bx = []
profiles_By = []
profiles_Bz = []
profiles_rho = []
profiles_T = []
profiles_times = []

if len(sys.argv)==3:
    start= int(sys.argv[1])
    stop = int(sys.argv[2])
else:
    sys.stderr.write("Usage: plot_VSC.py <starting_index> <final_index+1> \n")
    sys.exit()

Re = 6.371e+6 # Earth radius in m
mp = 1.6726219e-27
mu0 = 1.256637e-6

inputLocation="/home/ad/ukko2-wrk/group/spacephysics/vlasiator/2D/ABC/bulk/"
outputLocation="./"
outputfile = outputLocation + "VSC_plot.png"

times=np.arange(start,stop)

# VSC position in metres
pos = [6.95e7, 3.6e7, 0.0]

cellid = None

for i,time in enumerate(times):
    
    
    timefull=str(time).rjust(7, '0')
    file_name=inputLocation+"bulk."+timefull+".vlsv" # EDIT

    print("Extracting timestep "+str(time)+" s")
    f=pt.vlsvfile.VlsvReader(file_name=file_name)

    profiles_times.append(f.read_parameter("time"))
    
    if cellid==None:
        cellid = f.get_cellid(pos)

    B = f.read_variable("B", cellids=cellid)
    profiles_B.append(np.linalg.norm(B)*1e9) # to nanotesla
    profiles_Bx.append(B[0]*1e9)
    profiles_By.append(B[1]*1e9)
    profiles_Bz.append(B[2]*1e9)
    profiles_rho.append(f.read_variable("rho", cellids=cellid)*1e-6) # to 1/cc
    profiles_T.append(f.read_variable("Temperature", cellids=cellid)*1e-6) #to MK

# Init figure    
fig = plt.figure()
fig.set_size_inches(6,9)

fig.add_subplot(4,1,1)
fig.add_subplot(4,1,2)
fig.add_subplot(4,1,3)
fig.add_subplot(4,1,4)

axes = fig.get_axes()
ax = axes[0]
for ax in axes[0:3]:
    for label in ax.xaxis.get_ticklabels():
        label.set_visible(False)
    ax.set_xlabel("")
axes[3].set_xlabel(r"Simulation time [s]")

lw = 2

# plot each profile

ax=axes[0]
ax.set_ylabel(r"$B$ [nT]", fontsize=16)
ax.plot(profiles_times,profiles_B, lw=lw, color='k', ls='solid', label="mag B")

ax=axes[1]
ax.set_ylabel(r"$B_i$ [nT]", fontsize=16)
ax.plot(profiles_times,profiles_Bx, lw=lw, color='green', ls='solid', label="Bx")
ax.plot(profiles_times,profiles_By, lw=lw, color='blue', ls='solid', label="By")
ax.plot(profiles_times,profiles_Bz, lw=lw, color='red', ls='solid', label="Bz")

ax=axes[2]
ax.set_ylabel(r"$n_p \,[\mathrm{cm}^{-3}]$", fontsize=16)
ax.plot(profiles_times,profiles_rho, lw=lw, color='k', ls='solid', label="rho")

ax=axes[3]
ax.set_ylabel(r"$T$ [MK]", fontsize=16)
ax.plot(profiles_times,profiles_T, lw=lw, color='k', ls='solid', label="T")

# legend = ax.legend(loc=1) #low right
fig.savefig(outputfile, bbox_inches='tight')
