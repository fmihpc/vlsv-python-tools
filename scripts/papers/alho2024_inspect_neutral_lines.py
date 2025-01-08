# 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2024 University of Helsinki
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

""" This script is used to produce sliced plots in the local LMN coordinates,
as used in Alho+2024 https://doi.org/10.5194/angeo-42-145-2024 (Figures
8b, 8c, 10c, 10d).

   Usage: python alho2024_inspect_neutral_lines.py file_bulk out_dir
   Script expects the following arguments:
    param file_bulk: file to analyse; see required variables below
    param out_dir: output directory
    param cellids: -1 (all) or a whitespace-separated list of cellids

"""

import analysator as pt
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
from matplotlib.lines import Line2D
import os


if len(sys.argv) < 4:
   print("Usage: python alho2024_inspect_neutral_lines.py file_bulk out_dir ")
   print("Script expects the following arguments:")
   print(" param file_bulk: file to analyse; see required variables below")
   print(" param out_dir: output directory")
   print(" param cellids: -1 (all) or a whitespace-separated list of cellids")
   sys.exit()

else:
   try:
      fn = sys.argv[1]
   except:
      raise ValueError("Error reading file_bulk from arguments")
   assert os.path.isfile(fn), "file " + fn + " does not exist"
   try:
      outfolder = sys.argv[2]
   except:
      raise ValueError("Error reading out_dir from arguments")
   assert os.path.isdir(outfolder), "folder " + outfolder + " does not exist"
   try:
      cid0 = int(sys.argv[3])
      if cid0 > 0:
         cids_analyse = []
         for i in range(3,len(sys.argv)):
            cids_analyse.append(int(sys.argv[i]))
   except:
      raise ValueError("Error while reading CellIDs")

threshold = 0.36

# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

RE = 6371e3

flmn = pt.vlsvfile.VlsvReader(fn)

Ls = flmn.read_variable("LMN_magnetopause/vg_L")
Ms = flmn.read_variable("LMN_magnetopause/vg_M")
Ns = flmn.read_variable("LMN_magnetopause/vg_N")
ds = flmn.read_variable("vg_LMN_neutral_line_distance").squeeze()
Js = flmn.read_variable("vg_J")
d = flmn.read_variable("LMN_magnetopause/vg_jacobian_b")
Dimn = flmn.read_variable("vg_MDD_dimensionality")
CellIds = flmn.read_variable("CellID")

B = flmn.read_variable_as_fg("vg_b_vol")
allcoords = flmn.read_variable("vg_coordinates_cell_center")
dxs = flmn.read_variable_as_fg("vg_dxs")

# Example file caching system for slow operations - mapping to fg can be rather slow...
# but notably, the allcoords assignment used to be a list comprehension over CellIds,
# which is even slower. So here this caching is likely not even needed.

# picklefn ="inspect_neutrals_cache.npz"
# try:
#    loads = np.load(picklefn)
#    B = loads['B']
#    dxs = loads['dxs']
#    allcoords = loads['allcoords']
#    print("Loaded B,dxs")
# except:
#    B = flmn.read_variable_as_fg("vg_b_vol")
#    dxs = flmn.read_variable_as_fg("vg_dxs")
#    allcoords = flmn.read_variable("vg_coordinates_cell_center")
#    np.savez(picklefn, B=B, dxs=dxs,allcoords=allcoords)


boxed = np.all(np.abs(allcoords) < 40*6371e3,axis=-1) & (np.sum(allcoords**2,axis=-1) > (5.5*RE)**2)


hits = np.array(ds < threshold) & boxed
scatterpts = allcoords[hits,:]

if cid0 == -1:
   cids_analyse = CellIds[hits]

for ci,cid in enumerate(cids_analyse):
   print(cid)
   i = np.argwhere(CellIds == cid)
   # if(ds[i] > 0.36):
   #   continue
   dx = flmn.get_cell_dx(cid)
   reflevel = flmn.get_amr_level(cid)
   coords = allcoords[i,:].squeeze()

   M = (Ms[i,:]/np.linalg.norm(Ms[i,:])).squeeze()
   L = (Ls[i,:]/np.linalg.norm(Ls[i,:])).squeeze()
   N = (Ns[i,:]/np.linalg.norm(Ns[i,:])).squeeze()

   nd = 6
   low = coords - dx*nd
   hi = coords + dx*nd
   cp = np.dot(coords/RE,M)


   lowi, upi = flmn.get_bbox_fsgrid_slicemap(low,hi)

   Bsub = flmn.get_bbox_fsgrid_subarray(low,hi,B)

   dsub = dxs[lowi[0]:upi[0]+1, lowi[1]:upi[1]+1, lowi[2]:upi[2]+1,0]


   l0 = -np.min(dx)*nd
   l1 = +np.min(dx)*nd
   n0 = -np.min(dx)*nd
   n1 = +np.min(dx)*nd

   rot = np.stack([L,M,N])

   nfg = Bsub[:,:,:,0].shape
   [XmeshXY,YmeshXY] = np.meshgrid(np.linspace(l0/1e3,l1/1e3,num=nfg[0]+1),np.linspace(n0/1e3,n1/1e3,num=nfg[2]+1),indexing='xy')
   XmeshCentres = XmeshXY[:-1,:-1] + 0.5*(XmeshXY[0,1]-XmeshXY[0,0])
   YmeshCentres = YmeshXY[:-1,:-1] + 0.5*(YmeshXY[1,0]-YmeshXY[0,0])

   Bl = np.dot(Bsub,L)
   Bm = np.dot(Bsub,M)
   Bn = np.dot(Bsub,N)
   Bmag = np.linalg.norm(Bsub,axis=-1)


   c_in=0.5*np.array(Bl.shape)
   c_out = c_in
   offset=c_in-rot.T@c_out
   Bl = scipy.ndimage.affine_transform(Bl,np.linalg.inv(rot),mode='nearest',order=1,offset=offset)
   Bm = scipy.ndimage.affine_transform(Bm,np.linalg.inv(rot),mode='nearest',order=1,offset=offset)
   Bn = scipy.ndimage.affine_transform(Bn,np.linalg.inv(rot),mode='nearest',order=1,offset=offset)
   Bmag = (Bl**2 + Bn**2)**0.5
   dxsln = scipy.ndimage.affine_transform(dsub,np.linalg.inv(rot),mode='nearest',order=1,offset=offset)
   
   plt.close()

   fig = plt.figure(figsize=(16,12))
   ax_LN_down = fig.add_subplot(4,2,1)
   ax_LN_mid = fig.add_subplot(4,2,3)
   ax_LN_up = fig.add_subplot(4,2,5)
   ax_3d_overview = fig.add_subplot(4,2,(2,4), projection="3d")
   ax_LM = fig.add_subplot(4,2,7)
   ax_3d_detail = fig.add_subplot(4,2,(6,8), projection="3d")
   
   Nslice = (nfg[2]//4,-nfg[2]//4)
   Mslice = int(nfg[1]//2 - dx[1]/1e6)
   ax_LN_down.streamplot(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bl[:,Mslice,Nslice[0]:Nslice[1]].transpose(),Bn[:,Mslice,Nslice[0]:Nslice[1]].transpose())

   pcm = ax_LN_down.pcolormesh(XmeshXY[Nslice[0]:Nslice[1],:],YmeshXY[Nslice[0]:Nslice[1],:], 1e9*Bmag[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="nuuk_r")
   cb = plt.colorbar(pcm,ax=ax_LN_down)
   cb.set_label("$B_{LN}/\mathrm{nT}$")
   ax_LN_down.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],dxsln[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="gray_r", levels=[1.5e6, 3e6, 6e6])
   ax_LN_down.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bl[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="bam", levels=[0.0])
   ax_LN_down.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bn[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="bam_r",levels=[0.0])

   ax_LN_down.set_xlabel("L/km")
   ax_LN_down.set_ylabel("M = {:2.0f} km\nN/km".format(-dx[1]/1e3))
   ax_LN_down.grid()
   # cmap = mpl.cm.get_cmap("bam",256)
   cmap = mpl.colormaps.get_cmap("bam")
   custom_lines = [Line2D([0], [0], color="tab:blue", lw=2, label="$B_{LN}$"),
                     Line2D([0], [0], color=cmap(0.0), lw=2, label="$B_L = 0$"),
                     Line2D([0], [0], color=cmap(1.), lw=2, label="$B_N = 0$"),
                     Line2D([0], [0], color="gray", lw=2,label="$\Delta{}X_{vg}$")]
   ax_LN_down.legend(handles=custom_lines,loc="upper right")


   Mslice = int(nfg[1]//2)
   ax_LN_mid.streamplot(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bl[:,Mslice,Nslice[0]:Nslice[1]].transpose(),Bn[:,Mslice,Nslice[0]:Nslice[1]].transpose())

   pcm = ax_LN_mid.pcolormesh(XmeshXY[Nslice[0]:Nslice[1],:],YmeshXY[Nslice[0]:Nslice[1],:], 1e9*Bmag[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="nuuk_r")
   cb = plt.colorbar(pcm,ax=ax_LN_mid)
   cb.set_label("$B_{LN}/\mathrm{nT}$")
   ax_LN_mid.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],dxsln[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="gray_r", levels=[1.5e6, 3e6, 6e6])
   ax_LN_mid.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bl[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="bam", levels=[0.0])
   ax_LN_mid.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bn[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="bam_r",levels=[0.0])

   ax_LN_mid.set_xlabel("L/km")
   ax_LN_mid.set_ylabel("M = 0 km\nN/km")
   ax_LN_mid.legend(handles=custom_lines,loc="upper right")
   ax_LN_mid.grid()

   Mslice = int(nfg[1]//2 + dx[1]/1e6)
   ax_LN_up.streamplot(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bl[:,Mslice,Nslice[0]:Nslice[1]].transpose(),Bn[:,Mslice,Nslice[0]:Nslice[1]].transpose())

   pcm = ax_LN_up.pcolormesh(XmeshXY[Nslice[0]:Nslice[1],:],YmeshXY[Nslice[0]:Nslice[1],:], 1e9*Bmag[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="nuuk_r")
   cb = plt.colorbar(pcm,ax=ax_LN_up)
   cb.set_label("$B_{LN}/\mathrm{nT}$")
   ax_LN_up.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],dxsln[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="gray_r", levels=[1.5e6, 3e6, 6e6])
   ax_LN_up.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bl[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="bam", levels=[0.0])
   ax_LN_up.contour(XmeshCentres[Nslice[0]:Nslice[1],:],YmeshCentres[Nslice[0]:Nslice[1],:],Bn[:,Mslice,Nslice[0]:Nslice[1]].transpose(),cmap="bam_r",levels=[0.0])

   ax_LN_up.set_xlabel("L/km")
   ax_LN_up.set_ylabel("M = {:2.0f} km\nN/km".format(dx[1]/1e3))
   ax_LN_up.grid()
   custom_lines = [Line2D([0], [0], color="tab:blue", lw=2, label="$B_{LN}$"),
                     Line2D([0], [0], color=cmap(0.0), lw=2, label="$B_L = 0$"),
                     Line2D([0], [0], color=cmap(1.), lw=2, label="$B_N = 0$"),
                     Line2D([0], [0], color="gray", lw=2,label="$\Delta{}X_{vg}$")]
   ax_LN_up.legend(handles=custom_lines,loc="upper right")

   ax_LM.streamplot(XmeshCentres,YmeshCentres,Bl[:,:,nfg[2]//2].transpose(),Bm[:,:,nfg[2]//2].transpose())
   pcm = ax_LM.pcolormesh(XmeshXY,YmeshXY, 1e9*Bm[:,:,nfg[2]//2].transpose(),norm=mpl.colors.CenteredNorm(),cmap="vik")
   cb = plt.colorbar(pcm,ax=ax_LM)
   cb.set_label("$B_M/\mathrm{nT}$")
   ax_LM.contour(XmeshCentres,YmeshCentres,dxsln[:,:,nfg[2]//2].transpose(),cmap="gray_r", levels=[1.5e6, 3e6, 6e6])
   ax_LM.contour(XmeshCentres,YmeshCentres,Bl[:,:,nfg[2]//2].transpose(),cmap="bam", levels=[0.0])
   ax_LM.contour(XmeshCentres,YmeshCentres,Bn[:,:,nfg[2]//2].transpose(),cmap="bam_r",levels=[0.0])
   ax_LM.set_xlabel("L/km")
   ax_LM.set_ylabel("N = 0 km\nM/km")
   ax_LM.grid()
   ax_LM.legend(handles=custom_lines,loc="upper right")


   u = np.linspace(0, 2 * np.pi, 100)
   v = np.linspace(0, np.pi, 100)
   x = 1 * np.outer(np.cos(u), np.sin(v))
   y = 1 * np.outer(np.sin(u), np.sin(v))
   z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
   ax_3d_overview.plot_surface(x, y, z)
   sc = ax_3d_overview.scatter(allcoords[hits,0]/RE,allcoords[hits,1]/RE,allcoords[hits,2]/RE, c=d[hits,6], s=1, alpha=0.3, marker=',', cmap="roma_r",vmin=-1e-15,vmax=1e-15,label="X/O")
   cb = plt.colorbar(sc,ax=ax_3d_overview, pad=0.2)
   cb.set_label("$\partial{}_LB_N / \mathrm{Tm}^-1$ (positive is X)")
   ax_3d_overview.quiver(coords[0]/RE,coords[1]/RE,coords[2]/RE, L.squeeze()[0], L.squeeze()[1], L.squeeze()[2], color="r", length=20, label="L")
   ax_3d_overview.quiver(coords[0]/RE,coords[1]/RE,coords[2]/RE, M.squeeze()[0], M.squeeze()[1], M.squeeze()[2], color="g", length=20, label="M")
   ax_3d_overview.quiver(coords[0]/RE,coords[1]/RE,coords[2]/RE, N.squeeze()[0], N.squeeze()[1], N.squeeze()[2], color="b", length=20, label="N")
   ax_3d_overview.set_xlabel("$X/\mathrm{R_E}$")
   ax_3d_overview.set_ylabel("$Y/\mathrm{R_E}$")
   ax_3d_overview.set_zlabel("$Z/\mathrm{R_E}$")
   ax_3d_overview.legend(loc="upper right")
   set_axes_equal(ax_3d_overview)

   hits_small_x = hits & np.all(np.abs(allcoords-coords) < dx*nd,axis=-1) & (np.sum(allcoords**2,axis=-1) > (5.5*RE)**2) & (d[:,6] > 0)
   hits_small_o = hits & np.all(np.abs(allcoords-coords) < dx*nd,axis=-1) & (np.sum(allcoords**2,axis=-1) > (5.5*RE)**2) & (d[:,6] < 0)

   ncoords = (rot@((allcoords-coords).T)).T

   sc = ax_3d_detail.scatter(ncoords[hits_small_x,0]/RE,ncoords[hits_small_x,1]/RE,ncoords[hits_small_x,2]/RE, c=d[hits_small_x,6], s=40, marker='x', cmap="roma_r",norm=mpl.colors.CenteredNorm(),label="X")
   sc = ax_3d_detail.scatter(ncoords[hits_small_o,0]/RE,ncoords[hits_small_o,1]/RE,ncoords[hits_small_o,2]/RE, c=d[hits_small_o,6], s=10, marker='o', cmap="roma_r",norm=mpl.colors.CenteredNorm(),label="O")
   cb = plt.colorbar(sc,ax=ax_3d_detail, pad=0.2)
   cb.set_label("$\partial{}_LB_N / \mathrm{Tm}^-1$ (positive is X)")
   Ll = rot@L
   Mm = rot@M
   Nn = rot@N
   ax_3d_detail.quiver(0,0,0, Ll.squeeze()[0], Ll.squeeze()[1], Ll.squeeze()[2], color="r", length=3, label="L")
   ax_3d_detail.quiver(0,0,0, Mm.squeeze()[0], Mm.squeeze()[1], Mm.squeeze()[2], color="g", length=3, label="M")
   ax_3d_detail.quiver(0,0,0, Nn.squeeze()[0], Nn.squeeze()[1], Nn.squeeze()[2], color="b", length=3, label="N")
   ax_3d_detail.set_xlabel("$L/\mathrm{R_E}$")
   ax_3d_detail.set_ylabel("$M/\mathrm{R_E}$")
   ax_3d_detail.set_zlabel("$N/\mathrm{R_E}$")
   ax_3d_detail.legend(loc="upper right")
   set_axes_equal(ax_3d_detail)

   xostr = ""
   # Unused, but you can sort hits by gradient strengths to different folder here
   # (create the folders)
   if (d[i,6].squeeze()> 0):
      xostr = " (X)"
      if (d[i,6].squeeze()*dx[0] > 1e-14):
         folder = "X_1e-0"
      elif (d[i,6].squeeze()*dx[0] > 1e-15):
         folder = "X_1e-1"
      elif (d[i,6].squeeze()*dx[0] > 1e-16):
         folder = "X_1e-2"
      elif (d[i,6].squeeze()*dx[0] > 1e-17):
         folder = "X_1e-3"
      else:
         folder = "X_low"
   else:
      xostr = " (O)"
      if (d[i,6].squeeze()*dx[0] < -1e-14):
         folder = "O_1e-0"
      elif (d[i,6].squeeze()*dx[0] < -1e-15):
         folder = "O_1e-1"
      elif (d[i,6].squeeze()*dx[0] < -1e-16):
         folder = "O_1e-2"
      elif (d[i,6].squeeze()*dx[0] < -1e-17):
         folder = "O_1e-3"
      else:
         folder = "O_low"
   folder = ""

   plt.suptitle("CellID {:d}".format(cid) + 
               ", J = ({:3e},{:3e},{:3e}), {:3e} A".format(*tuple(Js[i,:].squeeze()),np.linalg.norm(Js[i,:].squeeze())) +
               "\nr/RE = [{0:.1f}, {1:.1f}, {2:.1f}]; dBNdL = {3:.3e}".format(*tuple(coords/RE),d[i,6].squeeze()) +
               xostr + ", ds = {:.2f}".format(ds[i].squeeze()) +
               ", Di = ({0:.2f},{1:.2f},{2:.2f})".format(*tuple(Dimn[i,:].squeeze())) +
               "\n LMN: ({0:.2f},{1:.2f},{2:.2f}),({3:.2f},{4:.2f},{5:.2f}),({6:.2f},{7:.2f},{8:.2f})".format(*tuple(L),*tuple(M),*tuple(N)))
   
   plt.savefig(outfolder+'/'+folder+"/fig{:012d}.png".format(cid), dpi=150)
   plt.close()
