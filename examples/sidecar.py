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

'''
This file contains an example on how to use the vlsvReader and vlsvWriter
functionalities to derive quantities (especially differential operations 
on an AMR mesh via uniform grid resampling) from .vlsv variables and store
them to a sidecar file AMR grid.

The file first defines a reader for input data and a writer for the output,
and copies over a list of variables desired for the output (CellID is 
definitely worth copying over). A definition for curl on fsgrid is given,
after which fg_b (a face-centered field) is re-centered as volumetric. J is
then calculated on fsgrid from the volumetric B, which is then downsampled 
onto SpatialGrid via averaging and written into the sidecar file.

Specific to this script is the combining of data from two vlsv files: one containing
the run-time data, and another containing the constant background magnetic field and
its Jacobian. 

A Script like this can be adapted for personal use, and
passed to a job submission system for generating a multitude of sidecars
via e.g. array jobs.
'''

import analysator as pt
import numpy as np
import sys
import time

q = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27
eps0 = 8.854e-12 


def fg_vol_curl(reader, array):
   dx = reader.get_fsgrid_cell_size()
   dummy,  dFx_dy, dFx_dz = np.gradient(array[:,:,:,0], *dx)
   dFy_dx, dummy,  dFy_dz = np.gradient(array[:,:,:,1], *dx)
   dFz_dx, dFz_dy, dummy  = np.gradient(array[:,:,:,2], *dx)

   rotx = dFz_dy - dFy_dz
   roty = dFx_dz - dFz_dx
   rotz = dFy_dx - dFx_dy

   return np.stack([rotx, roty, rotz], axis=-1)

def fg_vol_jacobian(reader):
   if ("fg_dbgbxvoldy" in reader.get_all_variables()):
      dx = reader.get_fsgrid_cell_size() # these are stored as differences for now
      dFx_dx = reader.read_fsgrid_variable("fg_dbgbxvoldx")/dx[0]
      dFx_dy = reader.read_fsgrid_variable("fg_dbgbxvoldy")/dx[1]
      dFx_dz = reader.read_fsgrid_variable("fg_dbgbxvoldz")/dx[2]
      dFy_dx = reader.read_fsgrid_variable("fg_dbgbyvoldx")/dx[0]
      dFy_dy = reader.read_fsgrid_variable("fg_dbgbyvoldy")/dx[1]
      dFy_dz = reader.read_fsgrid_variable("fg_dbgbyvoldz")/dx[2]
      dFz_dx = reader.read_fsgrid_variable("fg_dbgbzvoldx")/dx[0]
      dFz_dy = reader.read_fsgrid_variable("fg_dbgbzvoldy")/dx[1]
      dFz_dz = reader.read_fsgrid_variable("fg_dbgbzvoldz")/dx[2]
   else:
      dx = reader.get_fsgrid_cell_size()
      array = reader.read_variable_as_fg("vg_b_vol")

      dFx_dx, dFx_dy, dFx_dz = np.gradient(array[:,:,:,0], *dx)
      dFy_dx, dFy_dy, dFy_dz = np.gradient(array[:,:,:,1], *dx)
      dFz_dx, dFz_dy, dFz_dz = np.gradient(array[:,:,:,2], *dx)

   return np.stack(np.array(
                    [dFx_dx, dFx_dy, dFx_dz,
                     dFy_dx, dFy_dy, dFy_dz,
                     dFz_dx, dFz_dy, dFz_dz]
                   ),
                   axis=-1)

def vg_vol_perb_jacobian(reader):
   if ("vg_dperbxvoldy" in reader.get_all_variables()):
      dx = reader.get_fsgrid_cell_size() # these are stored as differences for now
      dFx_dx = reader.read_variable("vg_dperbxvoldx")
      dFx_dy = reader.read_variable("vg_dperbxvoldy")
      dFx_dz = reader.read_variable("vg_dperbxvoldz")
      dFy_dx = reader.read_variable("vg_dperbyvoldx")
      dFy_dy = reader.read_variable("vg_dperbyvoldy")
      dFy_dz = reader.read_variable("vg_dperbyvoldz")
      dFz_dx = reader.read_variable("vg_dperbzvoldx")
      dFz_dy = reader.read_variable("vg_dperbzvoldy")
      dFz_dz = reader.read_variable("vg_dperbzvoldz")

   return np.stack(np.array(
                    [dFx_dx, dFx_dy, dFx_dz,
                     dFy_dx, dFy_dy, dFy_dz,
                     dFz_dx, dFz_dy, dFz_dz]
                   ),
                   axis=-1)


def main():

   f_bgb = pt.vlsvfile.VlsvReader(fn_bgb)
   f_perb = pt.vlsvfile.VlsvReader(fn_perb)

   writer = pt.vlsvfile.VlsvWriter(f_bgb, fnout)

   print(f_bgb.get_all_variables())
   print(f_bgb)
   cellIds=f_bgb.read_variable("CellID")



   t = time.time()
   print("copy_var_list")
   #writer.copy_variables_list(f, ["CellID","vg_b_vol"])
   writer.copy_variables_list(f_bgb, ["CellID"])
   print('writer init and var copy, elapsed:', time.time()-t)


   # print('Processing for J, elapsed', time.time()-t)
   # fg_b_vol=f.read_fg_variable_as_volumetric('fg_b')
   # print('fgbvol read, shape ', fg_b_vol.shape, 'elapsed:', time.time()-t)

   # fg_J = fg_vol_curl(f, fg_b_vol)/1.25663706144e-6
   # print('fgbvol curled, elapsed:', time.time()-t)

   f_bgb.map_vg_onto_fg()
   print('vg mapped to fg, elapsed:', time.time()-t)
   argsorti=cellIds.argsort()
   rev_argosorti=argsorti.argsort()

   fg_b_jacob = fg_vol_jacobian(f_bgb)
   fg_b_vol = f_bgb.read_fsgrid_variable("fg_b_background_vol")

   vg_bgb_jacobian = f_bgb.fsgrid_array_to_vg(fg_b_jacob)
   vg_bgb_jacobian = np.reshape(vg_bgb_jacobian, (vg_bgb_jacobian.shape[0],9))
   fooids = f_perb.read_variable("CellID")
   # vg_bgb_jacobian = vg_bgb_jacobian[fooids.argsort()]
   # vg_bgb_jacobian = vg_bgb_jacobian[rev_argosorti]
   print('fg_J mapped to vg, elapsed:', time.time()-t)

   vg_perb_jacobian = vg_vol_perb_jacobian(f_perb)
   vg_perb_jacobian = np.reshape(vg_perb_jacobian, (vg_perb_jacobian.shape[0],9))
   vg_perb_jacobian = vg_perb_jacobian[fooids.argsort()]
   vg_perb_jacobian = vg_perb_jacobian[rev_argosorti]

   writer.write(vg_bgb_jacobian+vg_perb_jacobian,'vg_jacobian_B','VARIABLE','SpatialGrid')

   vg_b_vol = f_bgb.fsgrid_array_to_vg(fg_b_vol)
   writer.write(vg_b_vol,'vg_b_vol','VARIABLE','SpatialGrid')

   print('J written, elapsed:', time.time()-t)


if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print("Usage: python neutral_lines.py file_in file_out ")
        print("Script expects the following arguments:")
        print(" param file_in_bgb: input vlsv file with background b and its jacobians")
        print(" param file_in: input vlsv file with the perturbed b-fields")
        print(" param file_out: output vlsv file")
        sys.exit()
    

    fn_bgb = sys.argv[1]
    fn_perb = sys.argv[2]
    fnout = sys.argv[3]

    main()
