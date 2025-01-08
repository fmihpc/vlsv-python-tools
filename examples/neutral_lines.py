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

''' This is an example file to produce LMN coordinates and FOTE null line
metrics as in Alho+2024. Thresholding the output data with ``vg_lmn_neutral_line_distance ~< 0.2`` 
is the suggested way of detecting neutral lines with the FOTE method.

Usage:
    python neutral_lines.py file_in file_out

    file_in:
        Input vlsv file with at least the following data: ``CellID``, ``vg_b_vol``, ``vg_jacobian_b``, and perturbed
        B jacobian components ``vg_dperbxvoldx``
    
    file_out:
        Output vlsv file (will be overwritten). 


    
Input parameters:
    file_in: 

'''

import analysator as pt
import numpy as np
import sys


def main():
    f = pt.vlsvfile.VlsvReader(fn)

    cids = f.read_variable("CellID")

    fw = pt.vlsvfile.VlsvWriter(f, fnout, copy_meshes="SpatialGrid")

    # Some foundational variables to copy over
    fw.copy_variables(f,["CellID",
                         "vg_b_vol",
                         "vg_jacobian_B"])
    
    # These come nicely via the reduction pipelines
    fw.copy_variables(f, ["vg_j",
                          "vg_mdd_dimensionality",
                           "vg_lmn_neutral_line_distance",
                           "vg_lmn_L_flip_distance"])

    LMNs = f.read_variable("vg_lmn",cids)
    fw.write(LMNs.reshape((-1,9)), "vg_LMN", "VARIABLE","SpatialGrid")

    # Read and rotate the B jacobian to the local LMN coordinates.
    LMN_jacob = f.read_variable("vg_jacobian_B", cids)
    LMN_jacob = np.reshape(LMN_jacob,(LMN_jacob.shape[0],3,3))
    LMN_jacob = np.transpose(LMNs,(0, 2, 1)) @ LMN_jacob @ LMNs

    # Extract and write the X/O discriminating partial derivative
    dBNdL = LMN_jacob[:,2,0]
    fw.write(dBNdL, "vg_dBNdL", "VARIABLE", "SpatialGrid")

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python neutral_lines.py file_in file_out ")
        print("Script expects the following arguments:")
        print(" param file_in: input vlsv file")
        print(" param file_out: output vlsv file")
        sys.exit()
    

    fn = sys.argv[1]
    fnout = sys.argv[2]

    main()
