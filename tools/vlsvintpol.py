#!/usr/bin/python
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

import pytools as pt
import numpy as np
import sys
import argparse
import logging


def extract_file(filename):
    out = []
    try:
        cellids=[]
        values=[]
        f=pt.vlsvfile.VlsvReader(filename)
        f.optimize_open_file()
        t=f.read_parameter("time")
        if t == None:
            t=f.read_parameter("t")
        if t == None:	    
            logging.info("Unknown time format in file " + filename)

        
        for coord in coords:
            if(args.re):
                cellids.append(f.get_cellid(coord * 6371000))
            else:
                cellids.append(f.get_cellid(coord))

        for i,var in enumerate(variables):
            values.append(f.read_variable(variables[i],operator=operators[i],cellids=cellids))
        
        for i,id in enumerate(cellids):
            out_line = str(t) + " " +  ' '.join(map(str, coords[i])) + " " + str(id)
            for j,varval in enumerate(values):
                out_line = out_line +  " " + str(varval[i])
            out.append([filename, out_line])
        f.optimize_close_file()
    except:
        out.append([filename, "#Could not read " + filename])
        pass
    
    return out


parser = argparse.ArgumentParser()
parser.add_argument('-var', nargs='*', help="a list of variable.operator's to output, e.g., v.magnitude rho B.x " )
parser.add_argument('-i', nargs='*', help="a list of vlsv files")
parser.add_argument('-c', help="A file with coordinates (can also be give from stdin)")
parser.add_argument('-re', action='store_true', help="Coordinates in RE, in meters by default")
parser.add_argument('-n', help="Number of processes to use, default 1")
args = parser.parse_args()


if args.var is None:
    #defaults
    varnames=["rho","B.magnitude","B.x","B.y","B.z"]
else:
    varnames=args.var

#read in variables and their operatorsx
operators=[]
variables=[]
for i,var in enumerate(varnames):
    varop=var.split(".")
    variables.append(varop[0])
    if len(varop)==1:
        operators.append("pass")
    else:
        operators.append(varop[1])

#read in coordinates
if args.c is None:
    coords = np.loadtxt(sys.stdin, dtype=np.float)
else:
    coords = np.loadtxt(args.c, dtype=np.float)

#if just single point make it into array with 1 row
coords = np.atleast_2d(coords)

if(args.re):
    print("#t X_RE Y_RE Z_RE CELLID " + " ".join(varnames))
else:
    print("#t X Y Z CELLID " + " ".join(varnames))

if args.n is None:
    numproc = 1
else:
    numproc = int(args.n)


## Parallel processing
from multiprocessing import Pool
if __name__ == '__main__':
   pool = Pool(numproc)
   return_array = pool.map(extract_file, args.i)


for i in sorted(return_array):
    for j in i:
        print(j[1])
