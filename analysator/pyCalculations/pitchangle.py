#
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2019 University of Helsinki
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

import analysator as pt
import numpy as np
import sys, os
from output import output_1d
import logging

def pitch_angles( vlsvReader,
                     cellid,
                     nbins=10,
                     filename=None,
                     filedir=None, step=None,
                     outputdir=None, outputfile=None,
                     cosine=False,
                     plasmaframe=False,
                     vcut=None,
                     vcutmax=None,
                     pop="proton"):

   ''' Calculates the pitch angle distribution for a given cell

      :param vlsvReader:        Some VlsvReader class with a file open. Can be overriden by keywords.
      :param cellid:            The cell id whose pitch angle the user wants
                              NOTE: The cell id must have a velocity distribution!
      :kword filename:          path to .vlsv file to use for input.
      :kword filedir:           Optionally provide directory where files are located and use step for bulk file name
      :kword step:              output step index, used for constructing output (and possibly input) filename

      :kword nbins:             How many bins to use for the distribution
      :kword cosine:            True if returning the pitch angles as a cosine(alpha) plot [-1,1].
                              If false, returns as pitch angle in degrees [0,180].

      :kword plasmaframe:       True if the user wants to get the pitch angle distribution
                              in the plasma frame (for this population).
                              If set to a string, will try to use the string as a variable for
                              the frame to transform into.
                              If set to a 3-element vector, will use that frame instead.

      :kword vcut:              Set to True to ignore velocity cells below 2x the thermal speed.
                              If set to a number, will use that velocity in m/s instead.
      :kword vcutmax:           Set to True to ignore velocity cells above 2x the thermal speed.
                              If set to a number, will use that velocity in m/s instead.

      :kword outputdir:         Optional (recommended) method to save results to a file in the given directory.
                              If directory does not exist, it will be created. Filenames within directory are
                              generated automatically.
      :kword outputfile:        Provide exact output file name (including complete path)

      :kword pop:               Active population, defaults to proton (avgs)

      :returns: pitch angles and avgs [pitch_angles, avgs]

      .. code-block:: python

         # Example usage:
         vlsvReader = VlsvReader("restart.0000798.vlsv")
         result = pitch_angles( vlsvReader=vlsvReader, 1924, cosine=True,
                              plasmaframe=True, outputdir="/wrk/username/pitchangledirectory/" )
   '''

   # Input file or object
   if filename!=None:
      vlsvReader=pt.vlsvfile.VlsvReader(filename)
   elif ((filedir!=None) and (step!=None)):
      filename = filedir+'bulk.'+str(step).rjust(7,'0')+'.vlsv'
      vlsvReader=pt.vlsvfile.VlsvReader(filename)

   # Transform to a different frame?
   frame = [0.,0.,0.]
   if plasmaframe is not False:
      if isinstance(plasmaframe, str):
         frame = vlsvReader.read_variable(plasmaframe, cellid)
      elif plasmaframe is True:
         if vlsvReader.check_variable("moments"): # restart
            frame = vlsvReader.read_variable('restart_V', cellid)
         elif vlsvReader.check_variable("V"):
            frame = vlsvReader.read_variable('V', cellid)
         elif vlsvReader.check_variable("vg_V"):
            frame = vlsvReader.read_variable('vg_V', cellid)
         elif vlsvReader.check_variable(pop+"/vg_V"):
            frame = vlsvReader.read_variable(pop+'/vg_V', cellid)
         else:
            logging.info("Error extracting plasma frame velocity!")
      elif len(plasmaframe)==3: # Assume it's a vector
         frame = plasmaframe

   # Find the magnetic field direction
   if vlsvReader.check_variable("B"):
      B = vlsvReader.read_variable('B', cellid)
   elif vlsvReader.check_variable("vg_b_vol"):
      B = vlsvReader.read_variable('vg_b_vol', cellid)
   elif vlsvReader.check_variable("fg_b"):
      coordinates = vlsvReader.get_cell_coordinates(cellid)
      B = vlsvReader.read_interpolated_fsgrid_variable('fg_b', coordinates)
   Bmag = np.linalg.norm(B)
   B_unit = B / Bmag

   # verify population
   if pop=="proton":
      if not vlsvReader.check_population(pop):
         if vlsvReader.check_population("avgs"):
            pop="avgs"
            #logging.info("Auto-switched to population avgs")
         else:
            logging.info("Unable to detect population "+pop+" in .vlsv file!")
            sys.exit()
   else:
      if not vlsvReader.check_population(pop):
         logging.info("Unable to detect population "+pop+" in .vlsv file!")
         sys.exit()

   # Read temperature for thermal speed
   if (vcut is True or vcutmax is True):
      if vlsvReader.check_variable("moments"): # restart, use overall rho/rhom and pressure
         rhom = vlsvReader.read_variable("restart_rhom", cellid)
         PDiagonal = vlsvReader.read_variable("pressure", cellid)
         Pressure = (PDiagonal[0] + PDiagonal[1] + PDiagonal[2])*(1./3.)
         vth = np.sqrt(Pressure*8./(rhom*np.pi))
      else:
         if vlsvReader.check_variable(pop+"/vg_rho"): # v5
            vth = vlsvReader.read_variable(pop+"/vg_thermalvelocity", cellid)
         elif vlsvReader.check_variable(pop+"/rho"): # multipop
            vth = vlsvReader.read_variable(pop+"/thermalvelocity", cellid)
         else:
            vth = vlsvReader.read_variable("thermalvelocity", cellid)

   if vcut is True:
      vcutoff=2*vth
   elif vcut is not None: # assume it's a number to use as the speed in m/s
      vcutoff=vcut
   else:
      vcutoff=0

   if vcutmax is True:
      vcutoffmax=2*vth
   elif vcutmax is not None: # assume it's a number to use as the speed in m/s
      vcutoffmax=vcutmax
   else:
      vcutoffmax = np.inf

   # Read the velocity cells:
   velocity_cell_data = vlsvReader.read_velocity_cells(cellid, pop=pop)
   vcellids = list(velocity_cell_data.keys())
   avgs = list(velocity_cell_data.values())

   # Transform to a frame
   v = vlsvReader.get_velocity_cell_coordinates(vcellids, pop=pop) - frame

   # Get the angles:
   v_norms = np.linalg.norm(v,axis=-1)
   v_unit = v/v_norms[:,np.newaxis]
   if cosine == True:
      pitch_angles = (v_unit*B_unit).sum(-1)
      units = "cos alpha"
      pitchrange = (-1,1)
   else:
      pitch_angles = np.arccos((v_unit*B_unit).sum(-1)) * (180./np.pi)
      units = "degree"
      pitchrange = (0,180)

   # Calculate velocity cell sizes
   [vxsize, vysize, vzsize] = vlsvReader.get_velocity_mesh_size(pop=pop)
   [vxmin, vymin, vzmin, vxmax, vymax, vzmax] = vlsvReader.get_velocity_mesh_extent(pop=pop)
   dvx=(vxmax-vxmin)/(4*vxsize)
   dvy=(vymax-vymin)/(4*vysize)
   dvz=(vzmax-vzmin)/(4*vzsize)
   dv3=dvx*dvy*dvz

   # Clip negative avgs to zero
   # (Some elements may be negative due to ghost cell propagation effects)
   avgs = np.array(avgs).clip(min=0) * dv3

   # Mask off cells below threshold
   cond1 = (v_norms > vcutoff)
   cond2 = (v_norms < vcutoffmax)
   condition = np.logical_and(cond1,cond2)
   logging.info(cond1.shape,cond2.shape,condition.shape)
   # Get the velocity cells above cutoff speed
   #vcellids_nonsphere = np.extract(condition, vcellids)
   # Get the avgs
   avgs_nonsphere = np.extract(condition, avgs)
   # Get the pitch-angles (or cosines)
   pitch_nonsphere = np.extract(condition, pitch_angles)

   # Generate a histogram
   weights, angles = np.histogram(pitch_nonsphere, nbins, range=pitchrange, weights=avgs_nonsphere )

   # Wrap the data into a custon format
   result = output_1d([angles, weights], ["Pitch_angle", "sum_avgs"], [units, "1/m3"])

   rho_summed    = np.sum(avgs)
   rho_nonsphere = np.sum(avgs_nonsphere)
   logging.info("rho: " + str(rho_summed) + ", rho_nonsphere: " + str(rho_nonsphere))

   if outputfile!=None or outputdir!=None: # Save output to file
      # Generate filename
      timestr='{:4.1f}'.format(vlsvReader.read_parameter("time"))
      if outputfile==None:
         outputfile = outputdir+"/pitchangle_weights_cellid_"+str(cellid).rjust(7,'0')+"_time_"+timestr+".txt"
      if outputfile!=None and outputdir!=None:
         logging.info("Please use either outputfile or outputdir, not both. Ignoring outputdir.")

      # Check to find actual target sub-directory
      outputprefixind = outputfile.rfind('/')
      if outputprefixind >= 0:
         outputdir = outputfile[:outputprefixind+1]
      # Ensure output directory exists and is writeable
      if not os.path.exists(outputdir):
         try:
            os.makedirs(outputdir)
         except:
            pass
      if not os.access(outputdir, os.W_OK):
         logging.info("No write access for directory "+outputdir+"! Exiting.")
         return


      outfilewrite=open(outputfile,'w')
      outfilewrite.write("# cellid time rho rho_nonsphere Bx By Bz pop\n")
      outfilewrite.write(str(int(cellid))+" "+timestr)
      outfilewrite.write(' {:E} {:E} {:E} {:E} {:E}'.format(rho_summed,rho_nonsphere,B[0],B[1],B[2])+" "+pop+"\n")

      outfilewrite.write("# nbins, bin_edges\n")
      for angle in angles:
         outfilewrite.write('{:E} '.format(angle))
      outfilewrite.write("\n")
      outfilewrite.write("# bin_values\n")
      for weight in weights:
         outfilewrite.write('{:E} '.format(weight))
      outfilewrite.write("\n")

      outfilewrite.close()

   # Return the histogram
   return result
