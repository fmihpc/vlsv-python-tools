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

''' A file for finding null lines and miscellaneous data
'''
import numpy as np
import sys
import logging

# Box: correct distance to corners hg_sdf, see https://mercury.love/hg_sdf/
def fBox(query_points, box_corner=np.array([0.5,0.5,0.5])):
   ''' Function that finds the signed distance function from query points to
   a Cartesian axis-aligned box. The box centroids are assumed to be at the origin,
   and the default side length is 1. Default broadcasts the box_corner array
   to match the number of query points.

   :param query_points:  n-by-3 ndarray (or single 3-vector) of coordinates (x,y,z)
   :param box_corner:    Specify the top corner(s) of the box(es) - symmetric around origin
   :returns: a numpy array of signed distance functions.
   '''
   p = np.atleast_2d(query_points)
   b = np.atleast_2d(box_corner)
   b = np.broadcast_to(b, p.shape)
   d = np.abs(p) - b
   vec30 = np.zeros_like(d)
   return np.linalg.norm(np.fmax(d, vec30),axis=-1) + np.amax(np.fmin(d, vec30),axis=-1)


def LMN_null_lines_FOTE(LMNs, jacobs, Bs, dxs, coords):
   ''' Function that uses a linear approximation of B from B and its Jacobian
   in the LMN coordinates to get an approximate distance to the neutral line.
   inputs:
   * LMN basis vectors stack
   * Jacobian of B in 9-element vector stack
   * vg_b_vol

   return a measure for closeness to the cell center for a neutral line and other data.
   The return value is the minimum of the signed distance function between the neutral
   line 

   '''
   stack = True
   LMNs = LMNs.copy()
   jacobs = jacobs.copy()
   dxs = dxs.copy()
   Bs = Bs.copy()
   if(Bs.shape == (3,)):  # I want these to be stacks
      stack = False
      LMNs = np.array([LMNs])
      jacobs = np.array([jacobs])
      dxs = np.array(dxs)
      Bs = np.array([Bs])

   jacobs = np.reshape(jacobs,(jacobs.shape[0],3,3))
   Ls = LMNs[:,:,0]
   Ms = LMNs[:,:,1]
   Ns = LMNs[:,:,2]
   n_cells = Bs.shape[0]
   #rot = [basis_L[1:3,i] basis_M[1:3,i] basis_N[1:3,i]]
   #so basically just LMN
   # rotate the jacobians
   jacobs = np.transpose(LMNs,(0, 2,1)) @ jacobs @ LMNs

   BL = np.sum(Ls*Bs, axis=-1)
   BM = np.sum(Ms*Bs, axis=-1)
   BN = np.sum(Ns*Bs, axis=-1)
   unit = 1
   gradBL = jacobs[:, 0,:]*unit
   #gradBM = jacobs[1,:]
   gradBN = jacobs[:, 2,:]*unit

   gradBLn = np.linalg.norm(gradBL,axis=-1)
   gradBNn = np.linalg.norm(gradBN,axis=-1)
   mask = np.isfinite(gradBLn) & np.isfinite(gradBNn) # require sane gradients
   mask = mask & (gradBLn >0) & (gradBNn >0)          # require nonzero gradients
   mask = mask & np.all(np.isfinite(Ls), axis=-1)     # these should be covered by above..
   mask = mask & np.all(np.isfinite(Ms), axis=-1)
   mask = mask & np.all(np.isfinite(Ns), axis=-1)


   gradBL = gradBL/np.broadcast_to(gradBLn,(3,n_cells)).transpose()
   gradBN = gradBN/np.broadcast_to(gradBNn,(3,n_cells)).transpose()
   dots = np.sum(gradBL*gradBN, axis=-1)
   mask = mask & (dots != 1)
   # Distance to zero plane for BL an BN
   sL = -BL/(gradBLn)
   sN = -BN/(gradBNn)

   n_line = np.full(gradBL.shape,np.nan)

   n_line[mask,:] = np.cross(gradBL,gradBN, axis=-1)[mask,:]
   n_line = n_line/np.broadcast_to(np.linalg.norm(n_line,axis=-1),(3,n_cells)).transpose()

   #Find a line intercept point and its norm
   n_line_intercept = np.ones_like(gradBL)#(L_zero_intercept + N_zero_intercept) # well this is just wrong when not perp
   n_line_intercept.fill(np.inf)

   c1 = sL - sN*(dots)/(1 - dots**2)
   c2 = sN - sL*(dots)/(1 - dots**2)
   n_line_intercept[mask,:] = (((gradBL*np.broadcast_to(c1,(3,n_cells)).transpose() + gradBN*np.broadcast_to(c2,(3,n_cells)).transpose())/dxs))[mask,:]

   #rotate the n_line_intercept to xyz instead of LMN
   nn = n_line_intercept[:,np.newaxis,:]

   n_line_intercept[mask,:] = np.matmul(nn, LMNs).squeeze()[mask,:] # inverse of rotation for row vectors
   n_line[mask,:] = np.matmul(n_line[:,np.newaxis,:], LMNs).squeeze()[mask,:]

   # Make certain that we have a point along the line that is closest to the cell center
   par_dist = np.sum(n_line_intercept*n_line, axis=-1)[:,np.newaxis]
   n_line_intercept = n_line_intercept - n_line*np.broadcast_to(np.sum(n_line_intercept*n_line, axis=-1),(3,n_cells)).transpose() # is it not nearest to 0 already?
   #s_line = np.linalg.norm(n_line_intercept,axis=-1)
   s_line = np.full(gradBLn.shape,np.inf)
   s_line[mask] = fBox(n_line_intercept, np.array([0.5,0.5,0.5]))[mask] # Proper SDF, but this is just the initial point

   #Let's go back a dx or two so we need only to look at forward intercepts
   init_interval = 4
   n_line_intercept = n_line_intercept - n_line*init_interval/2

   lineintercept = n_line_intercept
   linevec = n_line/np.broadcast_to(np.linalg.norm(n_line,axis=-1),(3,n_cells)).transpose()
   
   # Golden section search.
   golden_ratio_factor = 0.38197

   # a,b,c,x are displacement vectors along the line normal, with
   # 
   # |----------------------| < bracket interval
   # a       b     x        c    
   a = np.zeros_like(s_line)
   b = np.ones_like(s_line)*init_interval*golden_ratio_factor
   c = np.ones_like(s_line)*init_interval
   x = np.ones_like(s_line)*b+(c-b)*golden_ratio_factor

   # Are we bracketing the interval a-b or b-c?
   bracket_dir = np.ones_like(s_line)

   # fb, fx are the SDF values to be minimized, evaluated at points
   # given by displacements b and x:
   fb = fBox(lineintercept + b[:,np.newaxis]*linevec)
   fx = fBox(lineintercept + x[:,np.newaxis]*linevec)
   # (these could be initialized to zero here)
   
   # Stepping mask - if true, this triplet is still to converge.
   in_stepping = mask.copy()
   # Check that the bracket is well-defined
   in_stepping = (
                  in_stepping & 
                  (fBox(lineintercept + a[:,np.newaxis]*linevec) > fBox(lineintercept + b[:,np.newaxis]*linevec)) & 
                  (fBox(lineintercept + c[:,np.newaxis]*linevec) > fBox(lineintercept + b[:,np.newaxis]*linevec))
                  )

   tolerance = 0.01 # interval to reach in dx
   niter = 0
   maxiters = 20

   
   while(np.any(in_stepping[mask])): 

      # Evaluate the function at b and x
      fb[in_stepping] = fBox(lineintercept[in_stepping,:] + b[in_stepping,np.newaxis]*linevec[in_stepping,:])
      fx[in_stepping] = fBox(lineintercept[in_stepping,:] + x[in_stepping,np.newaxis]*linevec[in_stepping,:])

      # Depending on the result of fb < fx, update bracketing triplet 

      # Bracket_dir 1: interval (b,c) larger than (a,b) (initial case)
      # fb < fx: bracket updates from
      # a--b-x--c
      # to
      # a--b-c---
      #   ^ - new x goes there (bracket_dir will flip)
      # a[(bracket_dir == 1) & (fb < fx) & in_stepping] = a[(bracket_dir == 1) & (fb < fx) & in_stepping]
      # b[(bracket_dir == 1) & (fb < fx) & in_stepping] = b[(bracket_dir == 1) & (fb < fx) & in_stepping]
      c[(bracket_dir == 1) & (fb < fx) & in_stepping] = x[(bracket_dir == 1) & (fb < fx) & in_stepping]

      # fb >= fx: bracket updates from
      # a--b-x--c
      # to
      #    a-b--c
      #       ^ - new x goes there
      a[(bracket_dir == 1) & (fb >= fx) & in_stepping] = b[(bracket_dir == 1) & (fb >= fx) & in_stepping]
      b[(bracket_dir == 1) & (fb >= fx) & in_stepping] = x[(bracket_dir == 1) & (fb >= fx) & in_stepping]
      # c[(bracket_dir == 1) & (fb >= fx) & in_stepping] = c[(bracket_dir == 1) & (fb >= fx) & in_stepping]

      # Bracket_dir -1: interval (a,b) larger than (b,c)
      # fb < fx: bracket updates from
      # a--x-b--c
      # to
      # ---a-b--c
      #       ^ - new x goes there (bracket_dir will flip)
      a[(bracket_dir == -1) & (fb < fx) & in_stepping] = x[(bracket_dir == -1) & (fb < fx) & in_stepping]
      # b[(bracket_dir == -1) & (fb < fx) & in_stepping] = b[(bracket_dir == -1) & (fb < fx) & in_stepping]
      # c[(bracket_dir == -1) & (fb < fx) & in_stepping] = c[(bracket_dir == -1) & (fb < fx) & in_stepping]

      # fb >= fx: bracket updates from
      # a--x-b--c
      # to
      # a--b-c--
      #   ^ - new x goes there
      # a[(bracket_dir == -1) & (fb >= fx) & in_stepping] = a[(bracket_dir == -1) & (fb >= fx) & in_stepping]
      b[(bracket_dir == -1) & (fb >= fx) & in_stepping] = x[(bracket_dir == -1) & (fb >= fx) & in_stepping]
      c[(bracket_dir == -1) & (fb >= fx) & in_stepping] = b[(bracket_dir == -1) & (fb >= fx) & in_stepping]

      bracket_dir[(b - a) < (c - b)] = 1
      bracket_dir[(b - a) >= (c - b)] = -1
      x[in_stepping & (bracket_dir == -1)] = b[in_stepping & (bracket_dir == -1)] - ((b-a)*golden_ratio_factor)[in_stepping & (bracket_dir == -1)]
      x[in_stepping & (bracket_dir ==  1)] = b[in_stepping & (bracket_dir ==  1)] + ((c-b)*golden_ratio_factor)[in_stepping & (bracket_dir ==  1)]

      in_stepping = in_stepping & ((c - a) > tolerance)
      niter = niter+1
      # print(niter,"iterations, remaining: ", np.sum(in_stepping), 
      #       "right ", np.sum(bracket_dir[in_stepping] ==1), "left ", np.sum(bracket_dir[in_stepping]==-1))
      if(niter >= maxiters):
         logging.warning("Golden section search in LMN_null_lines_FOTE in "+ __file__ +" reached max iterations - some cell failed to converge?")
         break

   # Construct a line segment of local dx length where the neutral line is approximated to be
   s_line[mask] = np.fmin(fb,fx)[mask]
   a = coords+(n_line_intercept)*dxs
   b = n_line
   par_dist = np.sum((n_line_intercept*dxs)*n_line, axis=-1)[:,np.newaxis]
   stck = np.hstack((a,b,s_line[:,np.newaxis],coords,-(n_line_intercept)*dxs,
                  par_dist))
   if stack:
      return s_line, stck
   else:
      return s_line[0], stck[0]
