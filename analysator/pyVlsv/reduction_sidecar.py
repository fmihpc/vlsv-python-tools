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

''' A file for doing data reduction on variables
A file for doing data reduction on variables. This file contains datareducers
that mainly operate on sidecar (containing auxiliary/cached derived quantities)
files. Imported at the end of reduction.py.

Specifically, the GTG-GGT -derived operations here require the full magnetic field jacobian,
which is not part of the default Vlasiator outputs and requires extra steps to acquire with
high precision.
'''

from null_lines import LMN_null_lines_FOTE
from reducer import DataReducerVariable
import numpy as np

def GGT( variables ):
   ''' Data reducer for MDD, or if you want to multiply a tensor with its transpose
   '''
   stack = True
   if(variables[0].shape == (9,)):
      stack = False
      jacobian = np.array([variables[0]]).reshape((variables[0].shape[0],3,3))
   else:   
      jacobian = variables[0].reshape((variables[0].shape[0],3,3))
   jacobian = jacobian.transpose((0,2,1)) # The axes are flipped at some point, correcting for that
   GGT = np.matmul(jacobian, jacobian.transpose((0,2,1)))
   if stack:
      return GGT
   else:
      return GGT[0,:,:]

   print("Error in GGT")
   return -1

def GTG( variables ):
   ''' Data reducer for MGA, or if you want to multiply a tensor's transpose with the tensor
   '''
   stack = True
   if(variables[0].shape == (9,)):
      stack = False
      jacobian = np.array([variables[0]]).reshape((variables[0].shape[0],3,3))
   else:
      jacobian = variables[0].reshape((variables[0].shape[0],3,3))
   jacobian = jacobian.transpose((0,2,1)) # The axes are flipped at some point, correcting for that
   GTG = np.matmul(jacobian.transpose((0,2,1)), jacobian)
   if stack:
      return GTG
   else:
      return GTG[0,:,:]

   print("Error in GGT")
   return -1

def MGA( variables ):
   ''' Data reducer for obtaining the MGA eigensystem
   '''
   stack = True
   GTG = variables[0] # This needs to be a stack of tensors (GTG does this)
   if(GTG.shape == (3,3,)):
      stack = False
      GTG = np.array([GTG]) # I want this to be a stack of tensors

   MGA_eigenvalues, MGA_eigenvectors = np.linalg.eigh(GTG)
   inds = np.argsort(MGA_eigenvalues,axis=-1)
   MGA_eigenvalues = np.take_along_axis(MGA_eigenvalues, inds, axis=-1)[:,::-1]
   ninds = np.repeat(inds[:,:,np.newaxis],3,2).transpose((0,2,1))

   MGA_eigenvectors = np.take_along_axis(MGA_eigenvectors,ninds, axis=-1)[:,:,::-1]
   if stack:
      return MGA_eigenvectors
   else:
      return MGA_eigenvectors[0,:,:]
   print("Error in MGA")
   return -1

def MDD( variables ):
   ''' Data reducer for obtaining the MGA eigensystem
   '''
   stack = True
   GGT = variables[0] # either a tensor, vector, array, or value

   if(GGT.shape == (3,3,)):
      stack = False
      GGT = np.array([GGT]) # I want this to be a stack of tensors

   MDD_eigenvalues, MDD_eigenvectors = np.linalg.eigh(GGT)
   inds = np.argsort(MDD_eigenvalues,axis=-1)

   ninds = np.repeat(inds[:,:,np.newaxis],3,2).transpose((0,2,1))

   MDD_eigenvectors = np.take_along_axis(MDD_eigenvectors,ninds,axis=-1)[:,:,::-1]
   MDD_eigenvalues = np.take_along_axis(MDD_eigenvalues, inds, axis=-1)[:,::-1]

   if stack:
      return MDD_eigenvectors
   else:
      GGT = GGT[0,:,:]
      return MDD_eigenvectors[0,:,:]

   print("Error in MGA")
   return -1

def MDD_dimensionality( variables ):
   ''' Data reducer for obtaining the MGA eigensystem
        Rezeau+2018, doi:10.1002/2017JA024526
        Returns a vector with values (D1,D2,D3), with
        each Di | 0 < Di < 1 and D1+D2+D3 = 1;
        good for e.g. ternary plotting or "Truecolor" plots.
        Uses the square roots of MDD eigenvalues to handle 
        the dimensionality in units of T/m; eigenvalue scaling
        is pretty harsh when in T^2/m^2 and T/m is more
        natural to consider as spatial variation of B.
   '''
   stack = True
   GGT = variables[0] # either a tensor, vector, array, or value

   if(variables[0].shape == (3,3,)):
      stack = False
      GGT = np.array([GGT]) # I want this to be a stack of tensors

   MDD_eigenvalues = np.linalg.eigvalsh(GGT)
   MDD_eigenvalues = MDD_eigenvalues[:,::-1]

   orig_err = np.seterr(all='ignore')
   MDD_eigenvalues = np.sqrt(MDD_eigenvalues) # to (n)T/m

   Ds = np.zeros_like(MDD_eigenvalues)
   Ds[:,0] = (MDD_eigenvalues[:,0] - MDD_eigenvalues[:,1])/MDD_eigenvalues[:,0]
   Ds[:,1] = (MDD_eigenvalues[:,1] - MDD_eigenvalues[:,2])/MDD_eigenvalues[:,0]
   Ds[:,2] = MDD_eigenvalues[:,2]/MDD_eigenvalues[:,0]
   np.seterr(**orig_err)

   if stack:
      return Ds
   else:
      GGT = GGT[0,:,:]
      return Ds[0,:]

   print("Error in MGA")
   return -1

def LMN( variables ):
   ''' Data reducer to calculate the LMN basis vectors
        using the full B jacobian
        Returns a stack of LMN unit vectors
   '''

   stack = True
   MGA_vecs = variables[0].copy()
   MDD_vecs = variables[1].copy()
   Js = variables[2].copy()
   if(Js.shape == (3,)):
      stack = False
      MGA_vecs = np.array([MGA_vecs])
      MDD_vecs = np.array([MDD_vecs])
      Js = np.array([Js]) # I want this to be a stack of vectors

   zeroJs = np.linalg.norm(Js, axis=-1) == 0
   Js[zeroJs,:] = MGA_vecs[zeroJs,:,2] # MGA N for dummy data

   Ls = MGA_vecs[:,:,0]
   # Ls = MDD_vecs[:,:,1]
   Ns = MDD_vecs[:,:,0]
   Ns[zeroJs, :] = MGA_vecs[zeroJs,:,1]
   #Ns = MGA_vecs[:,:,2]

   projs = np.sum(Ns*Ls,axis=-1)[:,np.newaxis]
   Ns = Ns - Ls*projs
   norms = np.linalg.norm(Ns,axis=-1)
   np.divide(Ns, norms[:,np.newaxis], out = Ns, where = (norms != 0)[:,np.newaxis])
   NxL = np.cross(Ns,Ls,axis=-1)

   mask = np.array([np.sum(Js*NxL,axis=-1) < 0])
   mrep =  np.repeat(mask,(3,),axis=0).transpose()
   np.multiply(Ns, -1, out=Ns, where=mrep)
   
   Ns[norms==0,:] = np.nan
   
   Ms = np.cross(Ns,Ls,axis=-1) # Y = Z x X
   
   norms = np.linalg.norm(Ms,axis=-1)
   np.divide(Ms, norms[:,np.newaxis], out=Ms, where = (norms!=0)[:,np.newaxis])
   Ms[norms==0,:] = np.nan

   if stack:
      return np.stack((Ls,Ms,Ns),axis=-1)
   else:
      return np.stack((Ls,Ms,Ns),axis=-1)[0,:,:]

   print("Error in LMN")
   return -1

def LMN_xoline_distance( variables ):
   ''' Datareducer that uses a linear approximation of B from B and its Jacobian
   in the LMN coordinates to get an approximate distance to the neutral line.
   inputs:
   * LMN basis vectors stack
   * Jacobian of B in 9-element vector stack
   * vg_b_vol

   Return the minimum signed distance (SD) to the cell for a neutral line in the FOTE
   approximation. For the point d with minimal SD to the cell on the neutral line:
      case a: d within the cube: d < 0; at the origin, for a unit cube centered at the origin d = -0.5
      case b: d outside the cube: d > 0, d being the minimal distance to the surface of the cube
   '''

   LMNs = variables[0]
   jacobs = variables[1]
   Bs = variables[2]
   dxs = variables[3]
   coords = variables[4]
   stack = True
   if(Bs.shape == (3,)):  # I want these to be stacks
      stack = False

   # deferring the actual calculations under pyCalculations - enables more outputs
   # from the FOTE function.
   s_line, _ = LMN_null_lines_FOTE(LMNs, jacobs, Bs, dxs, coords)
   if stack:
      return s_line
   else:
      return s_line[0]

# not finished
def nullpoint_distance( variables ):
   ''' Datareducer that uses a linear approximation of B from B and its Jacobian
   to calculate the distance to a magnetic null point (B == [0,0,0]).
   inputs:
   * LMN basis vectors stack
   * Jacobian of B in 9-element vector stack
   * vg_b_vol

   Return a signed distance function to the cell for a null point. For 
   the point d closest to the cell center on the line:
      case a: d within the cube: d < 0; at the origin, for a unit cube centered at the origin d = -0.5
      case b: d outside the cube: d > 0, d being the minimal distance to the surface of the cube

   '''
   raise NotImplementedError("This function is not finished, but there is demand for it. Someone might get around to implementing this sooner rather than later.")
   Bs = variables[0]
   jacobs = variables[1]
   dxs = variables[2]
   coords = variables[3]
   stack = True
   if(Bs.shape == (3,)):  # I want these to be stacks
      stack = False
      jacobs = np.array([jacobs])
      dxs = np.array(dxs)
      Bs = np.array([Bs])

   jacobs = np.reshape(jacobs,(jacobs.shape[0],3,3))
   n_cells = Bs.shape[0]
   #rot = [basis_L[1:3,i] basis_M[1:3,i] basis_N[1:3,i]]
   #so basically just LMN
   # rotate the jacobians
   # jacobs = np.transpose(LMNs,(0, 2,1)) @ jacobs @ LMNs

   Bx = Bs[:,0] #np.sum(Ls*Bs, axis=-1)
   By = Bs[:,1] #np.sum(Ms*Bs, axis=-1)
   Bz = Bs[:,2] #np.sum(Ns*Bs, axis=-1)
   gradBx = jacobs[:, 0,:]
   gradBy = jacobs[:, 1,:]
   gradBz = jacobs[:, 2,:]

   gradBxn = np.linalg.norm(gradBx,axis=-1)
   gradByn = np.linalg.norm(gradBz,axis=-1)
   gradBzn = np.linalg.norm(gradBz,axis=-1)

   # Distance to zero plane for BL an BN; nb. jacobian in nT/m..
   sx = Bx/(gradBxn)
   sy = By/(gradByn)
   sz = Bz/(gradBzn)
   x_zero_intercept = gradBx/np.broadcast_to(gradBxn,(3,n_cells)).transpose()
   x_zero_intercept = x_zero_intercept*np.broadcast_to(sx,(3,n_cells)).transpose()/dxs
   y_zero_intercept = gradBz/np.broadcast_to(gradByn,(3,n_cells)).transpose()
   y_zero_intercept = y_zero_intercept*np.broadcast_to(sy,(3,n_cells)).transpose()/dxs
   z_zero_intercept = gradBz/np.broadcast_to(gradBzn,(3,n_cells)).transpose()
   z_zero_intercept = z_zero_intercept*np.broadcast_to(sz,(3,n_cells)).transpose()/dxs

   n_line = np.cross(x_zero_intercept,z_zero_intercept, axis=-1)
   #Find a line intercept point and its norm
   n_line_intercept = (x_zero_intercept + z_zero_intercept)


   #rotate the n_line_intercept to xyz instead of LMN
   nn = n_line_intercept[:,np.newaxis,:]
   # print("nn", nn)
   #### commented out mid-edit for missing LMNs
   #### n_line_intercept = np.matmul(nn, LMNs).squeeze() # inverse of rotation for row vectors
   #### n_line = np.matmul(n_line[:,np.newaxis,:], LMNs).squeeze()
   # print("nl",n_line_intercept)
   s_line = np.linalg.norm(n_line_intercept,axis=-1)
   # Bl = np.repeat(np.array([BL]),3,axis=0).transpose()*Ls
   # Bm = np.repeat(np.array([BN]),3,axis=0).transpose()*Ns
   # Bn = np.repeat(np.array([BM]),3,axis=0).transpose()*Ms
   # print("Bl",Bl[:,np.newaxis,:])
   # #print(np.stack((Bl,Bm,Bn),axis=-1))
   # print("rotd",np.matmul(Bl[:,np.newaxis,:], LMNs))
   # print("rotd",np.matmul(Bn[:,np.newaxis,:], LMNs))
   # print("rotd",np.matmul(Bm[:,np.newaxis,:], LMNs))

# This is horrible, I sort of wish we had spherical shells
   # mirror across the origin on all axes if on negative hemisphere -
   # easier to check for intersection only with xyz = 0.5 planes
   flip_coords = np.ones_like(n_line_intercept)
   lineintercept = n_line_intercept
   linevec = n_line/np.broadcast_to(np.linalg.norm(n_line,axis=-1),(3,n_cells)).transpose()
   # print("linevec",linevec)
   # print("n_line", n_line)
   np.multiply(n_line_intercept, -1, out=lineintercept, where= n_line_intercept < 0)
   np.multiply(n_line, -1, out=linevec, where= n_line_intercept < 0)
   
   # find distances to the 
   d_planes = np.divide(0.5-lineintercept,linevec)
   d_planesi = np.divide(0.5-lineintercept,-linevec)
   # print("dp",d_planes)
   # print(d_planesi)

   hits = np.all(np.abs(n_line_intercept) < 0.5, axis=-1) # these are certain
   eps = 1e-12

   
   for d in [0,1,2]:
      ds = (np.repeat(np.array([d_planes[:,d]]).transpose(),3,axis=1))
      
      new_intercepts = lineintercept
      ds = linevec*ds
      
      new_intercepts +=  ds
      hits = hits | np.all(np.abs(new_intercepts) <= 0.5+eps, axis=-1)
      ds = (np.repeat(np.array([d_planesi[:,d]]).transpose(),3,axis=1))
      ds = linevec*ds
      new_intercepts = lineintercept + ds
      hits = hits | np.all(np.abs(new_intercepts) <= 0.5+eps, axis=-1)
   
   np.divide(-1, s_line,out=s_line, where=hits)
   # np.add(s_line, 1,out=s_line, where=hits) # so that barely hitting approaches 0-
   # np.subtract(s_line, 0.5, out=s_line, where=np.logical_not(hits)) # so that barely missing approaches 0+
   if stack:
      return s_line
   else:
      return s_line[0]

def L_flip_distance( variables ):
   ''' Datareducer that uses a linear approximation of B from B and its Jacobian
   in the LMN coordinates to get an approximate distance to the neutral plane.
   inputs:
   * LMN basis vectors stack
   * Jacobian of B in 9-element vector stack
   * vg_b_vol

   Returns distance from cell center to the plane where BL = 0, units of dx.
   '''

   LMNs = variables[0]
   jacobs = variables[1]
   Bs = variables[2]
   dxs = variables[3]
   stack = True
   if(Bs.shape == (3,)):  # I want these to be stacks
      stack = False
      LMNs = np.array([LMNs])
      jacobs = np.array([jacobs])
      dxs = np.array(dxs)
      Bs = np.array([Bs])

   jacobs = np.reshape(jacobs,(jacobs.shape[0],3,3))
   Ls = LMNs[:,:,0]
   #Ns = LMNs[:,:,2]
   n_cells = Bs.shape[0]
   #rot = [basis_L[1:3,i] basis_M[1:3,i] basis_N[1:3,i]]
   #so basically just LMN
   # rotate the jacobians
   jacobs = np.transpose(LMNs,(0, 2,1)) @ jacobs @ LMNs

   BL = np.sum(Ls*Bs, axis=-1)
   gradBL = -1*jacobs[:, 0,:]

   gradBLn = np.linalg.norm(gradBL,axis=-1)
   #gradBNn = np.linalg.norm(gradBN,axis=-1)

   # Distance to zero plane for BL an BN
   sL = BL/(gradBLn)

   L_zero_intercept = gradBL/np.broadcast_to(gradBLn,(3,n_cells)).transpose()
   L_zero_intercept = L_zero_intercept*np.broadcast_to(sL,(3,n_cells)).transpose()/dxs

   return L_zero_intercept

def N_flip_distance( variables ):
   ''' Datareducer that uses a linear approximation of B from B and its Jacobian
   in the LMN coordinates to get an approximate distance to the neutral plane.
   inputs:
   * LMN basis vectors stack
   * Jacobian of B in 9-element vector stack
   * vg_b_vol

   Returns distance from cell center to the plane where BL = 0, units of dx.
   '''

   LMNs = variables[0]
   jacobs = variables[1]
   Bs = variables[2]
   dxs = variables[3]
   stack = True
   if(Bs.shape == (3,)):  # I want these to be stacks
      stack = False
      LMNs = np.array([LMNs])
      jacobs = np.array([jacobs])
      dxs = np.array(dxs)
      Bs = np.array([Bs])

   jacobs = np.reshape(jacobs,(jacobs.shape[0],3,3))
   #Ls = LMNs[:,:,0]
   Ns = LMNs[:,:,2]
   n_cells = Bs.shape[0]
   #rot = [basis_L[1:3,i] basis_M[1:3,i] basis_N[1:3,i]]
   #so basically just LMN
   # rotate the jacobians
   jacobs = np.transpose(LMNs,(0, 2,1)) @ jacobs @ LMNs

   BN = np.sum(Ns*Bs, axis=-1)
   gradBN = -1*jacobs[:, 2,:]

   #gradBLn = np.linalg.norm(gradBL,axis=-1)
   gradBNn = np.linalg.norm(gradBN,axis=-1)

   # Distance to zero plane for BN
   sN = BN/(gradBNn)
   N_zero_intercept = gradBN/np.broadcast_to(gradBNn,(3,n_cells)).transpose()
   N_zero_intercept = N_zero_intercept*np.broadcast_to(sN,(3,n_cells)).transpose()/dxs

   return N_zero_intercept


from reduction import v5reducers

v5reducers["vg_gtg"] =                    DataReducerVariable(["vg_jacobian_b"], GTG, "T^2/m^2", 9, latex=r"$\mathcal{G}^\intercal\mathcal{G}$",latexunits=r"$\mathrm{T}^2\,\mathrm{m}^{-2}$")
v5reducers["vg_ggt"] =                    DataReducerVariable(["vg_jacobian_b"], GGT, "T^2/m^2", 9, latex=r"$\mathcal{G}\mathcal{G}^\intercal$",latexunits=r"$\mathrm{T^2}\,\mathrm{m}^{-2}$")
v5reducers["vg_mga"] =                    DataReducerVariable(["vg_gtg"], MGA, "-", 9, latex=r"$\mathrm{MGA}$",latexunits=r"")
v5reducers["vg_mdd"] =                    DataReducerVariable(["vg_ggt"], MDD, "-", 9, latex=r"$\mathrm{MDD}$",latexunits=r"")
v5reducers["vg_mdd_dimensionality"] =     DataReducerVariable(["vg_ggt"], MDD_dimensionality, "-", 3, latex=r"$\\mathrm{D}_{i}$",latexunits=r"")
v5reducers["vg_lmn"] =                    DataReducerVariable(["vg_mga","vg_mdd", "vg_j"], LMN, "-", 9, latex=r"$\mathrm{LMN}$",latexunits=r"")
v5reducers["vg_lmn_neutral_line_distance"] =  DataReducerVariable(["vg_lmn","vg_jacobian_b", "vg_b_vol", "vg_dx", "vg_coordinates"], LMN_xoline_distance, "dx", 1, latex=r"$\mathrm{LMN}_\mathrm{SDF,FOTE}$",latexunits=r"$\Delta x$")
v5reducers["vg_lmn_l_flip_distance"] =  DataReducerVariable(["vg_lmn","vg_jacobian_b", "vg_b_vol", "vg_dx"], L_flip_distance, "dx", 1, latex=r"$\mathcal{G}^\intercal\mathcal{G}$",latexunits=r"$\Delta x$")
v5reducers["vg_lmn_m_flip_distance"] =  DataReducerVariable(["vg_lmn","vg_jacobian_b", "vg_b_vol", "vg_dx"], N_flip_distance, "dx", 1, latex=r"$\mathcal{G}^\intercal\mathcal{G}$",latexunits=r"$\Delta x$")
