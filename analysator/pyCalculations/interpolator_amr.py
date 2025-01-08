from scipy.spatial import Delaunay
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from operator import itemgetter
# from time import time
import warnings
import logging

importerror = None

try:
   from scipy.interpolate import RBFInterpolator
except Exception as e:
   importerror = e
   class RBFInterpolator(object):
      def __init__(self, pts, vals, **kwargs):
         raise importerror #Exception("Module load error")
   


# With values fi at hexahedral vertices and trilinear basis coordinates ksi,
# return the trilinear interpolant for fi
# Vectorized. Assumes that the first dimension ksii, fii are the index for an individual query - 
# singleton dimension required for a single query!
def f(ksii, fii):
   ksi = np.atleast_2d(ksii)
   if(fii.ndim == 1):
      fi = fii[np.newaxis,:,np.newaxis]
   elif(fii.ndim == 2):
      fi = fii
      res = (1-ksi[:,0]) * (1-ksi[:,1]) * (1-ksi[:,2]) * fi[:,0] + \
               ksi[:,0]  * (1-ksi[:,1]) * (1-ksi[:,2]) * fi[:,1] + \
            (1-ksi[:,0]) *    ksi[:,1]  * (1-ksi[:,2]) * fi[:,2] + \
               ksi[:,0]  *    ksi[:,1]  * (1-ksi[:,2]) * fi[:,3] + \
            (1-ksi[:,0]) * (1-ksi[:,1]) *    ksi[:,2]  * fi[:,4] + \
               ksi[:,0]  * (1-ksi[:,1]) *    ksi[:,2]  * fi[:,5] + \
            (1-ksi[:,0]) *    ksi[:,1]  *    ksi[:,2]  * fi[:,6] + \
               ksi[:,0]  *    ksi[:,1]  *    ksi[:,2]  * fi[:,7]
      return res
   else:
      fi = fii

   # this broadcasts to ksi@ksi shape for scalar fis for whatever reason, so above
   res = (1-ksi[:,0,None]) * (1-ksi[:,1,None]) * (1-ksi[:,2,None]) * fi[:,0,:] + \
            ksi[:,0,None]  * (1-ksi[:,1,None]) * (1-ksi[:,2,None]) * fi[:,1,:] + \
         (1-ksi[:,0,None]) *    ksi[:,1,None]  * (1-ksi[:,2,None]) * fi[:,2,:] + \
            ksi[:,0,None]  *    ksi[:,1,None]  * (1-ksi[:,2,None]) * fi[:,3,:] + \
         (1-ksi[:,0,None]) * (1-ksi[:,1,None]) *    ksi[:,2,None]  * fi[:,4,:] + \
            ksi[:,0,None]  * (1-ksi[:,1,None]) *    ksi[:,2,None]  * fi[:,5,:] + \
         (1-ksi[:,0,None]) *    ksi[:,1,None]  *    ksi[:,2,None]  * fi[:,6,:] + \
            ksi[:,0,None]  *    ksi[:,1,None]  *    ksi[:,2,None]  * fi[:,7,:]
   if(res.shape[-1] == 1):
      return np.squeeze(res, axis = -1)
   else:
      return res


# With values fi at hexahedral vertices and trilinear basis coordinates ksi,
# return the interpolated gradient/jacobian of fi wrt. the trilinear basis at ksi
# Vectorized. Assumes that the first dimension ksii, fii are the index for an individual query - 
# singleton dimension required for a single query!
# does not handle scalars?
def df(ksii, fii):
   ksi = np.atleast_2d(ksii)
   # logging.info(fii.shape)
   if(fii.ndim == 1):
      fi = fii[np.newaxis,:,np.newaxis]
   elif(fii.ndim == 2):
      fi = np.atleast_3d(fii)#fii[:,:,np.newaxis]
   else:
      fi = fii
   # logging.info(fi)
   # logging.info(fi.shape)
   d0 =  -1 * (1-ksi[:,1,None]) * (1-ksi[:,2,None]) * fi[:,0,:] + \
          1 * (1-ksi[:,1,None]) * (1-ksi[:,2,None]) * fi[:,1,:] + \
         -1 *    ksi[:,1,None]  * (1-ksi[:,2,None]) * fi[:,2,:] + \
          1 *    ksi[:,1,None]  * (1-ksi[:,2,None]) * fi[:,3,:] + \
         -1 * (1-ksi[:,1,None]) *    ksi[:,2,None]  * fi[:,4,:] + \
          1 * (1-ksi[:,1,None]) *    ksi[:,2,None]  * fi[:,5,:] + \
         -1 *    ksi[:,1,None]  *    ksi[:,2,None]  * fi[:,6,:] + \
          1 *    ksi[:,1,None]  *    ksi[:,2,None]  * fi[:,7,:]
   d1 =  (1-ksi[:,0,None]) * -1  * (1-ksi[:,2,None]) * fi[:,0,:] + \
            ksi[:,0,None]  * -1  * (1-ksi[:,2,None]) * fi[:,1,:] + \
         (1-ksi[:,0,None]) *  1  * (1-ksi[:,2,None]) * fi[:,2,:] + \
            ksi[:,0,None]  *  1  * (1-ksi[:,2,None]) * fi[:,3,:] + \
         (1-ksi[:,0,None]) * -1  *    ksi[:,2,None]  * fi[:,4,:] + \
            ksi[:,0,None]  * -1  *    ksi[:,2,None]  * fi[:,5,:] + \
         (1-ksi[:,0,None]) *  1  *    ksi[:,2,None]  * fi[:,6,:] + \
            ksi[:,0,None]  *  1  *    ksi[:,2,None]  * fi[:,7,:]
   d2 =  (1-ksi[:,0,None]) * (1-ksi[:,1,None]) * -1 * fi[:,0,:] + \
            ksi[:,0,None]  * (1-ksi[:,1,None]) * -1 * fi[:,1,:] + \
         (1-ksi[:,0,None]) *    ksi[:,1,None]  * -1 * fi[:,2,:] + \
            ksi[:,0,None]  *    ksi[:,1,None]  * -1 * fi[:,3,:] + \
         (1-ksi[:,0,None]) * (1-ksi[:,1,None]) *  1 * fi[:,4,:] + \
            ksi[:,0,None]  * (1-ksi[:,1,None]) *  1 * fi[:,5,:] + \
         (1-ksi[:,0,None]) *    ksi[:,1,None]  *  1 * fi[:,6,:] + \
            ksi[:,0,None]  *    ksi[:,1,None]  *  1 * fi[:,7,:]
   res = np.stack((d0,d1,d2),axis = -1)
   return res

# For hexahedral vertices verts and point p, find the trilinear basis coordinates ksi
# that interpolate the coordinates of verts to the tolerance tol.
# This is an iterative procedure. Return nans in case of no convergence.
def find_ksi(p, v_coords, tol= .1, maxiters = 200):
   p = np.atleast_2d(p)
   v_coords = np.atleast_3d(v_coords)
   ksi0 = np.full_like(p, 0.5)
   J = df(ksi0, v_coords)
   # logging.info("J", J)
   ksi_n = ksi0
   ksi_n1 = np.full_like(ksi0, np.nan)
   f_n =  f(ksi_n,v_coords)
   # logging.info(f_n.shape, p.shape)
   # logging.info(f_n)
   f_n = f_n - p

   resolved = np.full((p.shape[0],),False, dtype=bool)
   diverged = np.full((p.shape[0],),False, dtype=bool)
   
   for i in range(maxiters):
      
      J[~resolved,:,:] = df(ksi_n[~resolved,:], v_coords[~resolved,:,:])
      f_n[~resolved,:] = f(ksi_n[~resolved,:],v_coords[~resolved,:,:])-p[~resolved,:]
      step = np.linalg.solve(J[~resolved,:,:], -f_n[~resolved,:])
      ksi_n1[~resolved,:] = step + ksi_n[~resolved,:] # r_(n+1) 
      ksi_n[~resolved,:] = ksi_n1[~resolved,:]
      
      resolved[~resolved] = (np.linalg.norm(f(ksi_n1[~resolved,:],v_coords[~resolved,:,:]) - p[~resolved,:],axis=1) < tol)
      resolved[~resolved] = np.linalg.norm(ksi_n1[~resolved,:],axis=1) > 1e2 # Don't bother if the solution is diverging either, set to nans later
         # convergence = True
      if np.all(resolved):
         break
         # diverged = np.linalg.norm(ksi_n1,axis=1) > 1e2
         # ksi_n1[diverged,:] = np.nan
         # # logging.info("All converged in ", i, "iterations")
         # return ksi_n1
      

   if np.all(resolved):
      # logging.info("Converged after ", i, "iterations")
      diverged = np.linalg.norm(ksi_n1,axis=1) > 1e2
      ksi_n1[diverged,:] = np.nan
      return ksi_n1
   else:
      # warnings.warn("Generalized trilinear interpolation did not converge for " + str(np.sum(~convergence)) + " points. Nans inbound.")
      diverged = np.linalg.norm(ksi_n1,axis=1) > 1e2
      ksi_n1[diverged,:] = np.nan
      ksi_n1[~resolved,:] = np.nan
      return ksi_n1
      
class HexahedralTrilinearInterpolator(object):
   ''' Class for doing general hexahedral interpolation, including degenerate hexahedra (...eventually).
   '''

   def __init__(self, pts, vals, **kwargs):
      self.pts = pts
      self.vals = vals
      # Call dual construction from here?
      self.reader = kwargs['reader']
      self.var = kwargs['var']
      self.operator = kwargs['op']

   def __call__(self, pt, cellids = None):
      pts = np.atleast_2d(pt)
      if(len(pts.shape) == 2):
         # t0 = time()
         vals = []
         duals = []
         ksis = []
         # for i,p in enumerate(pt):
         #    d, ksi = self.reader.get_dual(p)
         #    duals.append(d)
         #    ksis.append(ksi)
         duals, ksis = self.reader.get_dual(pts, cellids)
         duals_corners = np.array(itemgetter(*duals)(self.reader._VlsvReader__dual_cells))
         fi = self.reader.read_variable(self.var, duals_corners.reshape(-1), operator=self.operator)
         if(fi.ndim == 2):
            val_len = fi.shape[1]
         else:
            val_len = 1
         ksis = np.array(ksis).squeeze() # n x 1 x 3 ...... fix
         # logging.info(ksis.shape, fi.shape)
         if(val_len == 1):
            fi = fi.reshape((-1,8))
         else:
            fi = fi.reshape((-1,8,val_len))
         # logging.info('fi reshaped', fi.shape)
         vals = f(ksis, fi)
         # logging.info("irregular interpolator __call__ done in", time()-t0,"s")
         return vals
      
         # the following loop is not reached, kept for reference
         for i,p in enumerate(pts):
            # dual, ksi = self.reader.get_dual(np.atleast_2d(p))
            # dual = dual[0]
            # ksi = ksi[0]
            # logging.info("regular:",i,dual, ksi)
            dual = duals[i]
            ksi = ksis[i]
            # logging.info("from batch:", dual, ksi)
            if dual is None:
               vals.append(np.nan)
            else:
               # dual_corners = self.reader._VlsvReader__dual_cells[dual]
               dual_corners = duals_corners[i]
               fp = f(ksi, self.reader.read_variable(self.var, np.array(dual_corners), operator=self.operator)[np.newaxis,:])
               vals.append(fp)
         return np.array(vals)
      else:
         dual, ksi = self.reader.get_dual(pt)
         dual_corners = self.__dual_cells[dual]
         fp = f(ksi, self.reader.read_variable(self.var, np.array(dual_corners), operator=self.operator)[np.newaxis,:])
         return fp


supported_amr_interpolators = {'linear','rbf','delaunay'}

class AMRInterpolator(object):
   ''' Wrapper class for interpolators, esp. at refinement interfaces.
   Supported methods:

   linear
      - (nearly) C0 continuous, regular-grid trilinear interpolant extended to collapsed hexahedral cells.
      - Non-parametric
      - Exact handling of multiply-degenerate hexahedra is missing, with the enabling hack causing errors in trilinear coordinate on the order of 1m


   Radial Basis Functions, `rbf`
      - Accurate, slow-ish, but hard to make properly continuous and number of neighbors on which to base the interpolant is not trivial to find.
      - Not continuous with regular-grid trilinear interpolants, needs to be used in the entire interpolation domain.
      - kword options: "neighbors" for number of neighbors (64)
      - basis function uses SciPy default. A free parameter.

   delaunay (not recommended)
      - Choice of triangulation is not unique with regular grids, including refinement interfaces.
      - kword options as in qhull; "qhull_options" : "QJ" (breaks degeneracies)

   '''

   def __init__(self, reader, method = "linear", cellids=np.array([1,2,3,4,5],dtype=np.int64)):
      self.__reader = reader
      self.__cellids = np.array(list(set(cellids)),dtype=np.int64)
      self.duals = {}
      # Cannot initialize an empty Delaunay
      #self.__Delaunay = Delaunay(reader.get_cell_coordinates(self.__cellids), incremental = True, qhull_options="QJ Qc Q12")

   def add_cells(self, cells):
      new_cells = [c for c in cells if c not in self.__cellids]
      self.__cellids = np.append(self.__cellids, new_cells)
      
   def get_points(self):
      return self.__reader.get_cell_coordinates(self.__cellids)
   
   def get_interpolator(self, name, operator, coords, 
                        method="linear", 
                        methodargs={
                           "RBF":{"neighbors":64}, # Harrison-Stetson number of neighbors
                           "Delaunay":{"qhull_options":"QJ"}
                           }):
      # Check for old aliases
      if method.lower() == "trilinear":
         warnings.warn("'trilinear' interpolator method renamed to 'linear' for consistency")
         method = "linear"

      methodargs["linear"] = {"reader":self.__reader, "var" : name, "op":operator}

      pts = self.__reader.get_cell_coordinates(self.__cellids)
      vals = self.__reader.read_variable(name, self.__cellids, operator=operator)
      if method == "delaunay":
         self.__Delaunay = Delaunay(self.reader.get_cell_coordinates(self.__cellids),**methodargs[method])
         return LinearNDInterpolator(self.__Delaunay, vals)
      elif method == "rbf":
         try:
            return RBFInterpolator(pts, vals, **methodargs[method])
         except Exception as e:
            warnings.warn("RBFInterpolator could not be imported. SciPy >= 1.7 is required for this class. Falling back to Hexahedral trilinear interpolator. Error given was " + str(e))
            return HexahedralTrilinearInterpolator(pts, vals, **methodargs["linear"])
      elif method == "linear":
         return HexahedralTrilinearInterpolator(pts, vals, **methodargs[method])

