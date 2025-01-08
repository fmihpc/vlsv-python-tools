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
'''
import logging
import numpy as np
import matplotlib.pyplot as plt
from reducer import DataReducerVariable
from rotation import rotateTensorToVector, rotateArrayTensorToVector
from gyrophaseangle import gyrophase_angles
import vlsvvariables
import sys
import math

mp = 1.672622e-27
elementalcharge = 1.6021773e-19
kb = 1.38065e-23
mu_0 = 1.25663706144e-6
epsilon_0 = 8.8542e-12
speedoflight = 2.9979e8

def pass_op( variable ):
   # do nothing
   return variable

def x_component( variable ):
   if np.ndim(variable) == 1:
      return variable[0]
   return variable[:,0]

def y_component( variable ):
   if np.ndim(variable) == 1:
      return variable[1]
   return variable[:,1]

def z_component( variable ):
   if np.ndim(variable) == 1:
      return variable[2]
   return variable[:,2]

def magnitude( variable ):
   return np.linalg.norm(np.asarray(variable),axis=-1)

def absolute( variable ):
   return abs(variable)

def sumv( variable ):
   # Note: this is used to sum over multipops, thus the summing axis is zero
   if np.ndim(variable) > 3:
      logging.info('Error: Number of dimensions is too large')
      return
   else:
      # First dimension: populations
      # Second dimension: cells
      # Third dimension: components
      return np.sum(np.array(variable),axis=0)

def condition_matrix_array( condition, matrices ):
   # This routine is still very slow due to for-loops
   ''' Repeats condition n times and forms an array of it
       :param: condition    Some matrix of conditions
       :param: matrices     An array of matrices on which to apply the condition
       :returns: An array of arrays which only have the conditioned elements from each matrix

       .. code-block:: python
          Example: Pick the diagonal components
          diagonal_condition = np.array([
                      [True,  False, False],
                      [False, True,  False],
                      [False, False, True]
                              ])
          matrices = np.array([
                      [[3,  1, 2],
                       [0,  3, 0],
                       [5,  1, 3]],
                      [[5,  1, 2],
                       [0,  5, 2],
                       [1,  0, 5]]
                     ])
          logging.info condition_matrix_array( diagonal_condition, matrices )
          # Output:
          # array([[3,3,3], [5,5,5]])

   '''
   # Make sure the matrices is in the correct shape
   if np.ndim(matrices) == np.ndim(condition):
      matrices = np.reshape(matrices, tuple(np.concatenate(([1],[x for x in matrices.shape]))) )
   # Repeat the condition array n times
   condition_array = condition*np.reshape(np.ones(np.product(matrices.shape)), matrices.shape)
   # Get the number of true hits in the condition:
   num_of_true = 0
   for i in np.ravel(condition):
      if i == True:
         num_of_true = num_of_true + 1
   # Extract the elements that fill the condition:
   extracted = np.extract( condition_array, matrices )
   # Reshape the matrices:
   extracted = np.reshape(extracted, (len(matrices), num_of_true))
   if len(extracted) == 1:
      extracted = extracted[0]
   # Return the extracted elements
   return extracted



def restart_B( variables ):
   ''' Data reducer function for summing perturbed_B and background_B
   '''
   return variables[0]+variables[1]

def restart_V( variables ):
   ''' Data reducer function for finding bulk V for a restart fike
   '''
   moments = variables[0]

   if np.ndim(moments)==1: # single cell
      if len(moments)==4:  # pre-multipop restart
         V = moments[1:4]/moments[0]
      elif len(moments)==5: # multipop restart
         V = moments[1:4]
      elif len(moments)==6: # eVlasiator restart
         V = moments[1:4]
      else:
         logging.info("Unrecognized length for moments!")
         return None
   else: # array of cells
      if len(moments[0,:])==4:  # pre-multipop restart
         rho = np.ma.masked_less_equal(moments[:,0],0)
         V = np.ma.divide(moments[:,1:4],rho[:,np.newaxis])
      elif len(moments[0,:])==5: # multipop restart
         V = moments[:,1:4]
      elif len(moments[0,:])==6: # eVlasiator restart
         V = moments[:,1:4]
      else:
         logging.info("Unrecognized length for moments!")
         return None
   return V

def restart_rho( variables ):
   ''' Data reducer function for calculating proton number density rho from restart file
   '''
   moments = variables[0]

   if np.ndim(moments)==1: # single cell
      if len(moments)==4:  # pre-multipop restart
         rho = moments[0]
      else:
         logging.info("Unable to determine rho from moments!")
         return None
   else: # array of cells
      if len(moments[0,:])==4:  # pre-multipop restart
         rho = moments[:,0]
      else:
         logging.info("Unable to determine rho from moments!")
         return None
   return rho

def restart_rhom( variables ):
   ''' Data reducer function for calculating rhom from restart file
   '''
   moments = variables[0]
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp

   if np.ndim(moments)==1: # single cell
      if len(moments)==4:  # pre-multipop restart
         rhom = moments[0]*mass
      elif len(moments)==5: # multipop restart
         rhom = moments[0]
      elif len(moments)==6: # eVlasiator restart
         rhom = moments[0]
      else:
         logging.info("Unrecognized length for moments!")
         return None
   else: # array of cells
      if len(moments[0,:])==4:  # pre-multipop restart
         rhom = moments[:,0]*mass
      elif len(moments[0,:])==5: # multipop restart
         rhom = moments[:,0]
      elif len(moments[0,:])==6: # eVlasiator restart
         rhom = moments[:,0]
      else:
         logging.info("Unrecognized length for moments!")
         return None
   return rhom

def restart_rhoq( variables ):
   ''' Data reducer function for calculating rhoq from restart file
   '''
   moments = variables[0]
   charge = vlsvvariables.speciescharge[vlsvvariables.activepopulation]*elementalcharge

   if np.ndim(moments)==1: # single cell
      if len(moments)==4:  # pre-multipop restart
         rhoq = moments[0]*charge
      elif len(moments)==5: # multipop restart
         rhoq = moments[4]
      elif len(moments)==6: # eVlasiator restart
         rhoq = moments[4]
      else:
         logging.info("Unrecognized length for moments!")
         return None
   else: # array of cells
      if len(moments[0,:])==4:  # pre-multipop restart
         rhoq = moments[:,0]*charge
      elif len(moments[0,:])==5: # multipop restart
         rhoq = moments[:,4]
      elif len(moments[0,:])==6: # eVlasiator restart
         rhoq = moments[:,4]
      else:
         logging.info("Unrecognized length for moments!")
         return None
   return rhoq

def rhom( variables ):
   ''' Data reducer function for calculating rhom from pre-multipop file
   '''
   rho = variables[0]
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp
   return rho*mass

def rhoq( variables ):
   ''' Data reducer function for calculating rhoq from pre-multipop file
   '''
   rho = variables[0]
   charge = vlsvvariables.speciescharge[vlsvvariables.activepopulation]*elementalcharge
   return rho*charge

def precipitationintegralenergyflux( variables ):
   ''' Data reducer function for calculating the integral energy flux from differential
       energy fluxes of precipitating particles
       input: precipitationdiffflux
   '''
   diffflux = variables[0]
   energybins = np.asarray(vlsvvariables.speciesprecipitationenergybins[vlsvvariables.activepopulation]) # in eV
   # Building the energy bin widths
   dlogener = np.log(energybins[1]) - np.log(energybins[0])
   Ebinedges = np.zeros(len(energybins)+1)
   Ebinedges[1:-1] = np.sqrt(energybins[1:]*energybins[:-1])
   Ebinedges[0] = np.exp(np.log(energybins[0])-dlogener)
   Ebinedges[-1] = np.exp(np.log(energybins[-1])+dlogener)
   deltaE = Ebinedges[1:]-Ebinedges[:-1] # in eV

   # Calculating the quantity of energy in each bin and summing, masking too low values
   energyPerBin = diffflux*deltaE*energybins
   integralenergyflux = np.ma.masked_less_equal(energyPerBin.sum(axis=1),0.)
   # Result in eV/(cm2 s sr), convert in more usual unit of keV/(cm2 s sr) when returning
   return integralenergyflux/1e3

def precipitationmeanenergy( variables ):
   ''' Data reducer function for calculating the mean particle energy from differential
       energy fluxes of precipitating particles
       input: precipitationdiffflux
   '''
   diffflux = variables[0]
   energybins = np.asarray(vlsvvariables.speciesprecipitationenergybins[vlsvvariables.activepopulation]) # in eV
   # Building the energy bin widths
   dlogener = np.log(energybins[1]) - np.log(energybins[0])
   Ebinedges = np.zeros(len(energybins)+1)
   Ebinedges[1:-1] = np.sqrt(energybins[1:]*energybins[:-1])
   Ebinedges[0] = np.exp(np.log(energybins[0])-dlogener)
   Ebinedges[-1] = np.exp(np.log(energybins[-1])+dlogener)
   deltaE = Ebinedges[1:]-Ebinedges[:-1] # in eV

   # Calculating the number flux and so on, masking too low values
   particlesPerBin = diffflux*deltaE
   integralnumberflux = np.ma.masked_less_equal(particlesPerBin.sum(axis=1),0.)
   energyPerBin = particlesPerBin*energybins
   integralenergyflux = np.ma.masked_less_equal(energyPerBin.sum(axis=1),0.)
   meanenergy = np.ma.divide(integralenergyflux,integralnumberflux)
   # Result in eV, convert in more usual unit of keV when returning
   return meanenergy/1e3

def v( variables ):
   ''' Data reducer function for getting velocity from rho and rho_v
       variables[0] = rho_v and variables[1] = rho
   '''
   epsilon = sys.float_info.epsilon
   rho_v = np.array(variables[0])
   rho = np.ma.masked_less_equal(np.ma.masked_invalid(np.array(variables[1])),0)
   if np.ndim(rho) == 0:
      return np.ma.divide(rho_v,rho)
   else:
      return np.ma.divide(rho_v,rho[:,np.newaxis])

def vms( variables ):
   ''' Data reducer function for getting magnetosonic velocity 
       input: P, rhom, B
   '''
   epsilon = sys.float_info.epsilon
   P = variables[0]
   rho_m = np.ma.masked_less_equal(np.ma.masked_invalid(np.array(variables[1])),0)
   B = variables[2]
   if np.ndim(B) == 1:
      Btot = np.sqrt( np.sum( np.square(B) ) )
   else:
      Btot = np.sqrt( np.sum(np.square(B),axis=1) )
   vs = np.sqrt( np.ma.divide( P*5.0/3.0, rho_m ) )
   vA = np.ma.divide( Btot,np.sqrt( mu_0*rho_m ) )
   vms = np.sqrt( np.square(vs) + np.square(vA) )
   return vms

def vs( variables ):
   ''' Data reducer function for getting the sound speed
       input: P, rhom
   '''
   epsilon = sys.float_info.epsilon
   P = variables[0]
   rho_m = np.ma.masked_less_equal(np.ma.masked_invalid(np.array(variables[1])),0)
   vs = np.sqrt( np.ma.divide( P*5.0/3.0, rho_m ) )
   return vs

def va( variables ):
   ''' Data reducer function for getting the Alfven velocity
       input: rhom, B
   '''
   epsilon = sys.float_info.epsilon
   rho_m = np.ma.masked_less_equal(np.ma.masked_invalid(np.array(variables[0])),0)
   B = variables[1]
   if np.ndim(B) == 1:
      Btot = np.sqrt( np.sum( np.square(B) ) )
   else:
      Btot = np.sqrt( np.sum(np.square(B),axis=1) )
   vA = np.ma.divide( Btot,np.sqrt( rho_m*mu_0 ) )
   return vA

def MA( variables ):
   ''' Data reducer function for getting the Alfvenic Mach number
   '''
   bulkv = np.linalg.norm(variables[0],axis=-1)
   Alfvenspeed = np.ma.masked_less_equal(variables[1],0)
   MA = np.ma.divide(bulkv, Alfvenspeed)
   return MA

def Mms( variables ):
   ''' Data reducer function for getting the magnetosonic Mach number
   '''
   bulkv = np.linalg.norm(variables[0],axis=-1)
   magnetosonicspeed = np.ma.masked_less_equal(variables[1],0)
   Mms = np.ma.divide(bulkv, magnetosonicspeed)
   return Mms

def ParallelVectorComponent( variables ):
   ''' Data reducer function for vector component parallel to the magnetic field (or another vector)
   '''
   inputvector = variables[0]
   if inputvector is None:
      raise ValueError("Missing inputvector")
   bgvector = variables[1]
   if( np.ndim(inputvector)==1 ):
      if( np.linalg.norm(bgvector) != 0):
         bgnorm = bgvector/np.linalg.norm(bgvector)
         return (inputvector*bgnorm).sum()
      else:
         return 0
   else:
      bgnorm = np.ma.divide(bgvector, np.ma.masked_equal(np.linalg.norm(bgvector, axis=-1),0)[:,np.newaxis])
      return (inputvector*bgnorm).sum(-1)

def PerpendicularVectorComponent( variables ):
   ''' Data reducer function for vector component perpendicular to the magnetic field (or another vector)
   '''
   inputvector = variables[0]
   if inputvector is None:
      raise ValueError("Missing inputvector")
   bgvector = variables[1]
   if( np.ndim(inputvector)==1 ):
      if( np.linalg.norm(bgvector) != 0):
         bgnorm = bgvector / np.linalg.norm(bgvector)
         vpara = (inputvector*bgnorm).sum()
         vmag = np.linalg.norm(inputvector)
         presqrt = vmag*vmag - vpara*vpara
         if (presqrt>=0):
            return np.sqrt(presqrt)
         else:
            return 0
      else:
         return 0
   else:
      bgnorm = np.ma.divide(bgvector, np.ma.masked_equal(np.linalg.norm(bgvector, axis=-1),0)[:,np.newaxis])
      vpara = (inputvector*bgnorm).sum(-1)
      vmag = np.linalg.norm(inputvector, axis=-1)
      presqrt = np.sqrt(vmag*vmag - vpara*vpara)
      return np.sqrt(np.ma.masked_less(presqrt,0))

def FullTensor( variables ):
   ''' Data reducer function to reconstruct a full tensor from diagonal and off-diagonal
       components (e.g. the pressure tensor)
   '''
   TensorDiagonal = np.array(variables[0])
   TensorOffDiagonal = np.array(variables[1])
   if(np.ndim(TensorDiagonal)==1 ):      
      TensorDiagonal = TensorDiagonal.reshape(1,3)[0]
      TensorOffDiagonal = TensorOffDiagonal.reshape(1,3)[0]
      return np.array([[TensorDiagonal[0], TensorOffDiagonal[2], TensorOffDiagonal[1]],
                       [TensorOffDiagonal[2], TensorDiagonal[1], TensorOffDiagonal[0]],
                       [TensorOffDiagonal[1], TensorOffDiagonal[0], TensorDiagonal[2]]])
   else:
      result = np.empty([len(TensorDiagonal[:,0]),3,3]) # Warning, unitialized!
      result[:,0,0] = TensorDiagonal[:,0]
      result[:,0,1] = TensorOffDiagonal[:,2]
      result[:,0,2] = TensorOffDiagonal[:,1]
      result[:,1,0] = TensorOffDiagonal[:,2]
      result[:,1,1] = TensorDiagonal[:,1]
      result[:,1,2] = TensorOffDiagonal[:,0]
      result[:,2,0] = TensorOffDiagonal[:,1]
      result[:,2,1] = TensorOffDiagonal[:,0]
      result[:,2,2] = TensorDiagonal[:,2]
      return result

def RotatedTensor( variables ):
   ''' Data reducer for rotating e.g. the pressure tensor to align the z-component 
       with a vector, e.g. the magnetic field
   '''
   Tensor = variables[0]
   B = variables[1]
   if( np.ndim(B)==1 ):
      B = B.reshape(1,3)[0]
      Tensor = Tensor.reshape(3,3)
      return rotateTensorToVector(Tensor, B)
   else:
      return rotateArrayTensorToVector(Tensor, B)

def ParallelTensorComponent( variables ):
   ''' Data reducer for finding the parallel component from a rotated field-aligned tensor
   '''
   RotatedTensor = variables[0]
   if( np.ndim(RotatedTensor)==2 ):
      return RotatedTensor[2,2]
   else:
      return RotatedTensor[:,2,2]

def PerpendicularTensorComponent( variables ):
   ''' Data reducer for finding the perpendicular component from a rotated field-aligned tensor
       e.g. perpendicular pressure
   '''
   RotatedTensor = variables[0]
   if( np.ndim(RotatedTensor)==2 ):
      return 0.5*(RotatedTensor[0,0] + RotatedTensor[1,1])
   else:
      return 0.5*(RotatedTensor[:,0,0] + RotatedTensor[:,1,1])

def J( variables ):
   ''' Data reducer taking a jacobian (assume 9-component vector) and extracting the current
   via curl from the components of the jacobian (background or perturbed, as long as it has 9
    components)

   '''
   stack = True
   if(variables[0].shape == (9,)):
      stack = False
      variables[0] = np.array([variables[0]]) # I want this to be a stack of tensors
   jacobian = variables[0]#.reshape((variables[0].shape[0],3,3))
   # jacobian = jacobian.transpose((0,2,1)) # The axes are flipped at some point, correcting for that
   J = np.zeros((jacobian.shape[0],3))

   J[:,0] = (jacobian[:,7]-jacobian[:,5])/mu_0
   J[:,1] = (jacobian[:,2]-jacobian[:,6])/mu_0
   J[:,2] = (jacobian[:,3]-jacobian[:,1])/mu_0
   if stack:
      return J
   else:
      return J[0,:]


   logging.info("Error in J")
   return -1

def TensorFromScalars(variables):
   '''Construct a 9-element vector ("tensor") from nine scalar fields.
   '''

   return np.stack(np.array(
                    [variables[0], variables[1], variables[2],
                     variables[3], variables[4], variables[5],
                     variables[6], variables[7], variables[8]]
                   ),
                   axis=-1)


def Anisotropy( variables ):
   ''' Data reducer for finding the ratio of perpendicular to parallel components of a tensor
   '''
   RotatedTensor = variables[0]
   if( np.ndim(RotatedTensor)==2 ):
      divisor = np.ma.masked_equal(np.ma.masked_invalid(RotatedTensor[2,2]),0)
      return 0.5*np.ma.divide(RotatedTensor[0,0] + RotatedTensor[1,1], divisor)
   else:
      divisor = np.ma.masked_equal(np.ma.masked_invalid(RotatedTensor[:,2,2]),0)
      return 0.5*np.ma.divide(RotatedTensor[:,0,0] + RotatedTensor[:,1,1], divisor)

def Pressure( variables ):
   ''' Data reducer for finding the scalar pressure
   '''
   PTensorDiagonal = variables[0]
   return 1.0/3.0 * np.ma.sum(np.ma.masked_invalid(PTensorDiagonal),axis=-1)

def Pdyn( variables ):
   ''' Data reducer function for dynamic pressure
       Pdyn = rho_m*V^2
       input: V, rhom
   '''
   Vmag = np.linalg.norm(np.array(variables[0]), axis=-1)
   rhom = np.array(variables[1])
   return Vmag*Vmag*rhom

def Pdynx( variables ):
   ''' Data reducer function for dynamic pressure with just V_x
       input: V, rhom
   '''
   rhom = np.array(variables[1])
   V = np.array(variables[0])
   if( np.ndim(V)==2 ):
      Vx = V[:,0]
   else:
      Vx = V[0]
   return Vx*Vx*rhom

def Poynting( variables ):
   ''' Data reducer for the Poynting vector
   '''
   E=np.array(variables[0])
   B=np.array(variables[1])
   return np.cross(E, B) / mu_0

def Hallterm( variables ):
   ''' Data reducer for the deducing an estimate of the Hall term
   '''
   E=np.array(variables[0])
   V=np.array(variables[1])
   B=np.array(variables[2])
   return E + np.cross(V, B)

def Temperature( variables ):
   ''' Data reducer for converting pressure to temperature
   '''
   Pressure = variables[0] # either a tensor, vector, array, or value
   rho = variables[1] # eithern array or a value
   divisor = np.ma.masked_less_equal( np.ma.masked_invalid(np.array(rho)),0) * kb
   # assumes first dimension is either single cell or a single-dimension array
   if np.ndim(divisor)==0: # single cell
      if np.ndim(Pressure)==0:
         return np.ma.divide(Pressure, divisor)
      elif np.ndim(Pressure)==1:
         return np.ma.divide(Pressure, divisor[np.newaxis])
      elif np.ndim(Pressure)==2:
         return np.ma.divide(Pressure, divisor[np.newaxis,np.newaxis])
   else: # array of cellids
      if np.ndim(Pressure)==1:
         return np.ma.divide(Pressure, divisor)
      elif np.ndim(Pressure)==2:
         return np.ma.divide(Pressure, divisor[:,np.newaxis])
      elif np.ndim(Pressure)==3:
         return np.ma.divide(Pressure, divisor[:,np.newaxis,np.newaxis])
   # Should not reach here...
   logging.info("Error finding dimensions in calculating temperature!")
   return -1


def gyrotropy(variables):
# see Appendix in Swisdak 2016: https://doi.org/10.1002/2015GL066980  
    
    Pdiag, Poffdiag, B = variables
    
    if(np.ndim(B)==2):
        Pxx=Pdiag[:,0]
        Pyy=Pdiag[:,1]
        Pzz=Pdiag[:,2]
        Pxy = Poffdiag[:,0]
        Pxz = Poffdiag[:,1]
        Pyz = Poffdiag[:,2]
        B_norm = B / np.sqrt(np.sum(np.asarray(B)**2,axis=-1)[:, np.newaxis])
        bx,by,bz = B_norm[:,0],B_norm[:,1],B_norm[:,2]

    elif(np.ndim(B)==1): 
        Pxx=Pdiag[0]
        Pyy=Pdiag[1]
        Pzz=Pdiag[2]
        Pxy = Poffdiag[0]
        Pxz = Poffdiag[1]
        Pyz = Poffdiag[2]
        B_norm = B / np.sqrt( B[0]**2 + B[1]**2+ B[2]**2 )        
        bx,by,bz = B_norm[0],B_norm[1],B_norm[2]

    I1 = Pxx + Pyy + Pzz
    I2 = Pxx * Pyy + Pyy * Pzz + Pxx * Pzz  - (Pxy**2 + Pxz**2 + Pyz**2)
    Ppar = (  bx**2 * Pxx + by**2 * Pyy + bz**2 * Pzz +     
           2 * (bx * by * Pxy + bx * bz * Pxz + by * bz * Pyz ) )    
    Q = 1 - 4 * I2 / (  (I1 - Ppar)*(I1 + 3* Ppar)  )
    return Q

def MagneticPressure( variables ):
   ''' Data reducer for finding the magnetic pressure
   '''
   Magneticfield = variables[0]
   return np.sum(np.asarray(Magneticfield)**2,axis=-1) / 2.0 / mu_0

def beta( variables ):
   ''' Data reducer for finding the plasma beta
   '''
   Pressure = variables[0]
   Magneticfield = variables[1]   
   return 2.0 * mu_0 * np.ma.divide(Pressure, np.sum(np.asarray(Magneticfield)**2,axis=-1))

def beta_star( variables ):
   ''' Data reducer for finding the Brenner+2021 plasma beta
      beta* = (P_thermal + P_ram)/P_magnetic
   '''
   Pressure_thermal = variables[0]
   Pressure_dynamic = variables[1]
   Magneticfield = variables[2]   
   return 2.0 * mu_0 * np.ma.divide(Pressure_thermal + Pressure_dynamic, np.sum(np.asarray(Magneticfield)**2,axis=-1))


def rMirror( variables ):
   # More efficient file access, now just takes PTensor and B
   PT = variables[0]
   B = variables[1]
   PTRotated =  RotatedTensor([PT,B])
   TAniso = Anisotropy([PTRotated]) # PAniso == TAniso
   PPerp = PerpendicularTensorComponent([PTRotated])
   betaPerp = beta([PPerp,B])
   return betaPerp * (TAniso - 1)   

def thermalvelocity( variables ):
   Temperature = variables[0]
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp
   # Corrected to calculate the mean speed sqrt(8kT/pi m)
   thermalvelocity = np.sqrt(Temperature*(kb*8./(mass*math.pi)))
   return thermalvelocity

def Vstream( variables ):
   raise NotImplementedError("rhoVBackstream, rhoBackstream not defined here. Check implementaiton if required!")
   # rhoVstream = variables[0]
   # rhostream = variables[1]
   # rhoVNonBackstream = variables[2]
   # rhoNonBackstream = variables[3]
   # # get velocity of both populations:
   # vBackstream = v( [rhoVBackstream, rhoBackstream] )
   # vNonBackstream = v( [rhoVNonBackstream, rhoNonBackstream] )
   # vBeam = vBackstream - vNonBackstream
   # return vBeam # <- is a vector quantity

def v_beam( variables ):
   vBackstream = variables[0]
   vNonBackstream = variables[1]
   vBeam = vBackstream - vNonBackstream
   return vBeam # <- is a vector quantity

def v_beam_ratio( variables ):
   vBackstream = magnitude(variables[0])
   vNonBackstream = magnitude(variables[1])
   divisor = np.ma.masked_less_equal( np.ma.masked_invalid(vNonBackstream),0)
   return np.ma.divide(vBackstream,divisor)

def Bz_linedipole_avg( variables ):
   # This reducer needs to be verified
   x = variables[0]
   y = variables[1]
   z = variables[2]
   dx = variables[3]
   dy = variables[4]
   dz = variables[5]
   return -126.2e6*((dx+x)/(dx*(z**2+(dx+x)**2)) - x/(dx*(z**2+x**2)))

def Bz_linedipole_diff( variables ):
   # This reducer needs to be verified
   Bb = variables[0]
   Bzldp = variables[1]
   logging.info(Bzldp.shape)
   divisor = np.ma.masked_less_equal( np.ma.masked_invalid(magnitude(Bb)),0)
   return np.ma.divide(np.abs(Bb[:,2] - Bzldp), divisor)

def gyrophase_relstddev( variables, velocity_cell_data, velocity_coordinates ):
   # This reducer needs to be verified
   bulk_velocity = variables[0]
   B = variables[1]
   B_unit = B / np.linalg.norm(B)
   
   gyrophase_data = gyrophase_angles(bulk_velocity, B_unit, velocity_cell_data, velocity_coordinates)
   histo = plt.hist(gyrophase_data[0].data, weights=gyrophase_data[1].data, bins=36, range=[-180.0,180.0], log=False, normed=1)
   return np.std(histo[0])/np.mean(histo[0])

def Dng( variables ):
   # This reducer needs to be verified
   # This routine is still very slow due to for-loops
   PTensor = variables[0]
   PParallel = variables[1]
   PPerpendicular = variables[2]
   B = variables[3]
   # following Aunai et al 2103 syntax
   if( np.ndim(PTensor)==2 ):
      B_normalized = np.divide(B, np.sqrt(np.sum(B[:]**2)))
      idMatrix = np.diag(np.ones(3))
      G = PPerpendicular*idMatrix + (PParallel - PPerpendicular) * np.outer(B_normalized, B_normalized)
      N = PTensor - G
      return 2.0 * np.linalg.norm(N, 'fro') / PTensor.trace()
   else:
      B_normalized = np.divide(B, np.sqrt(np.sum(B[:]**2, axis=1))[:,None])
      idMatrix = np.diag(np.ones(3))
      G = [PPerpendicular[i]*idMatrix + (PParallel[i] - PPerpendicular[i]) * np.outer(B_normalized[i], B_normalized[i]) for i in np.arange(len(PParallel))]
      N = PTensor - G
      return [np.divide(2.0*np.linalg.norm(N[i], 'fro'), PTensor[i].trace()) for i in np.arange(len(PParallel))]

def ion_inertial( variables ):
   rho = np.ma.masked_less_equal(np.ma.masked_invalid(np.array(variables[0])),0)
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp
   charge = vlsvvariables.speciescharge[vlsvvariables.activepopulation]*elementalcharge
   omegapi = np.sqrt(rho * charge * charge / (mass*epsilon_0))
   di = np.ma.divide(speedoflight,omegapi)
   return di

def gyroperiod( variables ):
   B = np.array(variables[0])
   Bmag = np.linalg.norm(B,axis=-1)
   Bmag = np.ma.masked_less_equal(np.ma.masked_invalid(Bmag),0)
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp
   charge = vlsvvariables.speciescharge[vlsvvariables.activepopulation]*elementalcharge
   omegaci = abs(charge*Bmag/mass)
   #return np.ma.divide(2.*math.pi,omegaci)
   return 2.*math.pi*(omegaci**-1)

def plasmaperiod( variables ):
   rho = np.ma.masked_less_equal(np.ma.masked_invalid(np.array(variables[0])),0)
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp
   charge = vlsvvariables.speciescharge[vlsvvariables.activepopulation]*elementalcharge
   omegapi = abs(np.sqrt(rho/(mass*epsilon_0))*charge)
   #return np.ma.divide(2.*math.pi,omegapi)
   return 2.*math.pi*(omegapi**-1)

def larmor( variables ):
   B = variables[0]
   Bmag = np.linalg.norm(B, axis=-1)
   vth = variables[1] # thermal velocity
   mass = vlsvvariables.speciesamu[vlsvvariables.activepopulation]*mp
   charge = vlsvvariables.speciescharge[vlsvvariables.activepopulation]*elementalcharge
   return np.ma.divide(mass*vth,charge*Bmag)

def firstadiabatic( variables ):
   Tperp = variables[0]
   bvector = variables[1]
   B = np.linalg.norm(bvector, axis=-1)
   B = np.ma.masked_less_equal(np.ma.masked_invalid(B),0)
   return np.ma.divide(Tperp,B)

def vg_e_vol( variables, reader):
   fg_e = reader.read_fg_variable_as_volumetric("fg_e")
   return reader.fsgrid_array_to_vg(fg_e)

def JPerB_criteria( variables ):
   ''' Data reducer function for calculating J/B refinement criterion as it is done in Vlasiator
   '''
   J_per_B = variables[0]
   return np.log2(J_per_B * vlsvvariables.cellsize + 1E-30)# + vlsvvariables.J_per_B_modifier


def vg_coordinates_cellcenter( variables, reader):
   cellids = variables[0]
   return reader.get_cell_coordinates(cellids)

def vg_coordinates_lowcorner( variables, reader):
   cellids = variables[0]
   dxs = variables[1]
   return reader.get_cell_coordinates(cellids)-dxs/2

def vg_dx(variables, reader):
   cellids = variables[0]
   return reader.get_cell_dx(cellids)

def vg_reflevel(variables, reader):
   cellids = variables[0]
   return reader.get_amr_level(cellids)

def _normalize(vec):
   '''
      (private) helper function, normalizes a multidimensinonal array of vectors
      assume [...., 3] array
   '''
   return vec / np.linalg.norm(vec, axis = -1)[:, np.newaxis]

def ig_E( variables, reader ):
   ''' calculate in-plane ionospheric electric field from the ionospheric potential 'ig_potential'

       :returns: (in-plane) ionospheric electric field [V/m]

        The in-plane electric field has 2 degrees of freedom (orientation and magnitude),
        uniquely matching the potential at 2 corners of the triangular element.

        The algorithm:
            1. WLOG let one corner have potential V0=0. Let r1, r2 be position vectors for the other corners, at potentials V1 & V2.
            2. Construct an orthonormal basis {b1, b3} that spans the triangular element, where b1 || r1
            3. calculate the unique electric field E1 || b1, that gives the correct potential V1 at position r1
            4. infer the potential V1_B at the location r1_B (see figure)
            5. calculate the unique electric field E3 || b3, that is required to match the potentials at r1_B and r2.
            6. Calculate the total electric field E = E1 + E3
               .
              ^^\
     r2,(V2) / | 
            /  | 
           /   | r3 (d3), b3, E3
          /    |
         /    _|     \
        .____|_._____>.  r1 (d1), b1, E1, (V1)
    V0=0    r1_B, V1_B

       E = E1 + E3
   '''
   ig_potential = variables[0]
   #ig_potential = reader.read_variable('ig_potential')   # shape (n_nodes)
   c = reader.get_ionosphere_element_corners()          # Element corners. shape (n_elements, 3)
   n = reader.get_ionosphere_node_coords()       # Nodes. shape (n_nodes, 3)
   p = n[c,:]                               # shape (n_elements, 3, 3),  indexing (element, triangle corner, x-y-z position)     
   r1 = p[:,1,:] - p[:,0,:]
   r2 = p[:,2,:] - p[:,0,:]
   r_shape = r1.shape
   d1 = np.repeat(np.linalg.norm(r1, axis = 1), 3).reshape(r_shape)   # r1 distance (repeated 3 times for convenience)
   # define some orthogonal basis vectors b1 and b3 (where b1 || r1) that span the triangular element:
   b1 = _normalize(r1)
   r1_B = b1 * np.repeat(np.nansum(r2 * b1, axis = 1), 3).reshape(r_shape)
   r3 = r2 - r1_B
   b3 = _normalize(r3)
   d3 = np.repeat(np.linalg.norm(r3, axis = 1), 3).reshape(r_shape)
   # electric field E = -grad V:
   ig_potential_c = ig_potential[c]
   V1 = np.repeat(ig_potential_c[:,1] - ig_potential_c[:,0], 3).reshape(r_shape)  # potential diff. btw. nodes 1 and 0 of a given triangular face
   V2 = np.repeat(ig_potential_c[:,2] - ig_potential_c[:,0], 3).reshape(r_shape)
   E1 = -b1 * V1 / d1         # (E1 || b1)
   V1_B =  np.repeat(np.nansum(-E1 * r1_B, axis = 1), 3).reshape(r_shape)
   E3 = -b3 * (V2 - V1_B) / d3         # (E_1 || b1)
   E = E1 + E3
   # checked: V1 === np.nansum(-E * r1, axis = 1),  and   V2 === np.nansum(-E * r2, axis = 1)
   return E
   
def ig_inplanecurrent( variables, reader ):
   ''' Calculate in-plane current vector J, from J_i = sigma_ij * E_j

        :returns: height-integrated in-plane current [A/m], from the ionospheric electric field ig_E

        This is probably only needed to reconstruct the ig_inplanecurrent for .vlsv files
        that happen to be missing this variable (as in t=501-1000 in run FHA)

        NOTE: this data reducer produces an in-plane current that is in the triangular element's plane
              Tests have shown that saved variable 'ig_inplanecurrent' does NOT lie in the plane for some .vlsvs
              TODO: understand this discrepancy
   '''
   E = variables[0]
   ig_sigmah = np.repeat(reader.read_ionosphere_node_variable_at_elements('ig_sigmah'), 3).reshape(E.shape) 
   ig_sigmap = np.repeat(reader.read_ionosphere_node_variable_at_elements('ig_sigmap'), 3).reshape(E.shape)
   ig_r = reader.get_ionosphere_element_coords()
   ig_r_hat = _normalize(ig_r)
   n = reader.get_ionosphere_node_coords()       # nodes: shape (n_nodes, 3) vertices
   c = reader.get_ionosphere_element_corners()   # corners of elements: indices integers 0-(n_nodes-1), shape (n_elements, 3)
   p = n[c,:]                               # shape (n_elements, 3, 3),  indexing (element, triangle corner, x-y-z position)
   r1 = p[:,1,:] - p[:,0,:]
   r2 = p[:,2,:] - p[:,0,:]
   normal = _normalize(np.cross(r1, r2))    # normal direction wrt element face
   normal = normal * np.repeat(np.sign(np.sum(ig_r_hat * normal, axis = 1)), 3).reshape(normal.shape)  # ensure normals point outward
   # Account for B-field polarity: ig_b_hat = normal in southern hemisphere, and ig_b_hat = -normal in northern hemisphere:
   ig_b_hat = -normal * np.repeat(np.sign(ig_r[:,2]), 3).reshape(normal.shape)  # TODO: check precision of this z-coordinat test near the equator
   return ig_sigmap * E - ig_sigmah * np.cross(E, ig_b_hat)

#list of operators. The user can apply these to any variable,
#including more general datareducers. Can only be used to reduce one
#variable at a time
data_operators = {}
data_operators["pass"] = pass_op
data_operators["magnitude"] = magnitude
data_operators["absolute"] = absolute
data_operators["sum"] = sumv
data_operators["x"] = x_component
data_operators["y"] = y_component
data_operators["z"] = z_component

# Hacky way to add vector access
def makelambda(index):
   return lambda data: data[index] if np.ndim(data)==1 else data[:,index]
for i in range(50):
   data_operators[i] = makelambda(i)
   data_operators[str(i)] = data_operators[i]

   
# When vlsvreader tries to read data, it will check using the following order:
# 1) Is the variable directly in the file?
# 2) Is the name something that exists in the file, but only per-population? (answer: sum over populations)
# 3) Is the name a regular datareducer?
# 4) Is there a multipop datareducer for this variable
# The same logic is used for the variables required by datareducers as well.

# if the requested variable starts with "vg_", use the v5 datareducers, otherwise the originals.

#datareducers with more complex, case dependent structure.
datareducers = {}
datareducers["v"] =                      DataReducerVariable(["rho_v", "rho"], v, "m/s", 3, latex=r"$V$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vms"] =                    DataReducerVariable(["pressure", "rhom", "b"], vms, "m/s", 1, latex=r"$v_\mathrm{ms}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vs"] =                     DataReducerVariable(["pressure", "rhom"], vs, "m/s", 1, latex=r"$v_\mathrm{s}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["va"] =                     DataReducerVariable(["rhom", "b"], va, "m/s", 1, latex=r"$v_\mathrm{A}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["ma"] =                     DataReducerVariable(["v", "va"], MA, "", 1, latex=r"$M_\mathrm{A}$",latexunits=r"")
datareducers["mms"] =                    DataReducerVariable(["v", "vms"], Mms, "", 1, latex=r"$M_\mathrm{ms}$",latexunits=r"")

datareducers["vparallel"] =              DataReducerVariable(["v", "b"], ParallelVectorComponent, "m/s", 1, latex=r"$V_\parallel$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vperpendicular"] =         DataReducerVariable(["v", "b"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_\perp$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vparallelbackstream"] =    DataReducerVariable(["vbackstream", "b"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{st}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vperpendicularbackstream"]=DataReducerVariable(["vbackstream", "b"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{st}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vparallelnonbackstream"] =    DataReducerVariable(["vnonbackstream", "b"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{th}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vperpendicularnonbackstream"]=DataReducerVariable(["vnonbackstream", "b"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{th}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")

datareducers["eparallel"] =              DataReducerVariable(["e", "b"], ParallelVectorComponent, "V/m", 1, latex=r"$E_\parallel$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
datareducers["eperpendicular"] =         DataReducerVariable(["e", "b"], PerpendicularVectorComponent, "V/m", 1, latex=r"$E_\perp$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
datareducers["ejeparallel"] =              DataReducerVariable(["eje", "b"], ParallelVectorComponent, "V/m", 1, latex=r"$EJE_\parallel$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
datareducers["ejeperpendicular"] =         DataReducerVariable(["eje", "b"], PerpendicularVectorComponent, "V/m", 1, latex=r"$EJE_\perp$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")

datareducers["pdyn"] =            DataReducerVariable(["v", "rhom"], Pdyn, "Pa", 1, latex=r"$P_\mathrm{dyn}$",latexunits=r"$\mathrm{Pa}$")
datareducers["pdynx"] =            DataReducerVariable(["v", "rhom"], Pdynx, "Pa", 1, latex=r"$P_\mathrm{dyn,x}$",latexunits=r"$\mathrm{Pa}$")

datareducers["p_magnetic"] =            DataReducerVariable(["b"], MagneticPressure, "Pa", 1, latex=r"$P_\mathrm{mag}$",latexunits=r"$\mathrm{Pa}$")

datareducers["poynting"] = DataReducerVariable(["e", "b"], Poynting, "W/m2", 3, latex=r"$S$", latexunits=r"\mathrm{W}\,\mathrm{m}^{-2}$")
datareducers["hallterm"] = DataReducerVariable(["e", "v", "b"], Hallterm, "V/m", 3, latex=r"$E_\mathrm{Hall}$", latexunits=r"\mathrm{V}\,\mathrm{m}^{-1}$")
datareducers["firstadiabatic"] =    DataReducerVariable(["tperpendicular","b"], firstadiabatic, "K/T", 1, latex=r"$T_\perp B^{-1}$",latexunits=r"$\mathrm{K}\,\mathrm{T}^{-1}$")

# Reducers for simplifying access calls for old and/or new output data versions
datareducers["vbackstream"] =            DataReducerVariable(["rhovbackstream", "rhobackstream"], v, "m/s", 3, latex=r"$V_\mathrm{st}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vnonbackstream"] =         DataReducerVariable(["rhovnonbackstream", "rhononbackstream"], v, "m/s", 3, latex=r"$V_\mathrm{th}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["rhom"] =                   DataReducerVariable(["rho"], rhom, "kg/m3", 1, latex=r"$\rho_m$",latexunits=r"$\mathrm{kg}\,\mathrm{m}^{-3}$")
datareducers["rhoq"] =                   DataReducerVariable(["rho"], rhoq, "C/m3", 1, latex=r"$\rho_q$",latexunits=r"$\mathrm{C}\,\mathrm{m}^{-3}$")
# Reducers for restart files
datareducers["b"] =                      DataReducerVariable(["background_b", "perturbed_b"], restart_B, "T", 3, latex=r"$B$",latexunits=r"\mathrm{T}")
datareducers["restart_v"] =              DataReducerVariable(["moments"], restart_V, "m/s", 3, latex=r"$V$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["restart_rho"] =            DataReducerVariable(["moments"], restart_rho, "1/m3", 1, latex=r"$n_\mathrm{p}$",latexunits=r"$\mathrm{m}^{-3}$")
datareducers["restart_rhom"] =           DataReducerVariable(["moments"], restart_rhom, "kg/m3", 1, latex=r"$\rho_m$",latexunits=r"$\mathrm{kg}\,\mathrm{m}^{-3}$")
datareducers["restart_rhoq"] =           DataReducerVariable(["moments"], restart_rhoq, "C/m3", 1, latex=r"$\rho_q$",latexunits=r"$\mathrm{C}\,\mathrm{m}^{-3}$")

datareducers["pressure"] =               DataReducerVariable(["ptensordiagonal"], Pressure, "Pa", 1, latex=r"$P$", latexunits=r"$\mathrm{Pa}$")
datareducers["ptensor"] =                DataReducerVariable(["ptensordiagonal", "ptensoroffdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}$", latexunits=r"$\mathrm{Pa}$")
datareducers["ptensorrotated"] =         DataReducerVariable(["ptensor", "b"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pparallel"] =              DataReducerVariable(["ptensorrotated"], ParallelTensorComponent, "Pa", 1, latex=r"$P_\parallel$", latexunits=r"$\mathrm{Pa}$")
datareducers["pperpendicular"] =         DataReducerVariable(["ptensorrotated"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_\perp$", latexunits=r"$\mathrm{Pa}$")
datareducers["pperpoverpar"] =           DataReducerVariable(["ptensorrotated"], Anisotropy, "", 1, latex=r"$P_\perp P_\parallel^{-1}$", latexunits=r"")
datareducers["panisotropy"] =           DataReducerVariable(["ptensorrotated"], Anisotropy, "", 1, latex=r"$P_\perp P_\parallel^{-1}$", latexunits=r"")
datareducers["gyrotropy"] =             DataReducerVariable(["ptensordiagonal","ptensoroffdiagonal","b"], gyrotropy, "", 1, latex=r"$Q$", latexunits=r"")

datareducers["pbackstream"] =                 DataReducerVariable(["ptensorbackstreamdiagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{st}$", latexunits=r"$\mathrm{Pa}$")
datareducers["ptensorbackstream"] =           DataReducerVariable(["ptensorbackstreamdiagonal", "ptensorbackstreamoffdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{st}$", latexunits=r"$\mathrm{Pa}$")
datareducers["ptensorrotatedbackstream"] =    DataReducerVariable(["ptensorbackstream", "b"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{st}^\mathrm{R}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pparallelbackstream"] =         DataReducerVariable(["ptensorrotatedbackstream"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{st}}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pperpendicularbackstream"] =    DataReducerVariable(["ptensorrotatedbackstream"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{st}}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pperpoverparbackstream"] =      DataReducerVariable(["ptensorrotatedbackstream"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{st}} P_{\parallel,\mathrm{st}}^{-1}$", latexunits=r"")
datareducers["gyrotropybackstream"] =        DataReducerVariable(["ptensorbackstreamdiagonal", "ptensorbackstreamoffdiagonal","b"], gyrotropy, "", 1, latex=r"$Q_\mathrm{st}$", latexunits=r"")

datareducers["pnonbackstream"] =              DataReducerVariable(["ptensornonbackstreamdiagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{th}$", latexunits=r"$\mathrm{Pa}$")
datareducers["ptensornonbackstream"] =        DataReducerVariable(["ptensornonbackstreamdiagonal", "ptensornonbackstreamoffdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{th}$", latexunits=r"$\mathrm{Pa}$")
datareducers["ptensorrotatednonbackstream"] = DataReducerVariable(["ptensornonbackstream", "b"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{th}^\mathrm{R}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pparallelnonbackstream"] =      DataReducerVariable(["ptensorrotatednonbackstream"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{th}}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pperpendicularnonbackstream"] = DataReducerVariable(["ptensorrotatednonbackstream"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{th}}$", latexunits=r"$\mathrm{Pa}$")
datareducers["pperpoverparnonbackstream"] =   DataReducerVariable(["ptensorrotatednonbackstream"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{th}} P_{\parallel,\mathrm{th}}^{-1}$", latexunits=r"")
datareducers["gyrotropynonbackstream"] =     DataReducerVariable(["ptensorbackstreamdiagonal", "ptensorbackstreamoffdiagonal","b"], gyrotropy, "", 1, latex=r"$Q_\mathrm{th}$", latexunits=r"")

# Note: Temperature summing over multipop works only if only one population exists in simulation.
# T=P/(n*kb), calculating  sum(T)=sum(P)/(sum(n)*kb) is incorrect
# test-populations contribute to rho but not pressure (unless pressure is summed from ptensors).
datareducers["temperature"] =            DataReducerVariable(["pressure", "rho"], Temperature, "K", 1, latex=r"$T$", latexunits=r"$\mathrm{K}$")
datareducers["ttensor"] =                DataReducerVariable(["ptensor", "rho"], Temperature, "K", 9, latex=r"$\mathcal{T}$", latexunits=r"$\mathrm{K}$")
datareducers["ttensorrotated"] =         DataReducerVariable(["ptensorrotated", "rho"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}$", latexunits=r"$\mathrm{K}$")
datareducers["tparallel"] =              DataReducerVariable(["pparallel", "rho"], Temperature, "K", 1, latex=r"$T_\parallel$", latexunits=r"$\mathrm{K}$")
datareducers["tperpendicular"] =         DataReducerVariable(["pperpendicular", "rho"], Temperature, "K", 1, latex=r"$T_\perp$", latexunits=r"$\mathrm{K}$")

datareducers["tbackstream"] =            DataReducerVariable(["pbackstream", "rhobackstream"], Temperature, "K", 1, latex=r"$T_\mathrm{st}$", latexunits=r"$\mathrm{K}$")
datareducers["ttensorbackstream"] =      DataReducerVariable(["ptensorbackstream", "rhobackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{st}$", latexunits=r"$\mathrm{K}$")
datareducers["ttensorrotatedbackstream"]=DataReducerVariable(["ptensorrotatedbackstream", "rhobackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{st}^\mathrm{R}$", latexunits=r"$\mathrm{K}$")
datareducers["tparallelbackstream"] =    DataReducerVariable(["pparallelbackstream", "rhobackstream"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{st}}$", latexunits=r"$\mathrm{K}$")
datareducers["tperpendicularbackstream"]=DataReducerVariable(["pperpendicularbackstream", "rhobackstream"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{st}}$", latexunits=r"$\mathrm{K}$")

datareducers["tnonbackstream"] =              DataReducerVariable(["pnonbackstream", "rhononbackstream"], Temperature, "K", 1, latex=r"$T_\mathrm{th}$", latexunits=r"$\mathrm{K}$")
datareducers["ttensornonbackstream"] =        DataReducerVariable(["ptensornonbackstream", "rhononbackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{th}$", latexunits=r"$\mathrm{K}$")
datareducers["ttensorrotatednonbackstream"] = DataReducerVariable(["ptensorrotatednonbackstream", "rhononbackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{th}^\mathrm{R}$", latexunits=r"$\mathrm{K}$")
datareducers["tparallelnonbackstream"] =      DataReducerVariable(["pparallelnonbackstream", "rhononbackstream"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{th}}$", latexunits=r"$\mathrm{K}$")
datareducers["tperpendicularnonbackstream"] = DataReducerVariable(["pperpendicularnonbackstream", "rhononbackstream"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{th}}$", latexunits=r"$\mathrm{K}$")

# These ratios are identical to the pressure ratios
datareducers["tperpoverpar"] =                DataReducerVariable(["ptensorrotated"], Anisotropy, "", 1, latex=r"$T_\perp T_\parallel^{-1}$", latexunits=r"")
datareducers["tperpoverparbackstream"] =      DataReducerVariable(["ptensorrotatedbackstream"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{st}} T_{\parallel,\mathrm{st}}^{-1}$", latexunits=r"")
datareducers["tperpoverparnonbackstream"] =   DataReducerVariable(["ptensorrotatednonbackstream"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{th}} T_{\parallel,\mathrm{th}}^{-1}$", latexunits=r"")

# These ratios are identical to the pressure ratiosd
datareducers["betaperpoverpar"] =             DataReducerVariable(["ptensorrotated"], Anisotropy, "", 1, latex=r"$\beta_\perp \beta_\parallel^{-1}$", latexunits=r"")
datareducers["betaperpoverparbackstream"] =   DataReducerVariable(["ptensorrotatedbackstream"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{st}} \beta_{\parallel,\mathrm{st}}^{-1}$", latexunits=r"")
datareducers["betaperpoverparnonbackstream"] =DataReducerVariable(["ptensorrotatednonbackstream"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{th}} \beta_{\parallel,\mathrm{th}}^{-1}$", latexunits=r"")

datareducers["beta"] =                   DataReducerVariable(["pressure", "b"], beta ,"", 1, latex=r"$\beta$", latexunits=r"")
datareducers["betaparallel"] =           DataReducerVariable(["pparallel", "b"], beta ,"", 1, latex=r"$\beta_\parallel$", latexunits=r"")
datareducers["betaperpendicular"] =      DataReducerVariable(["pperpendicular", "b"], beta ,"", 1, latex=r"$\beta_\perp$", latexunits=r"")
datareducers["beta_star"] =               DataReducerVariable(["pressure", "pdyn", "b"], beta_star ,"", 1, latex=r"$\beta^\star$", latexunits=r"")


datareducers["rmirror"] =                DataReducerVariable(["ptensor", "b"], rMirror, "", 1, latex=r"$R_\mathrm{m}$")
datareducers["dng"] =                    DataReducerVariable(["ptensor", "pparallel", "pperpendicular", "b"], Dng, "", 1, latex=r"$\mathrm{Dng}$") # I think this has vector length 1?
datareducers["vbeam"] =                  DataReducerVariable(["vbackstream", "vnonbackstream"], v_beam, "m/s", 3, latex=r"$V_\mathrm{st}-V$", latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["vbeamratio"] =             DataReducerVariable(["vbackstream", "vnonbackstream"], v_beam_ratio, "", 1, latex=r"$V_\mathrm{st} V^{-1}$", latexunits=r"")
datareducers["thermalvelocity"] =               DataReducerVariable(["temperature"], thermalvelocity, "m/s", 1, latex=r"$v_\mathrm{th}$", latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
datareducers["larmor"] =              DataReducerVariable(["b","thermalvelocity"], larmor, "m", 1, latex=r"$r_\mathrm{L}$",latexunits=r"$\mathrm{m}$")
datareducers["plasmaperiod"] =    DataReducerVariable(["rho"], plasmaperiod, "s", 1, latex=r"$2\pi \Omega_{\mathrm{pi}}^{-1}$",latexunits=r"$\mathrm{s}$")
datareducers["di"] =              DataReducerVariable(["rho"], ion_inertial, "m", 1, latex=r"$d_\mathrm{i}$",latexunits=r"$\mathrm{m}$")
datareducers["bz_linedipole_avg"] =      DataReducerVariable(["x", "y", "z", "dx", "dy", "dz"], Bz_linedipole_avg, "T", 1, latex=r"$\langle B_{z,\mathrm{ld}}\rangle$")
datareducers["bz_linedipole_diff"] =     DataReducerVariable(["b", "bz_linedipole_avg"], Bz_linedipole_diff, "", 1, latex=r"$\Delta B_{z,\mathrm{ld}}$")

#reducers with useVspace
datareducers["gyrophase_relstddev"] =    DataReducerVariable(["v", "b"], gyrophase_relstddev, "", 1, useVspace=True) # I think this has vector length 1?





#multipopdatareducers
multipopdatareducers = {}
multipopdatareducers["pop/rhom"] =                   DataReducerVariable(["pop/rho"], rhom, "kg/m3", 1, latex=r"$\rho_{m,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{kg}\,\mathrm{m}^{-3}$")
multipopdatareducers["pop/rhoq"] =                   DataReducerVariable(["pop/rho"], rhoq, "C/m3", 1, latex=r"$\rho_{q,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{C}\,\mathrm{m}^{-3}$")

multipopdatareducers["pop/pdyn"] =            DataReducerVariable(["pop/v", "pop/rhom"], Pdyn, "Pa", 1, latex=r"$P_\mathrm{dyn,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pdynx"] =            DataReducerVariable(["pop/v", "pop/rhom"], Pdynx, "Pa", 1, latex=r"$P_\mathrm{dyn,\mathrm{REPLACEPOP},x}$",latexunits=r"$\mathrm{Pa}$")

multipopdatareducers["pop/vparallel"] =              DataReducerVariable(["pop/v", "b"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopdatareducers["pop/vperpendicular"] =         DataReducerVariable(["pop/v", "b"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")

multipopdatareducers["pop/vparallelbackstream"] =         DataReducerVariable(["pop/vbackstream", "b"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{st},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopdatareducers["pop/vparallelnonbackstream"] =         DataReducerVariable(["pop/vnonbackstream", "b"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{th},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopdatareducers["pop/vperpendicularbackstream"] =         DataReducerVariable(["pop/vbackstream", "b"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{st},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopdatareducers["pop/vperpendicularnonbackstream"] =         DataReducerVariable(["pop/vnonbackstream", "b"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{th},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")


multipopdatareducers["pop/pressure"] =               DataReducerVariable(["pop/ptensordiagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/ptensor"] =                DataReducerVariable(["pop/ptensordiagonal", "pop/ptensoroffdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/ptensorrotated"] =         DataReducerVariable(["pop/ptensor", "b"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pparallel"] =              DataReducerVariable(["pop/ptensorrotated"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pperpendicular"] =         DataReducerVariable(["pop/ptensorrotated"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pperpoverpar"] =           DataReducerVariable(["pop/ptensorrotated"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP}} P_{\parallel,\mathrm{REPLACEPOP}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/gyrotropy"] =              DataReducerVariable(["pop/ptensordiagonal", "pop/ptensoroffdiagonal","b"], gyrotropy, "", 1, latex=r"$Q_\mathrm{REPLACEPOP}$", latexunits=r"")

multipopdatareducers["pop/pbackstream"] =            DataReducerVariable(["pop/ptensorbackstreamdiagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/ptensorbackstream"] =      DataReducerVariable(["pop/ptensorbackstreamdiagonal", "pop/ptensorbackstreamoffdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/ptensorrotatedbackstream"]=DataReducerVariable(["pop/ptensorbackstream", "b"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pparallelbackstream"] =    DataReducerVariable(["pop/ptensorrotatedbackstream"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pperpendicularbackstream"]=DataReducerVariable(["pop/ptensorrotatedbackstream"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pperpoverparbackstream"] = DataReducerVariable(["pop/ptensorrotatedbackstream"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,st}} P_{\parallel,\mathrm{REPLACEPOP,st}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/gyrotropybackstream"] =   DataReducerVariable(["pop/ptensorrotatedbackstream"], gyrotropy, "", 1, latex=r"$Q_\mathrm{REPLACEPOP,st}$", latexunits=r"")

multipopdatareducers["pop/pnonbackstream"] =              DataReducerVariable(["pop/ptensornonbackstreamdiagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/ptensornonbackstream"] =        DataReducerVariable(["pop/ptensornonbackstreamdiagonal", "pop/ptensornonbackstreamoffdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/ptensorrotatednonbackstream"] = DataReducerVariable(["pop/ptensornonbackstream", "b"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pparallelnonbackstream"] =      DataReducerVariable(["pop/ptensorrotatednonbackstream"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pperpendicularnonbackstream"] = DataReducerVariable(["pop/ptensorrotatednonbackstream"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{Pa}$")
multipopdatareducers["pop/pperpoverparnonbackstream"] =   DataReducerVariable(["pop/ptensorrotatednonbackstream"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,th}} P_{\parallel,\mathrm{REPLACEPOP,th}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/gyrotropynonbackstream"] =     DataReducerVariable(["pop/ptensornonbackstreamdiagonal", "pop/ptensornonbackstreamoffdiagonal","b"], gyrotropy, "", 1, latex=r"$Q_\mathrm{REPLACEPOP,th}$", latexunits=r"")

multipopdatareducers["pop/temperature"] =            DataReducerVariable(["pop/pressure", "pop/rho"], Temperature, "K", 1, latex=r"$T_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/ttensor"] =                DataReducerVariable(["pop/ptensor", "pop/rho"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/ttensorrotated"] =         DataReducerVariable(["pop/ptensorrotated", "pop/rho"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/tparallel"] =              DataReducerVariable(["pop/pparallel", "pop/rho"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/tperpendicular"] =         DataReducerVariable(["pop/pperpendicular", "pop/rho"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{K}$")

multipopdatareducers["pop/tbackstream"] =            DataReducerVariable(["pop/pbackstream", "pop/rhobackstream"], Temperature, "K", 1, latex=r"$T_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/ttensorbackstream"] =      DataReducerVariable(["pop/ptensorbackstream", "pop/rhobackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/ttensorrotatedbackstream"]=DataReducerVariable(["pop/ptensorrotatedbackstream", "pop/rhobackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/tparallelbackstream"] =    DataReducerVariable(["pop/pparallelbackstream", "pop/rhobackstream"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/tperpendicularbackstream"]=DataReducerVariable(["pop/pperpendicularbackstream", "pop/rhobackstream"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{K}$")

multipopdatareducers["pop/tnonbackstream"] =              DataReducerVariable(["pop/pnonbackstream", "pop/rhononbackstream"], Temperature, "K", 1, latex=r"$T_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/ttensornonbackstream"] =        DataReducerVariable(["pop/ptensornonbackstream", "pop/rhononbackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/ttensorrotatednonbackstream"] = DataReducerVariable(["pop/ptensorrotatednonbackstream", "pop/rhononbackstream"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/tparallelnonbackstream"] =      DataReducerVariable(["pop/pparallelnonbackstream", "pop/rhononbackstream"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{K}$")
multipopdatareducers["pop/tperpendicularnonbackstream"] = DataReducerVariable(["pop/pperpendicularnonbackstream", "pop/rhononbackstream"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{K}$")

# These ratios are identical to the pressure ratios
multipopdatareducers["pop/tperpoverpar"] =                 DataReducerVariable(["pop/ptensorrotated"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP}} T_{\parallel,\mathrm{REPLACEPOP}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/tperpoverparbackstream"] =       DataReducerVariable(["pop/ptensorrotatedbackstream"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,st}} T_{\parallel,\mathrm{REPLACEPOP,st}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/tperpoverparnonbackstream"] =    DataReducerVariable(["pop/ptensorrotatednonbackstream"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,th}} T_{\parallel,\mathrm{REPLACEPOP,th}}^{-1}$", latexunits=r"")

# These ratios are identical to the pressure ratios
multipopdatareducers["pop/betaperpoverpar"] =              DataReducerVariable(["pop/ptensorrotated"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP}} \beta_{\parallel,\mathrm{REPLACEPOP}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/betaperpoverparbackstream"] =    DataReducerVariable(["pop/ptensorrotatedbackstream"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP,st}} \beta_{\parallel,\mathrm{REPLACEPOP,st}}^{-1}$", latexunits=r"")
multipopdatareducers["pop/betaperpoverparnonbackstream"] = DataReducerVariable(["pop/ptensorrotatednonbackstream"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP,th}} \beta_{\parallel,\mathrm{REPLACEPOP,th}}^{-1}$", latexunits=r"")

multipopdatareducers["pop/thermalvelocity"] =               DataReducerVariable(["pop/temperature"], thermalvelocity, "m/s", 1, latex=r"$v_\mathrm{th,REPLACEPOP}$", latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopdatareducers["pop/larmor"] =              DataReducerVariable(["b","pop/thermalvelocity"], larmor, "m", 1, latex=r"$r_\mathrm{L,REPLACEPOP}$",latexunits=r"$\mathrm{m}$")

multipopdatareducers["pop/firstadiabatic"] =    DataReducerVariable(["pop/tperpendicular","b"], firstadiabatic, "K/T", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP}} B^{-1}$",latexunits=r"$\mathrm{K}\,\mathrm{T}^{-1}$")
multipopdatareducers["pop/gyroperiod"] =    DataReducerVariable(["b"], gyroperiod, "s", 1, latex=r"$2\pi \Omega_{\mathrm{c},\mathrm{REPLACEPOP}}^{-1}$",latexunits=r"$\mathrm{s}$")
multipopdatareducers["pop/plasmaperiod"] =    DataReducerVariable(["pop/rho"], plasmaperiod, "s", 1, latex=r"$2\pi \Omega_{\mathrm{p},\mathrm{REPLACEPOP}}^{-1}$",latexunits=r"$\mathrm{s}$")

# Do these betas make sense per-population?
multipopdatareducers["pop/beta"] =                   DataReducerVariable(["pop/pressure", "b"], beta ,"", 1, latex=r"$\beta_\mathrm{REPLACEPOP}$", latexunits=r"")
multipopdatareducers["pop/betaparallel"] =           DataReducerVariable(["pop/pparallel", "b"], beta ,"", 1, latex=r"$\beta_{\parallel,\mathrm{REPLACEPOP}}$", latexunits=r"")
multipopdatareducers["pop/betaperpendicular"] =      DataReducerVariable(["pop/pperpendicular", "b"], beta ,"", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP}}$", latexunits=r"")

multipopdatareducers["pop/rmirror"] =                DataReducerVariable(["pop/ptensor", "b"], rMirror, "", 1, latex=r"$R_\mathrm{m,REPLACEPOP}$")
multipopdatareducers["pop/dng"] =                    DataReducerVariable(["pop/ptensor", "pop/pparallel", "pop/pperpendicular", "b"], Dng, "", 1, latex=r"$\mathrm{Dng}_\mathrm{REPLACEPOP}$")


##########################################
#  v5reducers for use with Vlasiator5 data
##########################################

v5reducers = {}

# IONOSPHERE ('ig_')
v5reducers["ig_inplanecurrent"] = DataReducerVariable(["ig_e"], ig_inplanecurrent, "A/m", 1, latex=r"$\vec{J}$",latexunits=r"$\mathrm{A}\,\mathrm{m}^{-1}$", useReader=True)
v5reducers["ig_e"] = DataReducerVariable(["ig_potential"], ig_E, "V/m", 1, latex=r"$\vec{E}$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$", useReader=True)

# MAGNETOSPHERE ('vg_')
v5reducers["vg_vms"] =                    DataReducerVariable(["vg_pressure", "vg_rhom", "vg_b_vol"], vms, "m/s", 1, latex=r"$v_\mathrm{ms}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
v5reducers["vg_vs"] =                     DataReducerVariable(["vg_pressure", "vg_rhom"], vs, "m/s", 1, latex=r"$v_\mathrm{s}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
v5reducers["vg_va"] =                     DataReducerVariable(["vg_rhom", "vg_b_vol"], va, "m/s", 1, latex=r"$v_\mathrm{A}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
v5reducers["vg_ma"] =                     DataReducerVariable(["vg_v", "vg_va"], MA, "", 1, latex=r"$M_\mathrm{A}$",latexunits=r"")
v5reducers["vg_mms"] =                    DataReducerVariable(["vg_v", "vg_vms"], Mms, "", 1, latex=r"$M_\mathrm{ms}$",latexunits=r"")

v5reducers["vg_v_parallel"] =              DataReducerVariable(["vg_v", "vg_b_vol"], ParallelVectorComponent, "m/s", 1, latex=r"$V_\parallel$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
v5reducers["vg_v_perpendicular"] =         DataReducerVariable(["vg_v", "vg_b_vol"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_\perp$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")

v5reducers["vg_e_vol"] =              DataReducerVariable([], vg_e_vol, "V/m", 1, latex=r"$E_\mathrm{vol,vg}$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$",useReader=True)

v5reducers["vg_e_parallel"] =              DataReducerVariable(["vg_e_vol", "vg_b_vol"], ParallelVectorComponent, "V/m", 1, latex=r"$E_\parallel$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
v5reducers["vg_e_perpendicular"] =         DataReducerVariable(["vg_e_vol", "vg_b_vol"], PerpendicularVectorComponent, "V/m", 1, latex=r"$E_\perp$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
v5reducers["vg_poynting"] = DataReducerVariable(["vg_e_vol", "vg_b_vol"], Poynting, "W/m2", 3, latex=r"$S$", latexunits=r"\mathrm{W}\,\mathrm{m}^{-2}$")

v5reducers["vg_eje_parallel"] =              DataReducerVariable(["vg_eje", "vg_b_vol"], ParallelVectorComponent, "V/m", 1, latex=r"$EJE_\parallel$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
v5reducers["vg_eje_perpendicular"] =         DataReducerVariable(["vg_eje", "vg_b_vol"], PerpendicularVectorComponent, "V/m", 1, latex=r"$EJE_\perp$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
v5reducers["vg_egradpe_parallel"] =              DataReducerVariable(["vg_e_gradpe", "vg_b_vol"], ParallelVectorComponent, "V/m", 1, latex=r"$E_{\nabla Pe,\parallel}$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")
v5reducers["vg_egradpe_perpendicular"] =         DataReducerVariable(["vg_e_gradpe", "vg_b_vol"], PerpendicularVectorComponent, "V/m", 1, latex=r"$E_{\nabla Pe,\perp}$",latexunits=r"$\mathrm{V}\,\mathrm{m}^{-1}$")

v5reducers["vg_pdyn"] =            DataReducerVariable(["vg_v", "vg_rhom"], Pdyn, "Pa", 1, latex=r"$P_\mathrm{dyn}$",latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_pdynx"] =            DataReducerVariable(["vg_v", "vg_rhom"], Pdynx, "Pa", 1, latex=r"$P_\mathrm{dyn,x}$",latexunits=r"$\mathrm{Pa}$")

v5reducers["vg_p_magnetic"] =            DataReducerVariable(["vg_b_vol"], MagneticPressure, "Pa", 1, latex=r"$P_\mathrm{mag}$",latexunits=r"$\mathrm{Pa}$")

v5reducers["vg_di"] =              DataReducerVariable(["proton/vg_rho"], ion_inertial, "m", 1, latex=r"$d_\mathrm{i}$",latexunits=r"$\mathrm{m}$")

v5reducers["vg_pressure"] =               DataReducerVariable(["vg_ptensor_diagonal"], Pressure, "Pa", 1, latex=r"$P$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_ptensor"] =                DataReducerVariable(["vg_ptensor_diagonal", "vg_ptensor_offdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_ptensor_rotated"] =         DataReducerVariable(["vg_ptensor", "vg_b_vol"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_parallel"] =              DataReducerVariable(["vg_ptensor_rotated"], ParallelTensorComponent, "Pa", 1, latex=r"$P_\parallel$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_perpendicular"] =         DataReducerVariable(["vg_ptensor_rotated"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_\perp$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_anisotropy"] =           DataReducerVariable(["vg_ptensor_rotated"], Anisotropy, "", 1, latex=r"$P_\perp P_\parallel^{-1}$", latexunits=r"")
v5reducers["vg_gyrotropy"] =             DataReducerVariable(["vg_ptensor_diagonal", "vg_ptensor_offdiagonal","vg_b_vol"], gyrotropy, "", 1, latex=r"$Q$", latexunits=r"")

v5reducers["vg_p_nonthermal"] =                 DataReducerVariable(["vg_ptensor_nonthermal_diagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{st}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_ptensor_nonthermal"] =           DataReducerVariable(["vg_ptensor_nonthermal_diagonal", "vg_ptensor_nonthermal_offdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{st}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_ptensor_rotated_nonthermal"] =    DataReducerVariable(["vg_ptensor_nonthermal", "vg_b_vol"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{st}^\mathrm{R}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_parallel_nonthermal"] =         DataReducerVariable(["vg_ptensor_rotated_nonthermal"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{st}}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_perpendicular_nonthermal"] =    DataReducerVariable(["vg_ptensor_rotated_nonthermal"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{st}}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_anisotropy_nonthermal"] =       DataReducerVariable(["vg_ptensor_rotated_nonthermal"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{st}} P_{\parallel,\mathrm{st}}^{-1}$", latexunits=r"")
v5reducers["vg_gyrotropy_nonthermal"] =          DataReducerVariable(["vg_ptensor_nonthermal_diagonal", "vg_ptensor_nonthermal_offdiagonal","vg_b_vol"], gyrotropy, "", 1, latex=r"$Q_\mathrm{st}$", latexunits=r"")

v5reducers["vg_p_thermal"] =              DataReducerVariable(["vg_ptensor_thermal_diagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{th}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_ptensor_thermal"] =        DataReducerVariable(["vg_ptensor_thermal_diagonal", "vg_ptensor_thermal_offdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{th}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_ptensor_rotated_thermal"] = DataReducerVariable(["vg_ptensor_thermal", "vg_b_vol"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{th}^\mathrm{R}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_parallel_thermal"] =      DataReducerVariable(["vg_ptensor_rotated_thermal"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{th}}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_perpendicular_thermal"] = DataReducerVariable(["vg_ptensor_rotated_thermal"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{th}}$", latexunits=r"$\mathrm{Pa}$")
v5reducers["vg_p_anisotropy_thermal"] =   DataReducerVariable(["vg_ptensor_rotated_thermal"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{th}} P_{\parallel,\mathrm{th}}^{-1}$", latexunits=r"")
v5reducers["vg_gyrotropy_thermal"] =     DataReducerVariable(["vg_ptensor_thermal_diagonal", "vg_ptensor_thermal_offdiagonal","vg_b_vol"], gyrotropy, "", 1, latex=r"$Q_\mathrm{th}$", latexunits=r"")

# Note: Temperature summing over multipop works only if only one population exists in simulation.
# T=P/(n*kb), calculating  sum(T)=sum(P)/(sum(n)*kb) is incorrect
# test-populations contribute to rho but not vg_pressure. (unless vg_pressure is summed from vg_ptensors)
v5reducers["vg_temperature"] =            DataReducerVariable(["vg_pressure", "vg_rho"], Temperature, "K", 1, latex=r"$T$", latexunits=r"$\mathrm{K}$")
v5reducers["vg_ttensor"] =                DataReducerVariable(["vg_ptensor", "vg_rho"], Temperature, "K", 9, latex=r"$\mathcal{T}$", latexunits=r"$\mathrm{K}$")
v5reducers["vg_ttensor_rotated"] =         DataReducerVariable(["vg_ptensor_rotated", "vg_rho"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}$", latexunits=r"$\mathrm{K}$")
v5reducers["vg_t_parallel"] =              DataReducerVariable(["vg_p_parallel", "vg_rho"], Temperature, "K", 1, latex=r"$T_\parallel$", latexunits=r"$\mathrm{K}$")
v5reducers["vg_t_perpendicular"] =         DataReducerVariable(["vg_p_perpendicular", "vg_rho"], Temperature, "K", 1, latex=r"$T_\perp$", latexunits=r"$\mathrm{K}$")

 # These ratios are identical to the pressure ratios
v5reducers["vg_t_anisotropy"] =                DataReducerVariable(["vg_ptensor_rotated"], Anisotropy, "", 1, latex=r"$T_\perp T_\parallel^{-1}$", latexunits=r"")
v5reducers["vg_t_anisotropy_nonthermal"] =      DataReducerVariable(["vg_ptensor_rotated_nonthermal"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{st}} T_{\parallel,\mathrm{st}}^{-1}$", latexunits=r"")
v5reducers["vg_t_anisotropy_thermal"] =   DataReducerVariable(["vg_ptensor_rotated_thermal"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{th}} T_{\parallel,\mathrm{th}}^{-1}$", latexunits=r"")
 # These ratios are identical to the pressure ratios
v5reducers["vg_beta_anisotropy"] =             DataReducerVariable(["vg_ptensor_rotated"], Anisotropy, "", 1, latex=r"$\beta_\perp \beta_\parallel^{-1}$", latexunits=r"")
v5reducers["vg_beta_anisotropy_nonthermal"] =   DataReducerVariable(["vg_ptensor_rotated_nonthermal"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{st}} \beta_{\parallel,\mathrm{st}}^{-1}$", latexunits=r"")
v5reducers["vg_beta_anisotropy_thermal"] =DataReducerVariable(["vg_ptensor_rotated_thermal"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{th}} \beta_{\parallel,\mathrm{th}}^{-1}$", latexunits=r"")

v5reducers["vg_beta"] =                   DataReducerVariable(["vg_pressure", "vg_b_vol"], beta ,"", 1, latex=r"$\beta$", latexunits=r"")
v5reducers["vg_beta_parallel"] =           DataReducerVariable(["vg_p_parallel", "vg_b_vol"], beta ,"", 1, latex=r"$\beta_\parallel$", latexunits=r"")
v5reducers["vg_beta_perpendicular"] =      DataReducerVariable(["vg_p_perpendicular", "vg_b_vol"], beta ,"", 1, latex=r"$\beta_\perp$", latexunits=r"")
v5reducers["vg_beta_star"] =               DataReducerVariable(["vg_pressure", "vg_pdyn", "vg_b_vol"], beta_star ,"", 1, latex=r"$\beta^\star$", latexunits=r"")

v5reducers["vg_rmirror"] =                DataReducerVariable(["vg_ptensor", "vg_b_vol"], rMirror, "", 1, latex=r"$R_\mathrm{m}$")

v5reducers["vg_restart_v"] =              DataReducerVariable(["moments"], restart_V, "m/s", 3, latex=r"$V$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
v5reducers["vg_restart_rho"] =            DataReducerVariable(["moments"], restart_rho, "1/m3", 1, latex=r"$n_\mathrm{p}$",latexunits=r"$\mathrm{m}^{-3}$")
v5reducers["vg_restart_rhom"] =           DataReducerVariable(["moments"], restart_rhom, "kg/m3", 1, latex=r"$\rho_m$",latexunits=r"$\mathrm{kg}\,\mathrm{m}^{-3}$")
v5reducers["vg_restart_rhoq"] =           DataReducerVariable(["moments"], restart_rhoq, "C/m3", 1, latex=r"$\rho_q$",latexunits=r"$\mathrm{C}\,\mathrm{m}^{-3}$")
v5reducers["vg_amr_jperb_criteria"] =           DataReducerVariable(["vg_amr_jperb"], JPerB_criteria, "", 1, latex=r"$\log_2 (J/B_{\perp} \cdot \Delta x_0)$",latexunits=r"1")

v5reducers["vg_coordinates"] =            DataReducerVariable(["CellID"], vg_coordinates_cellcenter, "m", 3, latex=r"$\vec{r}_\mathrm{cc}$", latexunits=r"$\mathrm{m}$", useReader=True)
v5reducers["vg_coordinates_cell_center"] =            DataReducerVariable(["CellID"], vg_coordinates_cellcenter, "m", 3, latex=r"$\vec{r}_\mathrm{cc}$", latexunits=r"$\mathrm{m}$", useReader=True)
v5reducers["vg_coordinates_cell_lowcorner"] =            DataReducerVariable(["CellID","vg_dx"], vg_coordinates_lowcorner, "m", 3, latex=r"$\vec{r}_\mathrm{cc}$", latexunits=r"$\mathrm{m}$", useReader=True)

v5reducers["vg_dx"] =            DataReducerVariable(["CellID"], vg_dx, "m", 3, latex=r"$\Delta{}\vec{r}$", latexunits=r"$\mathrm{m}$", useReader=True)
v5reducers["vg_dxs"] =            DataReducerVariable(["CellID"], vg_dx, "m", 3, latex=r"$\Delta{}\vec{r}$", latexunits=r"$\mathrm{m}$", useReader=True)
v5reducers["vg_reflevel"] =            DataReducerVariable(["CellID"], vg_reflevel, "", 1, latex=r"reflevel", latexunits=r"", useReader=True)

v5reducers["vg_jacobian_b"] =             DataReducerVariable(["vg_dbxvoldx","vg_dbxvoldy","vg_dbxvoldz","vg_dbyvoldx","vg_dbyvoldy","vg_dbyvoldz","vg_dbzvoldx","vg_dbzvoldy","vg_dbzvoldz"], TensorFromScalars, "T/m", 9, latex=r"$\vec{J}$",latexunits=r"$\mathrm{A}\,\mathrm{m}^{-2}$")
v5reducers["vg_jacobian_bper"] =          DataReducerVariable(["vg_dperbxvoldx","vg_dperbxvoldy","vg_dperbxvoldz","vg_dperbyvoldx","vg_dperbyvoldy","vg_dperbyvoldz","vg_dperbzvoldx","vg_dperbzvoldy","vg_dperbzvoldz"], TensorFromScalars, "T/m", 9, latex=r"$\vec{J}$",latexunits=r"$\mathrm{A}\,\mathrm{m}^{-2}$")
v5reducers["vg_j"] =                     DataReducerVariable(["vg_jacobian_bper"], J, "A/m^2", 3, latex=r"$\vec{J}$",latexunits=r"$\mathrm{A}\,\mathrm{m}^{-2}$")

#multipopv5reducers
multipopv5reducers = {}
multipopv5reducers["pop/vg_rhom"] =                   DataReducerVariable(["pop/vg_rho"], rhom, "kg/m3", 1, latex=r"$\rho_{m,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{kg}\,\mathrm{m}^{-3}$")
multipopv5reducers["pop/vg_rhoq"] =                   DataReducerVariable(["pop/vg_rho"], rhoq, "C/m3", 1, latex=r"$\rho_{q,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{C}\,\mathrm{m}^{-3}$")
multipopv5reducers["pop/vg_pdyn"] =            DataReducerVariable(["pop/vg_v", "pop/vg_rhom"], Pdyn, "Pa", 1, latex=r"$P_\mathrm{dyn,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_pdynx"] =            DataReducerVariable(["pop/vg_v", "pop/vg_rhom"], Pdynx, "Pa", 1, latex=r"$P_\mathrm{dyn,\mathrm{REPLACEPOP},x}$",latexunits=r"$\mathrm{Pa}$")

multipopv5reducers["pop/vg_v_parallel"] =              DataReducerVariable(["pop/vg_v", "vg_b_vol"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopv5reducers["pop/vg_v_perpendicular"] =         DataReducerVariable(["pop/vg_v", "vg_b_vol"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")

multipopv5reducers["pop/vg_v_parallel_thermal"] =              DataReducerVariable(["pop/vg_v_thermal", "vg_b_vol"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{th},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopv5reducers["pop/vg_v_perpendicular_thermal"] =         DataReducerVariable(["pop/vg_v_thermal", "vg_b_vol"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{th},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopv5reducers["pop/vg_v_parallel_nonthermal"] =              DataReducerVariable(["pop/vg_v_nonthermal", "vg_b_vol"], ParallelVectorComponent, "m/s", 1, latex=r"$V_{\parallel,\mathrm{st},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopv5reducers["pop/vg_v_perpendicular_nonthermal"] =         DataReducerVariable(["pop/vg_v_nonthermal", "vg_b_vol"], PerpendicularVectorComponent, "m/s", 1, latex=r"$V_{\perp,\mathrm{st},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")

multipopv5reducers["pop/vg_pressure"] =               DataReducerVariable(["pop/vg_ptensor_diagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_ptensor"] =                DataReducerVariable(["pop/vg_ptensor_diagonal", "pop/vg_ptensor_offdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_ptensor_rotated"] =         DataReducerVariable(["pop/vg_ptensor", "vg_b_vol"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_parallel"] =              DataReducerVariable(["pop/vg_ptensor_rotated"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_perpendicular"] =         DataReducerVariable(["pop/vg_ptensor_rotated"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_anisotropy"] =           DataReducerVariable(["pop/vg_ptensor_rotated"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP}} P_{\parallel,\mathrm{REPLACEPOP}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_gyrotropy"] =              DataReducerVariable(["pop/vg_ptensor_diagonal", "pop/vg_ptensor_offdiagonal","vg_b_vol"], gyrotropy, "", 1, latex=r"$Q_\mathrm{REPLACEPOP}$", latexunits=r"")


multipopv5reducers["pop/vg_p_nonthermal"] =            DataReducerVariable(["pop/vg_ptensor_nonthermal_diagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_ptensor_nonthermal"] =      DataReducerVariable(["pop/vg_ptensor_nonthermal_diagonal", "pop/vg_ptensor_nonthermal_offdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_ptensor_rotated_nonthermal"]=DataReducerVariable(["pop/vg_ptensor_nonthermal", "vg_b_vol"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_parallel_nonthermal"] =    DataReducerVariable(["pop/vg_ptensor_rotated_nonthermal"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_perpendicular_nonthermal"]=DataReducerVariable(["pop/vg_ptensor_rotated_nonthermal"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_anisotropy_nonthermal"] = DataReducerVariable(["pop/vg_ptensor_rotated_nonthermal"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,st}} P_{\parallel,\mathrm{REPLACEPOP,st}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_gyrotropy_nonthermal"] =    DataReducerVariable(["pop/vg_ptensor_nonthermal_diagonal", "pop/vg_ptensor_nonthermal_offdiagonal","vg_b_vol"], gyrotropy, "", 1, latex=r"$Q_\mathrm{st}$", latexunits=r"")

multipopv5reducers["pop/vg_p_thermal"] =              DataReducerVariable(["pop/vg_ptensor_thermal_diagonal"], Pressure, "Pa", 1, latex=r"$P_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_ptensor_thermal"] =        DataReducerVariable(["pop/vg_ptensor_thermal_diagonal", "pop/vg_ptensor_thermal_offdiagonal"], FullTensor, "Pa", 9, latex=r"$\mathcal{P}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_ptensor_rotated_thermal"] = DataReducerVariable(["pop/vg_ptensor_thermal", "vg_b_vol"], RotatedTensor, "Pa", 9, latex=r"$\mathcal{P}^\mathrm{R}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_parallel_thermal"] =      DataReducerVariable(["pop/vg_ptensor_rotated_thermal"], ParallelTensorComponent, "Pa", 1, latex=r"$P_{\parallel,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_perpendicular_thermal"] = DataReducerVariable(["pop/vg_ptensor_rotated_thermal"], PerpendicularTensorComponent, "Pa", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{Pa}$")
multipopv5reducers["pop/vg_p_anisotropy_thermal"] =   DataReducerVariable(["pop/vg_ptensor_rotated_thermal"], Anisotropy, "", 1, latex=r"$P_{\perp,\mathrm{REPLACEPOP,th}} P_{\parallel,\mathrm{REPLACEPOP,th}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_gyrotropy_thermal"] =     DataReducerVariable(["pop/vg_ptensor_thermal_diagonal", "pop/vg_ptensor_thermal_offdiagonal","vg_b_vol"], gyrotropy, "", 1, latex=r"$Q_\mathrm{REPLACEPOP,th}$", latexunits=r"")

multipopv5reducers["pop/vg_temperature"] =            DataReducerVariable(["pop/vg_pressure", "pop/vg_rho"], Temperature, "K", 1, latex=r"$T_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_ttensor"] =                DataReducerVariable(["pop/vg_ptensor", "pop/vg_rho"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_ttensor_rotated"] =         DataReducerVariable(["pop/vg_ptensor_rotated", "pop/vg_rho"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}_\mathrm{REPLACEPOP}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_t_parallel"] =              DataReducerVariable(["pop/vg_p_parallel", "pop/vg_rho"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_t_perpendicular"] =         DataReducerVariable(["pop/vg_p_perpendicular", "pop/vg_rho"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP}}$", latexunits=r"$\mathrm{K}$")

multipopv5reducers["pop/vg_t_nonthermal"] =            DataReducerVariable(["pop/vg_p_nonthermal", "pop/vg_rho_nonthermal"], Temperature, "K", 1, latex=r"$T_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_ttensor_nonthermal"] =      DataReducerVariable(["pop/vg_ptensor_nonthermal", "pop/vg_rho_nonthermal"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_ttensor_rotated_nonthermal"]=DataReducerVariable(["pop/vg_ptensor_rotated_nonthermal", "pop/vg_rho_nonthermal"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}_\mathrm{REPLACEPOP,st}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_t_parallel_nonthermal"] =    DataReducerVariable(["pop/vg_p_parallel_nonthermal", "pop/vg_rho_nonthermal"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_t_perpendicular_nonthermal"]=DataReducerVariable(["pop/vg_p_perpendicular_nonthermal", "pop/vg_rho_nonthermal"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,st}}$", latexunits=r"$\mathrm{K}$")

multipopv5reducers["pop/vg_t_thermal"] =              DataReducerVariable(["pop/vg_p_thermal", "pop/vg_rho_thermal"], Temperature, "K", 1, latex=r"$T_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_ttensor_thermal"] =        DataReducerVariable(["pop/vg_ptensor_thermal", "pop/vg_rho_thermal"], Temperature, "K", 9, latex=r"$\mathcal{T}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_ttensor_rotated_thermal"] = DataReducerVariable(["pop/vg_ptensor_rotated_thermal", "pop/vg_rho_thermal"], Temperature, "K", 9, latex=r"$\mathcal{T}^\mathrm{R}_\mathrm{REPLACEPOP,th}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_t_parallel_thermal"] =      DataReducerVariable(["pop/vg_p_parallel_thermal", "pop/vg_rho_thermal"], Temperature, "K", 1, latex=r"$T_{\parallel,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{K}$")
multipopv5reducers["pop/vg_t_perpendicular_thermal"] = DataReducerVariable(["pop/vg_p_perpendicular_thermal", "pop/vg_rho_thermal"], Temperature, "K", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,th}}$", latexunits=r"$\mathrm{K}$")

 # These ratios are identical to the pressure ratios
multipopv5reducers["pop/vg_t_anisotropy"] =                 DataReducerVariable(["pop/vg_ptensor_rotated"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP}} T_{\parallel,\mathrm{REPLACEPOP}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_t_anisotropy_nonthermal"] =       DataReducerVariable(["pop/vg_ptensor_rotated_nonthermal"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,st}} T_{\parallel,\mathrm{REPLACEPOP,st}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_t_anisotropy_thermal"] =    DataReducerVariable(["pop/vg_ptensor_rotated_thermal"], Anisotropy, "", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP,th}} T_{\parallel,\mathrm{REPLACEPOP,th}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_beta_anisotropy"] =              DataReducerVariable(["pop/vg_ptensor_rotated"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP}} \beta_{\parallel,\mathrm{REPLACEPOP}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_beta_anisotropy_nonthermal"] =    DataReducerVariable(["pop/vg_ptensor_rotated_nonthermal"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP,st}} \beta_{\parallel,\mathrm{REPLACEPOP,st}}^{-1}$", latexunits=r"")
multipopv5reducers["pop/vg_beta_anisotropy_thermal"] = DataReducerVariable(["pop/vg_ptensor_rotated_thermal"], Anisotropy, "", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP,th}} \beta_{\parallel,\mathrm{REPLACEPOP,th}}^{-1}$", latexunits=r"")

multipopv5reducers["pop/vg_thermalvelocity"] =               DataReducerVariable(["pop/vg_temperature"], thermalvelocity, "m/s", 1, latex=r"$v_\mathrm{th,REPLACEPOP}$", latexunits=r"$\mathrm{m}\,\mathrm{s}^{-1}$")
multipopv5reducers["pop/vg_larmor"] =                        DataReducerVariable(["vg_b_vol","pop/vg_thermalvelocity"], larmor, "m", 1, latex=r"$r_\mathrm{L,REPLACEPOP}$",latexunits=r"$\mathrm{m}$")

multipopv5reducers["pop/vg_firstadiabatic"] =    DataReducerVariable(["pop/vg_t_perpendicular","vg_b_vol"], firstadiabatic, "K/T", 1, latex=r"$T_{\perp,\mathrm{REPLACEPOP}} B^{-1}$",latexunits=r"$\mathrm{K}\,\mathrm{T}^{-1}$")
multipopv5reducers["pop/vg_gyroperiod"] =    DataReducerVariable(["vg_b_vol","pop/vg_rho"], gyroperiod, "s", 1, latex=r"$2\pi \Omega_{\mathrm{c},\mathrm{REPLACEPOP}}^{-1}$",latexunits=r"$\mathrm{s}$")
multipopv5reducers["pop/vg_plasmaperiod"] =    DataReducerVariable(["pop/vg_rho"], plasmaperiod, "s", 1, latex=r"$2\pi \Omega_{\mathrm{p},\mathrm{REPLACEPOP}}^{-1}$",latexunits=r"$\mathrm{s}$")
multipopv5reducers["pop/vg_precipitationintegralenergyflux"] = DataReducerVariable(["pop/vg_precipitationdifferentialflux"],precipitationintegralenergyflux, "keV/(cm2 s sr)", 1, latex=r"$\int \mathcal{F}_{\mathrm{prec},\mathrm{REPLACEPOP}}$",latexunits=r"$\mathrm{keV}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}$")
multipopv5reducers["pop/vg_precipitationmeanenergy"] = DataReducerVariable(["pop/vg_precipitationdifferentialflux"],precipitationmeanenergy, "keV", 1, latex=r"$<E_{\mathrm{prec},\mathrm{REPLACEPOP}}>$",latexunits=r"$\mathrm{keV}$")

# Do these betas make sense per-population?
multipopv5reducers["pop/vg_beta"] =                   DataReducerVariable(["pop/vg_pressure", "vg_b_vol"], beta ,"", 1, latex=r"$\beta_\mathrm{REPLACEPOP}$", latexunits=r"")
multipopv5reducers["pop/vg_beta_parallel"] =           DataReducerVariable(["pop/vg_p_parallel", "vg_b_vol"], beta ,"", 1, latex=r"$\beta_{\parallel,\mathrm{REPLACEPOP}}$", latexunits=r"")
multipopv5reducers["pop/vg_beta_perpendicular"] =      DataReducerVariable(["pop/vg_p_perpendicular", "vg_b_vol"], beta ,"", 1, latex=r"$\beta_{\perp,\mathrm{REPLACEPOP}}$", latexunits=r"")

multipopv5reducers["pop/vg_rmirror"] =                DataReducerVariable(["pop/vg_ptensor", "vg_b_vol"], rMirror, "", 1, latex=r"$R_\mathrm{m,REPLACEPOP}$")
multipopv5reducers["pop/vg_dng"] =                    DataReducerVariable(["pop/vg_ptensor", "pop/vg_p_parallel", "pop/vg_p_perpendicular", "vg_b_vol"], Dng, "", 1, latex=r"$\mathrm{Dng}_\mathrm{REPLACEPOP}$")

# The dictionary with deprecated data reducers
deprecated_datareducers = {}
deprecated_datareducers['agyrotropy'] = "Previous agyrotropy reducers were broken, use gyrotropy instead. See link (https://github.com/fmihpc/analysator/pull/262) "
deprecated_datareducers['agyrotropybackstream'] = "Previous agyrotropybackstream reducers were broken, use gyrotropybackstream instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['agyrotropynonbackstream'] = "Previous agyrotropynonbackstream reducers were broken, use gyrotropynonbackstream instead. See link (https://github.com/fmihpc/analysator/pull/262)"

deprecated_datareducers['pop/agyrotropy'] = "Previous pop/agyrotropy reducers were broken, use pop/gyrotropy instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['pop/agyrotropybackstream'] = "Previous pop/agyrotropybackstream reducers were broken, use pop/gyrotropybackstream instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['pop/agyrotropynonbackstream'] = "Previous pop/agyrotropynonbackstream reducers were broken, use pop/gyrotropynonbackstream instead. See link (https://github.com/fmihpc/analysator/pull/262)"

deprecated_datareducers['vg_agyrotropy'] = "Previous vg_agyrotropy reducers were broken, use vg_gyrotropy instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['vg_agyrotropy_nonthermal'] = "Previous vg_agyrotropy_nonthermal reducers were broken, use vg_gyrotropy_nonthermal instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['vg_agyrotropy_thermal'] = "Previous vg_agyrotropy_thermal reducers were broken, use vg_gyrotropy_thermal instead. See link (https://github.com/fmihpc/analysator/pull/262)"

deprecated_datareducers['pop/vg_agyrotropy'] = "Previous pop/vg_agyrotropy reducers were broken, use pop/vg_gyrotropy instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['pop/vg_agyrotropy_nonthermal'] = "Previous pop/vg_agyrotropy_nonthermal reducers were broken, use pop/vg_gyrotropy_nonthermal instead. See link (https://github.com/fmihpc/analysator/pull/262)"
deprecated_datareducers['pop/vg_agyrotropy_thermal'] = "Previous pop/vg_agyrotropy_thermal reducers were broken, use pop/vg_gyrotropy_thermal instead. See link (https://github.com/fmihpc/analysator/pull/262)"

import reduction_sidecar
