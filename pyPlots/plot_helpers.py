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

import logging
import pytools as pt
import numpy as np
import sys
from rotation import rotateTensorToVector
import warnings

PLANE = 'XY'
# or alternatively, 'XZ'
CELLSIZE = 300000.0 # cell size
DT = 0.5 # time step

kb = 1.38065e-23
elementalcharge = 1.6021773e-19
mu_0 = 1.25663706144e-6


def inplane(inputarray):
    # Assumes input array is of format [nx,ny,3]
    if PLANE=='XY':
        inputarray[:,:,2] = np.zeros(inputarray[:,:,2].shape)
    elif PLANE=='XZ':
        inputarray[:,:,1] = np.zeros(inputarray[:,:,1].shape)
    elif PLANE=='YZ':
        inputarray[:,:,0] = np.zeros(inputarray[:,:,0].shape)
    else:
        logging.info("Error defining plane!")
        return -1
    return inputarray

def inplanevec(inputarray):
    # Assumes input array is of format [nx,ny,3]
    if PLANE=='XY':
        return inputarray[:,:,0:2]
    elif PLANE=='XZ':
        return inputarray[:,:,0:3:2]
    elif PLANE=='YZ':
        return inputarray[:,:,1:3]
    else:
        logging.info("Error defining plane!")
        return -1

def numjacobian(inputarray):
    # Assumes input array is of format [nx,ny,3]
    nx,ny = inputarray[:,:,0].shape
    jac = np.zeros([nx,ny,3,3])
    if PLANE=='XY':
        jac[:,:,0,0], jac[:,:,0,1] = np.gradient(inputarray[:,:,0], CELLSIZE)
        jac[:,:,1,0], jac[:,:,1,1] = np.gradient(inputarray[:,:,1], CELLSIZE)
        jac[:,:,2,0], jac[:,:,2,1] = np.gradient(inputarray[:,:,2], CELLSIZE)
    elif PLANE=='XZ':
        jac[:,:,0,0], jac[:,:,0,2] = np.gradient(inputarray[:,:,0], CELLSIZE)
        jac[:,:,1,0], jac[:,:,1,2] = np.gradient(inputarray[:,:,1], CELLSIZE)
        jac[:,:,2,0], jac[:,:,2,2] = np.gradient(inputarray[:,:,2], CELLSIZE)
    elif PLANE=='YZ':
        jac[:,:,0,1], jac[:,:,0,2] = np.gradient(inputarray[:,:,0], CELLSIZE)
        jac[:,:,1,1], jac[:,:,1,2] = np.gradient(inputarray[:,:,1], CELLSIZE)
        jac[:,:,2,1], jac[:,:,2,2] = np.gradient(inputarray[:,:,2], CELLSIZE)
    else:
        logging.info("Error defining plane!")
        return -1
    # Output array is of format [nx,ny,3,3]
    #  :,:,component, derivativedirection
    # so dAx/dx = :,:,0,0
    #    DAy/dz = :,:,1,2
    return jac

def numjacobian3d(inputarray):
    # Assumes input array is of format [nx,ny,nz,3]
    nx,ny,nz = inputarray[:,:,:,0].shape
    jac = np.zeros([nx,ny,nz,3,3])
    jac[:,:,:,0,0], jac[:,:,:,0,1], jac[:,:,:,0,2] = np.gradient(inputarray[:,:,:,0], CELLSIZE)
    jac[:,:,:,1,0], jac[:,:,:,1,1], jac[:,:,:,1,2] = np.gradient(inputarray[:,:,:,1], CELLSIZE)
    jac[:,:,:,2,0], jac[:,:,:,2,1], jac[:,:,:,2,2] = np.gradient(inputarray[:,:,:,2], CELLSIZE)
    # Output array is of format [nx,ny,nz,3,3]
    #  :,:,component, derivativedirection
    # so for example: dAx/dx = :,:,:,0,0
    #                 DAy/dz = :,:,:,1,2
    return jac

def numgradscalar(inputarray):
    # Assumes input array is of format [nx,ny]
    nx,ny = inputarray.shape
    grad = np.ma.zeros([nx,ny,3])
    if PLANE=='XY':
        grad[:,:,0],grad[:,:,1] = np.gradient(inputarray, CELLSIZE)
    elif PLANE=='XZ':
        grad[:,:,0],grad[:,:,2] = np.gradient(inputarray, CELLSIZE)
    elif PLANE=='YZ':
        grad[:,:,1],grad[:,:,2] = np.gradient(inputarray, CELLSIZE)
    else:
        logging.info("Error defining plane!")
        return -1
    # Output array is of format [nx,ny,3]
    return grad

def expandMask(inputarray):
    mask = 1.*inputarray.mask
    if np.ndim(mask) == 3:
        mask = 1.*mask.any(axis=2)
    newmask = 1.*mask
    newmask = np.logical_or(np.roll(mask,1,axis=0),newmask)
    newmask = np.logical_or(np.roll(mask,-1,axis=0),newmask)
    newmask = np.logical_or(np.roll(mask,1,axis=1),newmask)
    newmask = np.logical_or(np.roll(mask,-1,axis=1),newmask)
    newmask = np.logical_or(np.roll(mask,(1,1),axis=(0,1)),newmask)
    newmask = np.logical_or(np.roll(mask,(1,-1),axis=(0,1)),newmask)
    newmask = np.logical_or(np.roll(mask,(-1,1),axis=(0,1)),newmask)
    newmask = np.logical_or(np.roll(mask,(-1,-1),axis=(0,1)),newmask)
    if np.ndim(inputarray) == 2:
        inputarray.mask = 1.*newmask
    else:
        for i in range(3):
            inputarray.mask[:,:,i] = 1.*newmask
    return inputarray

def numdiv(inputarray):
    # Assumes input array is of format [nx,ny,3]
    jac = numjacobian(inputarray)
    # Output array is of format [nx,ny]
    return jac[:,:,0,0] + jac[:,:,1,1] + jac[:,:,2,2]

def numdivtensor(inputtensor):
    # Assumes input tensor is of format [nx,ny,3,3]
    result = np.zeros_like(inputtensor[:,:,0,:])
    result[:,:,0] = numdiv(inputtensor[:,:,0,:])
    result[:,:,1] = numdiv(inputtensor[:,:,1,:])
    result[:,:,2] = numdiv(inputtensor[:,:,2,:])
    return result

def numcrossproduct(inputvector1, inputvector2):
    # assumes inputvectors are of shape [nx,ny,3]
    # in fact nothing special here
    # Output array is of format [nx,ny,3]
    return np.cross(inputvector1,inputvector2)

def numcurl(inputarray):
    # Assumes input array is of format [nx,ny,3]
    jac = numjacobian(inputarray)
    # Output array is of format [nx,ny,3]
    curl = np.zeros(inputarray.shape)
    curl[:,:,0] = jac[:,:,2,1]-jac[:,:,1,2]
    curl[:,:,1] = jac[:,:,0,2]-jac[:,:,2,0]
    curl[:,:,2] = jac[:,:,1,0]-jac[:,:,0,1]
    return curl

def numcurl3d(inputarray):
    # Assumes input array is of format [nx,ny,nz,3]
    jac = numjacobian3d(inputarray)
    # Output array is of format [nx,ny,nz,3]
    curl = np.zeros(inputarray.shape)
    curl[:,:,:,0] = jac[:,:,:,2,1]-jac[:,:,:,1,2]
    curl[:,:,:,1] = jac[:,:,:,0,2]-jac[:,:,:,2,0]
    curl[:,:,:,2] = jac[:,:,:,1,0]-jac[:,:,:,0,1]
    return curl

def vanleer(left, cent, right):
    # "vanLeer" limiter as in Vlasiator
    eps = np.finfo(float).eps
    numerator = np.maximum((right - cent) * (cent - left), np.zeros_like(cent))
    denumerator = right - left
    r = np.where(np.abs(denumerator) < eps, 0.0, 2.0 * numerator / denumerator)
    return r


def limitedgradient(inputarray, *varargs):
    # numpy-like limited gradient functionality using a limiter,
    # tries to function as a drop-in replacement for np.gradient,
    # and is heavily drawn from numpy.
    f = np.asanyarray(inputarray)
    N = f.ndim  # number of dimensions
    axes = tuple(range(N))

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            if np.ndim(distances) == 0:
                continue
            elif np.ndim(distances) != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            diffx = np.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N
    outvals = []
    for axis, ax_dx in zip(axes, dx):
        out = np.empty_like(f, dtype=f.dtype)
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        out[tuple()]
        out[tuple(slice1)] = vanleer(f[tuple(slice2)],
                                     f[tuple(slice1)],
                                     f[tuple(slice4)]) / ax_dx

        # handle edges by setting to zero
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        out[tuple(slice1)] = 0.0

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        out[tuple(slice1)] = 0.0

        outvals.append(out)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


def numcurllimited(inputarray):
    # Curl calculation using slope-limited gradients
    # Assumes input array is of format [nx,ny,3]
    nx, ny = inputarray[:, :, 0].shape
    jac = np.zeros([nx, ny, 3, 3])
    if PLANE == 'XY':
        jac[:, :, 0, 0], jac[:, :, 0, 1] = limitedgradient(inputarray[:, :, 0], CELLSIZE)
        jac[:, :, 1, 0], jac[:, :, 1, 1] = limitedgradient(inputarray[:, :, 1], CELLSIZE)
        jac[:, :, 2, 0], jac[:, :, 2, 1] = limitedgradient(inputarray[:, :, 2], CELLSIZE)
    elif PLANE == 'XZ':
        jac[:, :, 0, 0], jac[:, :, 0, 2] = limitedgradient(inputarray[:, :, 0], CELLSIZE)
        jac[:, :, 1, 0], jac[:, :, 1, 2] = limitedgradient(inputarray[:, :, 1], CELLSIZE)
        jac[:, :, 2, 0], jac[:, :, 2, 2] = limitedgradient(inputarray[:, :, 2], CELLSIZE)
    elif PLANE == 'YZ':
        jac[:, :, 0, 1], jac[:, :, 0, 2] = limitedgradient(inputarray[:, :, 0], CELLSIZE)
        jac[:, :, 1, 1], jac[:, :, 1, 2] = limitedgradient(inputarray[:, :, 1], CELLSIZE)
        jac[:, :, 2, 1], jac[:, :, 2, 2] = limitedgradient(inputarray[:, :, 2], CELLSIZE)
    else:
        logging.info("Error defining plane!")
        return -1
    # Output array is of format [nx,ny,3,3]
    #  :,:,component, derivativedirection
    # so dAx/dx = :,:,0,0
    #    DAy/dz = :,:,1,2

    # Output array is of format [nx,ny,3]
    curl = np.zeros(inputarray.shape)
    curl[:, :, 0] = jac[:, :, 2, 1] - jac[:, :, 1, 2]
    curl[:, :, 1] = jac[:, :, 0, 2] - jac[:, :, 2, 0]
    curl[:, :, 2] = jac[:, :, 1, 0] - jac[:, :, 0, 1]

    return curl


def numvecdotdelvec(inputarray1, inputarray2):
    # (V1 dot Del)V2
    # Assumesinput arrays are of format [nx,ny,3]
    if inputarray1.shape!=inputarray2.shape:
        logging.info("Error: Input array shapes don't match!",inputarray1.shape,inputarray2.shape)
        return -1
    result = np.zeros(inputarray1.shape)
    jac = numjacobian(inputarray2)
    result[:,:,0] = (inputarray1[:,:,0]*jac[:,:,0,0] +
                     inputarray1[:,:,1]*jac[:,:,0,1] +
                     inputarray1[:,:,2]*jac[:,:,0,2] )
    result[:,:,1] = (inputarray1[:,:,0]*jac[:,:,1,0] +
                     inputarray1[:,:,1]*jac[:,:,1,1] +
                     inputarray1[:,:,2]*jac[:,:,1,2] )
    result[:,:,2] = (inputarray1[:,:,0]*jac[:,:,2,0] +
                     inputarray1[:,:,1]*jac[:,:,2,1] +
                     inputarray1[:,:,2]*jac[:,:,2,2] )
    return result

def numvecdottensor(inputarray, inputtensor):
    # Assumesinput array is of format [nx,ny,3]
    # assumes inputtensor is of shape [nx,ny,3,3]
    result = np.zeros(inputarray.shape)
    result[:,:,0] = (inputarray[:,:,0]*inputtensor[:,:,0,0] # is this indexing correcT?
                  + inputarray[:,:,1]*inputtensor[:,:,0,1]
                  + inputarray[:,:,2]*inputtensor[:,:,0,2])
    result[:,:,1] = (inputarray[:,:,0]*inputtensor[:,:,1,0]
                  + inputarray[:,:,1]*inputtensor[:,:,1,1]
                  + inputarray[:,:,2]*inputtensor[:,:,1,2])
    result[:,:,2] = (inputarray[:,:,0]*inputtensor[:,:,2,0]
                  + inputarray[:,:,1]*inputtensor[:,:,2,1]
                  + inputarray[:,:,2]*inputtensor[:,:,2,2])
    # Output array is of format [nx,ny,3]
    return result

def TransposeVectorArray(inputarray):
    # Assumes input array is of format [nA,nB,nV]
    # Output array is of format [nB,nA,nV]
    return np.transpose(inputarray, (1,0,2))

def rotateTensorArrayToVectorArray(inputtensor, inputvector):
    # assumes inputtensor is of shape [nx,ny,3,3] and
    # inputvector is of shape [nx,ny,3]
    rotated = np.zeros(inputtensor.shape)
    nx, ny = inputtensor[:,:,0,0].shape
    # rotates so that tensor z-axis is parallel with vector
    for i in np.arange(nx):
        for j in np.arange(ny):
            rotated[i,j,:,:]= rotateTensorToVector(inputtensor[i,j,:,:],inputvector[i,j,:])
    # Output array is of format [nx,ny,3,3]
    return rotated

def TensorArrayParallelComponent(inputtensor):
    # assumes inputtensor is of shape [nx,ny,3,3]
    # Output array is of format [nx,ny]
    return inputtensor[:,:,2,2]

def TensorArrayPerpendicularComponent(inputtensor):
    # assumes inputtensor is of shape [nx,ny,3,3]
    # Output array is of format [nx,ny]
    return 0.5*(inputtensor[:,:,0,0]+inputtensor[:,:,1,1])

def TensorArrayAnisotropy(inputtensor):
    # assumes inputtensor is of shape [nx,ny,3,3]
    # Output array is of format [nx,ny]
    return np.divide(np.abs(TensorArrayPerpendicularComponent(inputtensor)), np.ma.masked_less_equal(np.abs(TensorArrayParallelComponent(inputtensor)),0))

def VectorArrayParallelComponent(inputvector, directionvector):
    # assumes inputvector and directionvector are of shape [nx,ny,3]
    dirnorm = np.divide(directionvector, np.linalg.norm(directionvector, axis=-1)[:,:,np.newaxis])
    # Need to perform dot product in smaller steps due to memory constraints
    # result = np.zeros(inputvector[:,:,0].shape)
    # for i in np.arange(len(inputvector[:,0,0])):
    #     result[i,:] = np.inner(inputvector[i,:,:],dirnorm[i,:,:])
    # Output array is of format [nx,ny]
    result = (inputvector*dirnorm).sum(-1) #dot product, alternatively numpy.einsum("ijk,ijk->ij",a,b)
    return result

def VectorArrayPerpendicularComponent(inputvector, directionvector):
    # assumes inputvector and directionvector are of shape [nx,ny,3]
    # Calculates the magnitude of the perpendicular vector component
    # of each inputvector to each directionvector.
    dirnorm = np.ma.divide(directionvector, np.linalg.norm(directionvector, axis=-1)[:,:,np.newaxis])
    # Need to perform dot product in smaller steps due to memory constraints
    # paravector = dirnorm
    # for i in np.arange(len(inputvector[:,0,0])):
    #     paravector[i,:] = paravector[i,:] * np.inner(inputvector[i,:,:],dirnorm[i,:,:])[:,:,np.newaxis]
    #paravector = dirnorm * np.inner(inputvector, dirnorm)[:,:,np.newaxis]
    paravector = dirnorm * (inputvector*dirnorm).sum(-1)[:,:,np.newaxis] #dot product, alternatively numpy.einsum("ijk,ijk->ij",a,b)
    perpcomp = np.linalg.norm(inputvector - paravector, axis=-1)
    # Output array is of format [nx,ny]
    return perpcomp

def VectorArrayPerpendicularVector(inputvector, directionvector):
    # assumes inputvector and directionvector are of shape [nx,ny,3]
    # Calculates the perpendicular vector component of each
    # inputvector to each directionvector *as a vector*
    dirnorm = np.divide(directionvector, np.linalg.norm(directionvector, axis=-1)[:,:,np.newaxis])
    paravector = dirnorm * (inputvector*dirnorm).sum(-1)[:,:,np.newaxis] #dot product, alternatively numpy.einsum("ijk,ijk->ij",a,b)
    perpvector = inputvector - paravector
    # Output array is of format [nx,ny,3]
    return perpvector

def VectorArrayAnisotropy(inputvector, directionvector):
    # assumes inputvector and directionvector are of shape [nx,ny,3]
    paracomp = VectorArrayParallelComponent(inputvector, directionvector)
    perpcomp = VectorArrayPerpendicularComponent(inputvector, directionvector)
    # Output array is of format [nx,ny]
    return np.divide(np.abs(perpcomp), np.ma.masked_less_equal(np.abs(paracomp),0))








def vec_MagneticPressureForce(inputarray):
    # assumes Magnetic field of shape [nx,ny,3]
    Bmap = inputarray
    Bnorm = np.linalg.norm(Bmap, axis=-1)
    mu0 = 1.25663706144e-6
    MagPressure = Bnorm*Bnorm/(2.*mu0)
    # Output array is of format [nx,ny,3]
    return (-1.)*numgradscalar(MagPressure)

def vec_MagneticTensionForce(inputarray):
    # Assumes input array is of format [nx,ny,3] with content B
    mu0 = 1.25663706144e-6
    jac = numjacobian(inputarray)
    nx, ny = inputarray[:,:,0].shape
    tensforce = np.zeros([nx,ny,3])
    tensforce[:,:,0] = (1./mu0) * (inputarray[:,:,0]*jac[:,:,0,0] + inputarray[:,:,1]*jac[:,:,0,1] + inputarray[:,:,2]*jac[:,:,0,2])
    tensforce[:,:,1] = (1./mu0) * (inputarray[:,:,0]*jac[:,:,1,0] + inputarray[:,:,1]*jac[:,:,1,1] + inputarray[:,:,2]*jac[:,:,1,2])
    tensforce[:,:,2] = (1./mu0) * (inputarray[:,:,0]*jac[:,:,2,0] + inputarray[:,:,1]*jac[:,:,2,1] + inputarray[:,:,2]*jac[:,:,2,2])
    # Output array is of format [nx,ny,3]
    return tensforce

def vec_ThermalPressureForce(inputarray):
    # assumes Thermal Pressure of shape [nx,ny]
    return (-1.)*numgradscalar(inputarray)

def vec_currentdensity(inputarray):
    # Assumes input array is of format [nx,ny,3] with content B or perb
    mu0 = 1.25663706144e-6
    # Output array is of format [nx,ny,3]
    return numcurl(inputarray) / mu0

def vec_currentdensity3d(inputarray):
    # Assumes input array is of format [nx,ny,nz,3] with content B
    mu0 = 1.25663706144e-6
    # Output array is of format [nx,ny,3]
    return numcurl3d(inputarray) / mu0

def vec_currentdensity_lim(inputarray):
    # Assumes input array is of format [nx,ny,3] with content B or perb
    mu0 = 1.25663706144e-6
    # Output array is of format [nx,ny,3]
    return numcurllimited(inputarray) / mu0

def vec_Hallterm(currentdensity, magneticfield, chargedensity):
    # assumes current density of shape [nx,ny,3]
    # assumes Magnetic field of shape [nx,ny,3]
    # assumes charge density of shape [nx,ny]
    crossp = np.cross(currentdensity, magneticfield)
    chargedensity = np.ma.masked_less_equal(chargedensity, 0) # ions only
    # Output array is of format [nx,ny,3]
    return np.ma.divide(crossp, chargedensity[:,:,np.newaxis])

def vec_ElectricFieldForce(electricfield, numberdensity):
    unitcharge = 1.602177e-19
    force = (-1.) * numberdensity[:,:,np.newaxis] * electricfield * unitcharge
    return force
    # force = np.zeros(np.array(electricfield.shape))
    # force[:,:,0] = (-1.) * numberdensity * electricfield[:,:,0] * unitcharge
    # force[:,:,1] = (-1.) * numberdensity * electricfield[:,:,1] * unitcharge
    # force[:,:,2] = (-1.) * numberdensity * electricfield[:,:,2] * unitcharge
    # return force

# pass_maps is a list of numpy arrays
# Each array has 2 dimensions [ysize, xsize]
# or 3 dimensions [ysize, xsize, components]

def expr_timeavg(pass_maps, requestvariables=False):
    if requestvariables==True:
        return []
    # Select first found variable
    listofkeys = iter(pass_maps[0])
    while True:
        var = next(listofkeys)
        if var!="dstep": break
    ntimes = len(pass_maps)

    warnings.warn("expr_timeavg cleaned to not produce undefined variable errors, see commit e1d2dd8ecaa7a0444ce56215f795a5237f792b1e for applied changes and check the output!")
    thismap = pass_maps[var]
    avgmap = np.zeros(np.array(thismap.shape))
    for i in range(ntimes):
        avgmap = np.add(avgmap, pass_maps[i][var])
    avgmap = np.divide(np.ma.masked_less_equal(avgmap,0), np.array([ntimes]))
    return avgmap

def expr_Diff(pass_maps, requestvariables=False):
    if requestvariables==True:
        return []
    # Assumes two "time steps" i.e. files to diff
    listofkeys = iter(pass_maps[0])
    while True:
        var = next(listofkeys)
        if var!="dstep": break
    map0=pass_maps[0][var]
    map1=pass_maps[1][var]
    if (map0.shape != map1.shape):
        logging.info("Error with diff: incompatible map shapes! " + str(map0.shape) + " vs " + str(map1.shape))
        sys.exit(-1)
    if (map0.dtype in ['uint8','uint16','uint32','uint64']):
        logging.info("Diff: Converting from unsigned to signed integers")
        map0 = map0.astype('int64')
        map1 = map1.astype('int64')
    return (map0-map1) # use keyword absolute to get abs diff

def expr_Hall(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','rhoq']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    RhoQmap = pass_maps['rhoq'].T # charge density
    Jmap = vec_currentdensity(Bmap)
    Hallterm = vec_Hallterm(Jmap,Bmap,RhoQmap)
    return np.swapaxes(Hallterm, 0,1)

def expr_Hall_lim(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','rhoq']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    RhoQmap = pass_maps['rhoq'].T # charge density
    Jmap_lim = vec_currentdensity_lim(Bmap)
    Hallterm_lim = vec_Hallterm(Jmap_lim,Bmap,RhoQmap)
    return np.swapaxes(Hallterm_lim, 0,1)

def expr_Hall_aniso(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','rhoq']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    RhoQmap = pass_maps['rhoq'].T # number density
    Jmap = vec_currentdensity(Bmap)
    Hallterm = vec_Hallterm(Jmap,Bmap,RhoQmap)
    return VectorArrayAnisotropy(Hallterm,Bmap).T

def expr_J(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']

    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Jmap = vec_currentdensity(Bmap)
    return np.swapaxes(Jmap, 0,1)

def expr_JperBperp(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']

    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Jmap = vec_currentdensity(Bmap)
    Bperp = VectorArrayPerpendicularComponent(Bmap, np.ma.masked_less_equal(Jmap,0))
    return np.swapaxes(np.ma.divide(np.linalg.norm(Jmap,axis=-1), np.ma.masked_less_equal(np.abs(Bperp),0)),0,1)

def expr_log2JperBperp(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']

    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Jmap = vec_currentdensity(Bmap)
    Bperp = VectorArrayPerpendicularComponent(Bmap, np.ma.masked_less_equal(Jmap,0))
    JperBperp = np.swapaxes(np.ma.divide(np.linalg.norm(Jmap,axis=-1), np.ma.masked_less_equal(np.abs(Bperp),0)),0,1)
    return np.ma.log2(JperBperp)

def expr_J3d(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['fg_b','3d']

    Bmap = pass_maps['fg_b'] # Magnetic field
    Jmap3d = vec_currentdensity3d(Bmap)
    return Jmap3d

def expr_J_limited_mag(pass_maps, requestvariables=False):
    # Slope-limited J = rot(B)
    if requestvariables==True:
        return ['B']

    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Jmap = vec_currentdensity_lim(Bmap)

    return np.linalg.norm(Jmap, axis=-1).T

def expr_J_dperB_mag(pass_maps, requestvariables=False):
    # J = rot(B) from Vlasiator-given derivatives
    if requestvariables==True:
        return ['vg_dperbxvoldy',
                'vg_dperbxvoldz',
                'vg_dperbyvoldx',
                'vg_dperbyvoldz',
                'vg_dperbzvoldx',
                'vg_dperbzvoldy',
                ]
    Bxdy = pass_maps['vg_dperbxvoldy'].T
    Bxdz = pass_maps['vg_dperbxvoldz'].T
    Bydx = pass_maps['vg_dperbyvoldx'].T
    Bydz = pass_maps['vg_dperbyvoldz'].T
    Bzdx = pass_maps['vg_dperbzvoldx'].T
    Bzdy = pass_maps['vg_dperbzvoldy'].T

    mu0 = 1.25663706144e-6
    shap = list(Bxdy.shape)
    shap.append(3)
    Jmap = np.zeros(shap)

    Jmap[:, :, 0] = (Bzdy - Bydz) / mu0
    Jmap[:, :, 1] = (Bxdz - Bzdx) / mu0
    Jmap[:, :, 2] = (Bydx - Bxdy) / mu0
    return np.linalg.norm(Jmap, axis=-1).T


def expr_J_mag_limiter_ratio(pass_maps, requestvariables=False):
    # Ratio between magnitudes of J = rot(B_vol) and J =
    if requestvariables is True:
        return ['B']

    Bmap = TransposeVectorArray(pass_maps['B'])  # Magnetic field

    mu0 = 1.25663706144e-6
    Jmap = np.zeros_like(Bmap)

    Jmap = vec_currentdensity(Bmap)

    jlimn = np.linalg.norm(vec_currentdensity_lim(Bmap), axis=-1)
    Jmap = np.where(jlimn < np.finfo(float).eps, 0.0, np.linalg.norm(Jmap, axis=-1) / jlimn)
    return Jmap.T

def expr_J_aniso(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Jmap = vec_currentdensity(Bmap)
    return VectorArrayAnisotropy(Jmap,Bmap).T

def expr_MagneticPressureForce(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    MagneticPressureForce = vec_MagneticPressureForce(Bmap)
    return np.swapaxes(MagneticPressureForce, 0,1)

def expr_MagneticPressureForce_aniso(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    MagneticPressureForce = vec_MagneticPressureForce(Bmap)
    MagneticPressureForceAniso = VectorArrayAnisotropy(MagneticPressureForce, Bmap)
    return MagneticPressureForceAniso.T

# def expr_MagneticPressureForce_inplane_mag(pass_maps):
#     Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
#     MagneticPressureForce = vec_MagneticPressureForce(Bmap)
#     return np.linalg.norm(inplane(MagneticPressureForce), axis=-1).T

# def expr_MagneticPressureForce_inplane_aniso(pass_maps):
#     Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
#     MagneticPressureForce = vec_MagneticPressureForce(Bmap)
#     MagneticPressureForceIPAniso = VectorArrayAnisotropy(inplane(MagneticPressureForce), inplane(Bmap))
#     return MagneticPressureForceIPAniso.T

def expr_ThermalPressureForce(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['Pressure']
    Pmap = pass_maps['Pressure'].T #Pressure (scalar)
    return np.swapaxes(vec_ThermalPressureForce(Pmap), 0,1)

def expr_ThermalPressureForce_aniso(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','Pressure']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Pmap = pass_maps['Pressure'].T #Pressure (scalar)
    ThermalPressureForceAniso = VectorArrayAnisotropy(vec_ThermalPressureForce(Pmap), Bmap)
    return ThermalPressureForceAniso.T

# def expr_ThermalPressureForce_inplane_aniso(pass_maps):
#     Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
#     Pmap = pass_maps['Pressure'].T #Pressure (scalar)
#     ThermalPressureForce = vec_ThermalPressureForce(Pmap)
#     ThermalPressureForceIPAniso = VectorArrayAnisotropy(inplane(ThermalPressureForce), inplane(Bmap))
#     return ThermalPressureForceIPAniso.T

def expr_Eforce(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['E','rho']
    Emap = TransposeVectorArray(pass_maps['E']) # Electric field
    Rhomap = pass_maps['rho'].T # number density
    Efieldforce = vec_ElectricFieldForce(Emap, Rhomap)
    return np.swapaxes(Efieldforce, 0,1)

def expr_Btension(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    MagTensionForce = vec_MagneticTensionForce(Bmap)
    return np.swapaxes(MagTensionForce, 0,1)

# No expr_Btension_aniso as it's all in the perpendicular component

def expr_Bforces(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    MagneticPressureForce = vec_MagneticPressureForce(Bmap)
    MagTensionForce = vec_MagneticTensionForce(Bmap)
    return np.swapaxes(MagneticPressureForce+MagTensionForce, 0,1)

def expr_Totforces(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','Pressure']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Pmap = pass_maps['Pressure'].T #Pressure (scalar)
    MagneticPressureForce = vec_MagneticPressureForce(Bmap)
    MagTensionForce = vec_MagneticTensionForce(Bmap)
    ThermalPressureForce = vec_ThermalPressureForce(Pmap)
    return np.swapaxes(ThermalPressureForce+MagneticPressureForce+MagTensionForce, 0,1)

def expr_Totforces_aniso(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','Pressure']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Pmap = pass_maps['Pressure'].T #Pressure (scalar)
    MagneticPressureForce = vec_MagneticPressureForce(Bmap)
    MagTensionForce = vec_MagneticTensionForce(Bmap)
    ThermalPressureForce = vec_ThermalPressureForce(Pmap)
    return VectorArrayAnisotropy(ThermalPressureForce+MagneticPressureForce+MagTensionForce, Bmap).T

def expr_ratio_thermal_mag(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','Pressure']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Pmap = pass_maps['Pressure'].T #Pressure (scalar)
    MagneticPressureForce = vec_MagneticPressureForce(Bmap)
    MagTensionForce = vec_MagneticTensionForce(Bmap)
    ThermalPressureForce = vec_ThermalPressureForce(Pmap)
    MagForcesTot = np.ma.masked_less_equal(np.linalg.norm(MagneticPressureForce+MagTensionForce, axis=-1), 0)
    ThermalPressureForceTot = np.linalg.norm(ThermalPressureForce, axis=-1)
    return np.divide(ThermalPressureForceTot,MagForcesTot).T

def expr_E_parallel(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E']
    # No need for transposing in this one
    Bmap = pass_maps['B'] # Magnetic field
    Emap = pass_maps['E'] # Electric field
    return VectorArrayParallelComponent(Emap,Bmap)

def expr_E_perpendicular(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E']
    # No need for transposing in this one
    Bmap = pass_maps['B'] # Magnetic field
    Emap = pass_maps['E'] # Electric field
    return VectorArrayPerpendicularComponent(Emap,Bmap)

def expr_flowcompression(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['V']
    Vmap = TransposeVectorArray(pass_maps['V']) # Bulk flow
    return numdiv(Vmap).T

# def expr_gradB_aniso(pass_maps):
#     Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
#     gradB = numjacobian(Bmap)
#     rotatedgradB = rotateTensorArrayToVectorArray(gradB,Bmap)
#     gradBaniso = TensorArrayAnisotropy(rotatedgradB)
#     return gradBaniso.T

# def expr_gradPTD_aniso(pass_maps):
#     Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
#     PTDmap = TransposeVectorArray(pass_maps['PTensorDiagonal']) # PressureTensorDiagonal
#     gradP = numjacobian(PTDmap)
#     rotatedgradP = rotateTensorArrayToVectorArray(gradP,Bmap)
#     gradPaniso = TensorArrayAnisotropy(rotatedgradP)
#     return gradPaniso.T

slippageVA=3937129.92717945   #Effective alfven speed (in m/s) to use when calculating slippage.
def expr_Slippage(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['E','B','V']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_Slippage expected a single timestep, but got multiple. Exiting.")
        quit()

    expr_Slippage.__name__ = r"Slippage $[v_\mathrm{A}]$"

    E = pass_maps['E']
    B = pass_maps['B']
    V = pass_maps['V']

    Vperp = VectorArrayPerpendicularVector(V,B)
    EcrossB = np.divide(np.cross(E,B), (B*B).sum(-1)[:,:,np.newaxis])
    metricSlippage = EcrossB-Vperp
    alfvenicSlippage = metricSlippage/slippageVA
    return alfvenicSlippage

def expr_EcrossB(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['E','B']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_EcrossB expected a single timestep, but got multiple. Exiting.")
        quit()

    E = pass_maps['E']
    B = pass_maps['B']

    EcrossB = np.divide(np.cross(E,B), (B*B).sum(-1)[:,:,np.newaxis])
    return EcrossB

def expr_betatron(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E','PPerpendicular']
    # pass_maps is a list of numpy arrays
    # Each array has 2 dimensions [ysize, xsize]
    # or 3 dimensions [ysize, xsize, components]

    # for time averaging, it's a list of lists, where the top level
    # is 2N+1 timesteps with the middle one the requested time step

    # This custom expression returns a proxy for betatron acceleration
    if type(pass_maps) is not list:
        # Not a list of time steps, calculating this value does not make sense.
        logging.info("expr_betatron expected a list of timesteps to average from, but got a single timestep. Exiting.")
        quit()

    # Multiple time steps were found. This should be 3, for a time derivative.
    dsteps = [x['dstep'] for x in pass_maps]
    curri = dsteps.index(0)
    previ = dsteps.index(-1)

    thesemaps = pass_maps[curri]
    pastmaps = pass_maps[previ]

    thisB = TransposeVectorArray(thesemaps['B'])
    pastB = TransposeVectorArray(pastmaps['B'])
    thisBmag = np.linalg.norm(thisB, axis=-1)
    pastBmag = np.linalg.norm(pastB, axis=-1)
    dBdt = (thisBmag-pastBmag)/DT

    thisE = TransposeVectorArray(thesemaps['E'])
    thisPperp = thesemaps['PPerpendicular'].T

    # E x B drift = (E X B)/B^2
    B2 = (thisBmag*thisBmag)
    UExB = numcrossproduct(thisE,thisB)/B2[:,:,np.newaxis]
    gradB = numgradscalar(thisBmag)

    # dot product is (A*B).sum(-1)
    result = (thisPperp/thisBmag)*(dBdt + (UExB*gradB).sum(-1))
    return result.T

def expr_Fermi(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E','PParallel']
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Emap = TransposeVectorArray(pass_maps['E']) # Electric field
    Pparallel = pass_maps['PParallel'].T #Pressure (scalar)

    Bmag = np.linalg.norm(Bmap, axis=-1)
    B2 = (Bmag*Bmag)
    UExB = numcrossproduct(Emap,Bmap)/B2[:,:,np.newaxis]

    Bnorm = (Bmap/Bmag[:,:,np.newaxis])
    Bjac = numjacobian(Bnorm)
    kappa = numvecdottensor(Bnorm,Bjac)
    # (A*B).sum(-1) is dot product
    result = Pparallel*(UExB*kappa).sum(-1)
    return result.T

def expr_EJ_parallel(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E']
    # Needs B,E
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Emap = TransposeVectorArray(pass_maps['E']) # Electric field

    Jmap = vec_currentdensity(Bmap)
    Jpara = VectorArrayParallelComponent(Jmap, Bmap)
    Epara = VectorArrayParallelComponent(Emap, Bmap)
    EJpara = (Jpara*Epara)
    return EJpara.T

def expr_Eacc_parallel(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E','V','rho']
    # No need for transposing in this one
    # Needs B,E,V,rho
    Bmap = pass_maps['B'] # Magnetic field
    Emap = pass_maps['E'] # Electric field
    Vmap = pass_maps['V'] # Bulk flow
    rhomap = pass_maps['rho'] # density
    protoncharge = .16021773e-18

    Vpara = VectorArrayParallelComponent(Vmap, Bmap)
    Epara = VectorArrayParallelComponent(Emap, Bmap)
    qnEVpara = (Vpara*Epara*rhomap*protoncharge)
    return qnEVpara


def expr_diamagnetic(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','rho','V','Pressure']
    # pass_maps is a list of numpy arrays
    # Each array has 2 dimensions [ysize, xsize]
    # or 3 dimensions [ysize, xsize, components]

    # for time averaging, it's a list of lists, where the top level
    # is 2N+1 timesteps with the middle one the requested time step

    # This custom expression returns a proxy for betatron acceleration
    if type(pass_maps) is not list:
        # Not a list of time steps, calculating this value does not make sense.
        logging.info("expr_diamagnetic expected a list of timesteps to average from, but got a single timestep. Exiting.")
        quit()

    # Multiple time steps were found. This should be 3, for a time derivative.
    dsteps = [x['dstep'] for x in pass_maps]
    curri = dsteps.index(0)
    previ = dsteps.index(-1)
    thesemaps = pass_maps[curri]
    pastmaps = pass_maps[previ]

    thisV = TransposeVectorArray(thesemaps['V'])
    pastV = TransposeVectorArray(pastmaps['V'])

    B = TransposeVectorArray(thesemaps['B'])
    rhom = thesemaps['rho'].T * 1.6726e-27
    Pres = thesemaps['Pressure'].T

    dUdt = (thisV-pastV)/DT
    rhodUdt = rhom[:,:,np.newaxis] * dUdt
    BperB2 =  B / (B*B).sum(-1)[:,:,np.newaxis]

    gradP = numgradscalar(Pres)

    term1 = numcrossproduct(BperB2, rhodUdt)
    term2 = numcrossproduct(BperB2, gradP)

    return np.swapaxes(term1+term2, 0,1)

def expr_diamagnetic_noinertial(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','Pressure']
    # pass_maps is a list of numpy arrays
    # Each array has 2 dimensions [ysize, xsize]
    # or 3 dimensions [ysize, xsize, components]

    B = TransposeVectorArray(pass_maps['B'])
    Pres = pass_maps['Pressure'].T

    BperB2 =  B / (B*B).sum(-1)[:,:,np.newaxis]
    gradP = numgradscalar(Pres)

    # Dropping the first term i.e. inertial current
    term2 = numcrossproduct(BperB2, gradP)

    return np.swapaxes(term2, 0,1)


def expr_jc(pass_maps, requestvariables=False):
    # current from curvature drift
    if requestvariables==True:
        return ['B','PParallel']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_jc expected a single timestep, but got multiple. Exiting.")
        quit()

    Pparallel = pass_maps['PParallel'].T #Pressure (scalar)
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    upBmag4 = np.linalg.norm(Bmap,axis=-1)**(-4)
    # (B dot Del)B
    BdotDelB = numvecdotdelvec(Bmap,Bmap)
    BxBdotDelB = numcrossproduct(Bmap, BdotDelB)
    result = BxBdotDelB * (Pparallel*upBmag4)[:,:,np.newaxis]
    return np.swapaxes(result, 0,1)

def expr_jg(pass_maps, requestvariables=False):
    # current from gradient drift
    if requestvariables==True:
        return ['B','PPerpendicular']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_jg expected a single timestep, but got multiple. Exiting.")
        quit()

    Pperp = pass_maps['PPerpendicular'].T #Pressure (scalar)
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Bmag = np.linalg.norm(Bmap,axis=-1)
    upBmag3 = Bmag**(-3)
    gradB = numgradscalar(Bmag)
    BxgradB = numcrossproduct(Bmap, gradB)
    result = BxgradB * (Pperp*upBmag3)[:,:,np.newaxis]
    return np.swapaxes(result, 0,1)

def expr_jgyr(pass_maps, requestvariables=False):
    # current from gyration effects (Daglis 1999) doi.org/10.1029/1999RG900009
    if requestvariables==True:
        return ['B','PPerpendicular']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_jgyr expected a single timestep, but got multiple. Exiting.")
        quit()

    Pperp = pass_maps['PPerpendicular'].T #Pressure (scalar)
    gradPperp = numgradscalar(Pperp)
    term1 = gradPperp

    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Bmag = np.linalg.norm(Bmap,axis=-1)
    gradB = numgradscalar(Bmag)
    B2 = Bmag*Bmag
    BpB2 = np.divide(Bmap,B2[:,:,np.newaxis])

    PperpGradBpB = Pperp[:,:,np.newaxis]*np.divide(gradB,Bmag[:,:,np.newaxis])
    term2 = PperpGradBpB

    # (B dot Del)B
    BdotDelB = numvecdotdelvec(Bmap,Bmap)
    PperppB2BdotDelB = np.divide(Pperp,B2)[:,:,np.newaxis] *BdotDelB
    term3 = PperppB2BdotDelB
    terms = term1-term2-term3
    jgyr = numcrossproduct(BpB2,terms)
    return np.swapaxes(jgyr, 0,1)

def expr_jring(pass_maps, requestvariables=False):
    # sum current (eq 4 Daglis 1999) doi.org/10.1029/1999RG900009
    if requestvariables==True:
        return ['B','PParallel','PPerpendicular']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_jring expected a single timestep, but got multiple. Exiting.")
        quit()

    Pperp = pass_maps['PPerpendicular'].T #Pressure (scalar)
    Ppara = pass_maps['PParallel'].T #Pressure (scalar)
    gradPperp = numgradscalar(Pperp)
    term1 = gradPperp

    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Bmag = np.linalg.norm(Bmap,axis=-1)
    gradB = numgradscalar(Bmag)
    B2 = Bmag*Bmag
    BpB2 = np.divide(Bmap,B2[:,:,np.newaxis])

    diffP = Ppara-Pperp
    diffPpB2 = np.divide(diffP,B2)
    # (B dot Del)B
    BdotDelB = numvecdotdelvec(Bmap,Bmap)
    PperppB2BdotDelB = diffPpB2[:,:,np.newaxis] *BdotDelB
    term2 = PperppB2BdotDelB
    terms = term1+term2
    jring = numcrossproduct(BpB2,terms)
    return np.swapaxes(jring, 0,1)

def expr_jp(pass_maps, requestvariables=False):
    # current from polarization drift
    if requestvariables==True:
        return ['B','V','rho']

    # This custom expression returns a proxy for betatron acceleration
    if type(pass_maps) is not list:
        # Not a list of time steps, calculating this value does not make sense.
        logging.info("expr_jp expected a list of timesteps to average from, but got a single timestep. Exiting.")
        quit()

    # Multiple time steps were found. This should be 3, for a time derivative.
    dsteps = [x['dstep'] for x in pass_maps]
    curri = dsteps.index(0)
    previ = dsteps.index(-1)
    thesemaps = pass_maps[curri]
    pastmaps = pass_maps[previ]

    thisV = TransposeVectorArray(thesemaps['V'])
    pastV = TransposeVectorArray(pastmaps['V'])
    dVdt = (thisV-pastV)/DT

    Bmap = TransposeVectorArray(thesemaps['B'])
    upBmag2 = np.linalg.norm(Bmap,axis=-1)**(-2)
    rhom = thesemaps['rho'].T * 1.6726e-27

    BxdVdt = numcrossproduct(Bmap, dVdt)
    result = BxdVdt * (rhom*upBmag2)[:,:,np.newaxis]
    return np.swapaxes(result, 0,1)

def expr_jm(pass_maps, requestvariables=False):
    # magnetization current
    if requestvariables==True:
        return ['B','PPerpendicular']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_jm expected a single timestep, but got multiple. Exiting.")
        quit()

    Pperp = pass_maps['PPerpendicular'].T #Pressure (scalar)
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    upBmag2 = np.linalg.norm(Bmap,axis=-1)**(-2)
    curlB = numcurl(Bmap)
    # (B dot Del)B
    result = -curlB * (Pperp*upBmag2)[:,:,np.newaxis]
    return np.swapaxes(result, 0,1)

def expr_ja(pass_maps, requestvariables=False):
    # Li, Guo, Li, Li (2017)
    # https://doi.org/10.3847/1538-4357/aa745e
    # additional current density (7), non-magnetization current
    if requestvariables==True:
        return ['B','PTensorRotated']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_ja expected a single timestep, but got multiple. Exiting.")
        quit()

    Ptensor = np.transpose(pass_maps['PTensorRotated'], (1,0,2,3))
    Bmap = TransposeVectorArray(pass_maps['B']) # Magnetic field
    Bmag = np.linalg.norm(Bmap,axis=-1)
    Bnorm = np.ma.divide(Bmap,Bmag[:,:,np.newaxis])
    upBmag2 = Bmag**(-2)
    Ppara = Ptensor[:,:,2,2]
    Pperp = 0.5*(Ptensor[:,:,0,0]+Ptensor[:,:,1,1])
    Pperpident = np.zeros_like(Ptensor)
    Pperpident[:,:,0,0] = Pperp
    Pperpident[:,:,1,1] = Pperp
    Pperpident[:,:,2,2] = Pperp
    bbtensor = np.einsum('ijk,ijl->ijkl',Bnorm,Bnorm)
    sumtensor = Ptensor - Pperpident - bbtensor*(Ppara-Pperp)[:,:,np.newaxis,np.newaxis]
    divsumtemsor = numdivtensor(sumtensor)
    result = - numcrossproduct(divsumtemsor,Bmap) * upBmag2[:,:,np.newaxis]
    return np.swapaxes(result, 0,1)


def expr_dLstardt(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['B','E','V']

    if type(pass_maps) is not list:
        # Not a list of time steps, calculating this value does not make sense.
        logging.info("expr_dLstardt expected a list of timesteps to average from, but got a single timestep. Exiting.")
        quit()

    # Multiple time steps were found. This should be 3, for a time derivative.
    dsteps = [x['dstep'] for x in pass_maps]
    curri = dsteps.index(0)
    previ = dsteps.index(-1)
    thesemaps = pass_maps[curri]
    pastmaps = pass_maps[previ]

    warnings.warn("expr_dLstardt cleaned to not produce undefined variable errors, see commit e1d2dd8ecaa7a0444ce56215f795a5237f792b1e for applied changes and check the output!")
    thisV = TransposeVectorArray(thesemaps['V'])
    pastV = TransposeVectorArray(pastmaps['V'])
    dVdt = (thisV-pastV)/DT

    Bmap = TransposeVectorArray(thesemaps['B'])
    upBmag2 = np.linalg.norm(Bmap,axis=-1)**(-2)
    rhom = thesemaps['rho'].T * 1.6726e-27

    BxdVdt = numcrossproduct(Bmap, dVdt)
    result = BxdVdt * (rhom*upBmag2)[:,:,np.newaxis]
    return np.swapaxes(result, 0,1)


################
## Note: pyPlots/plot_colormap.py already includes some functionality for plotting
## vectors and streamlines on top of the colormap. For simple variables, those will
## likely suffice. Use these external plotting routines for more involved calculations.


def overplotvectors(ax, XmeshXY,YmeshXY, pass_maps):
    # Example: B curvature force
    B = TransposeVectorArray(pass_maps['B'])
    vf = pt.plot.plot_helpers.vec_MagneticTensionForce(B)
    # any of the vec_ functions in plot_helpers.py are ok, and new ones can be constructed.
    # Variables already included in analysator can be plotted directly using plot_colormap.py.

    # Transpose back
    vf = TransposeVectorArray(vf)

    # Find vector lengths and define color
    lengths=np.linalg.norm(vf, axis=-1)
    colors = np.log10(lengths/np.mean(lengths))
    # Try to estimate step so there's about 100 vectors in the image area
    step = int(np.sqrt(colors.shape[0] * colors.shape[1]/100.))

    # inplane unit length vectors
    warnings.warn("usage of inplanevec(vf) is unverified! Used to be inplanevec(vectmap), with vectmap undefined. See changes in commit e1d2dd8ecaa7a0444ce56215f795a5237f792b1e and check if results are as expected!")
    vectmap = pt.plot.plot_helpers.inplanevec(vf)
    vectmap = vectmap / np.linalg.norm(vectmap, axis=-1)[:,:,np.newaxis]

    X = XmeshXY[::step,::step]
    Y = YmeshXY[::step,::step]
    U = vectmap[::step,::step,0]
    if PLANE=="XY":
        V = vectmap[::step,::step,1]
    elif PLANE=="XZ":
        V = vectmap[::step,::step,2]
    elif PLANE=="YZ":
        V = vectmap[::step,::step,0]
    C = colors[::step,::step]

    ax.quiver(X,Y,U,V,C, cmap='gray', units='dots',
              #scale=0.03/scale, # scale not defined - if you need to use this, adjust as necessary!
              headlength=2, headwidth=2,
              headaxislength=2, scale_units='dots', pivot='middle')


def overplotstreamlines(ax, XmeshXY,YmeshXY, pass_maps):
    # Select first valid variable
    listofkeys = iter(pass_maps)
    while True:
        var = next(listofkeys)
        if var!="dstep": break

    # Take in-plane component
    vfip = pt.plot.plot_helpers.inplanevec(var)

    X = XmeshXY
    Y = YmeshXY
    U = vfip[:,:,0]
    V = vfip[:,:,1]
    ax.streamplot(X,Y,U,V,linewidth=0.5, density=3, color='white')


def expr_electronflow(pass_maps, requestvariables=False):
    # Calculates the required current density to support the observed magnetic field.
    # Then calculates the required electron flow velocity to support that current.
    if requestvariables==True:
        return ['vg_b_vol','electron/vg_v','electron/vg_rho','proton/vg_v','proton/vg_rho']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronflow expected a single timestep, but got multiple. Exiting.")
        quit()

    unitcharge = 1.602177e-19
    mu0 = 1.25663706144e-6
    Bmap = TransposeVectorArray(pass_maps['vg_b_vol']) # Magnetic field
    evmap = TransposeVectorArray(pass_maps['electron/vg_v'])
    pvmap = TransposeVectorArray(pass_maps['proton/vg_v'])
    erhomap = pass_maps['electron/vg_rho'].T
    prhomap = pass_maps['proton/vg_rho'].T

    j = numcurl(Bmap)/mu0
    jprot = pvmap*prhomap[:,:,np.newaxis]*unitcharge
    jele = -evmap*erhomap[:,:,np.newaxis]*unitcharge
    reqjele = j - jprot
    reqjv = reqjele / erhomap[:,:,np.newaxis] / (-unitcharge)

    logging.info("mean jprot: " + str(np.mean(np.linalg.norm(jprot,axis=-1))) + ', mean jele: ' + str(np.mean(np.linalg.norm(jele,axis=-1))))
    logging.info("min jprot: " + str(np.min(np.linalg.norm(jprot,axis=-1))) +   ', min jele: ' + str(np.min(np.linalg.norm(jele,axis=-1))))
    logging.info("max jprot: " + str(np.max(np.linalg.norm(jprot,axis=-1))) +   ', max jele: ' + str(np.max(np.linalg.norm(jele,axis=-1))))

    logging.info("mean reqjv: " + np.mean(np.linalg.norm(reqjv,axis=-1))+'mean reqjele: ' + str(np.mean(np.linalg.norm(reqjele,axis=-1))))
    logging.info( "min reqjv: " + np.min(np.linalg.norm(reqjv,axis=-1)) + 'min reqjele: ' + str(np.min(np.linalg.norm(reqjele,axis=-1))))
    logging.info( "max reqjv: " + np.max(np.linalg.norm(reqjv,axis=-1)) + 'max reqjele: ' + str(np.max(np.linalg.norm(reqjele,axis=-1))))

    return np.swapaxes(reqjv, 0,1)

def expr_electronflowerr(pass_maps, requestvariables=False):
    # Calculates the required current density to support the observed magnetic field.
    # Then calculates the required electron flow velocity to support that current.
    # Then compares the current electron flow to that required.
    if requestvariables==True:
        return ['vg_b_vol','electron/vg_v','electron/vg_rho','proton/vg_v','proton/vg_rho']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronflow expected a single timestep, but got multiple. Exiting.")
        quit()

    unitcharge = 1.602177e-19
    mu0 = 1.25663706144e-6
    Bmap = TransposeVectorArray(pass_maps['vg_b_vol']) # Magnetic field
    evmap = TransposeVectorArray(pass_maps['electron/vg_v'])
    pvmap = TransposeVectorArray(pass_maps['proton/vg_v'])
    erhomap = pass_maps['electron/vg_rho'].T
    prhomap = pass_maps['proton/vg_rho'].T

    j = numcurl(Bmap)/mu0
    jprot = pvmap*prhomap[:,:,np.newaxis]*unitcharge
    jele = -evmap*erhomap[:,:,np.newaxis]*unitcharge
    reqjele = j - jprot
    reqjv = reqjele / erhomap[:,:,np.newaxis] / (-unitcharge)

    everror = evmap - reqjv
    logging.info("mean jprot",np.mean(np.linalg.norm(jprot,axis=-1)),'mean jele',np.mean(np.linalg.norm(jele,axis=-1)))
    logging.info("min jprot",np.min(np.linalg.norm(jprot,axis=-1)),'min jele',np.min(np.linalg.norm(jele,axis=-1)))
    logging.info("max jprot",np.max(np.linalg.norm(jprot,axis=-1)),'max jele',np.max(np.linalg.norm(jele,axis=-1)))

    logging.info("mean reqjv",np.mean(np.linalg.norm(reqjv,axis=-1)),'mean reqjele',np.mean(np.linalg.norm(reqjele,axis=-1)))
    logging.info("min reqjv",np.min(np.linalg.norm(reqjv,axis=-1)),'min reqjele',np.min(np.linalg.norm(reqjele,axis=-1)))
    logging.info("max reqjv",np.max(np.linalg.norm(reqjv,axis=-1)),'max reqjele',np.max(np.linalg.norm(reqjele,axis=-1)))

    return np.swapaxes(everror, 0,1)


# Helper function (external) for drawing on existing panel
def cavitons(ax, XmeshXY,YmeshXY, extmaps, requestvariables=False):
    if requestvariables==True:
        return ['rho', 'B', 'beta']

    # Check if extmaps has multiple time steps or just one
    if type(extmaps) is list:
        dsteps = [x['dstep'] for x in extmaps]
        curri = dsteps.index(0)
        rho = extmaps[curri]['rho']
        beta = extmaps[curri]['beta']
        # take magnitude of B
        B = np.linalg.norm(extmaps[curri]['B'],axis=-1)
    else:
        rho = extmaps['rho']
        beta = extmaps['beta']
        # take magnitude of B
        B = np.linalg.norm(extmaps['B'],axis=-1)

    # Colours to use
    color_cavitons = 'yellow'#'#924900'
    color_SHFAs    = 'k'#'#B66DFF'
    color_BS       = 'white'#'#FFFF6D'

    # thresholds
    level_bow_shock = 2.e+6
    level_n_caviton = 4.e+6
    level_B_caviton = 4.e-9
    level_beta_SHFA = 150
    level_beta_SHFA_SW = 10.

    # mask cavitons
    cavitons = np.ma.masked_greater_equal(B,level_B_caviton)
    cavitons.mask[rho > level_n_caviton] = True
    cavitons.fill_value = 0.
    cavitons[cavitons.mask == False] = 1.

    logging.info("cavitons: " + str(cavitons.sum()))

    # mask SHFAs
    SHFAs = np.ma.masked_greater_equal(B,level_B_caviton)
    SHFAs.mask[rho > level_n_caviton] = True
    SHFAs.mask[beta < level_beta_SHFA_SW] = True
    SHFAs.fill_value = 0.
    SHFAs[SHFAs.mask == False] = 1.
    logging.info("SHFA: " + str(SHFAs.sum()))
    # draw contours
    contour_shock = ax.contour(XmeshXY,YmeshXY,rho,[level_bow_shock],
                               linewidths=1.2, colors=color_BS,label='Bow shock')
    contour_cavitons = ax.contour(XmeshXY,YmeshXY,cavitons.filled(),[0.5], linewidths=1.5, colors=color_cavitons)
    contour_SHFAs = ax.contour(XmeshXY,YmeshXY,SHFAs.filled(),[0.5], linewidths=1.5, colors=color_SHFAs)

def expr_numberdensitycheck(pass_maps, requestvariables=False):
    # Calculates the required current density to support the observed magnetic field.
    # Then calculates the required electron flow velocity to support that current.
    # Then compares the current electron flow to that required.
    if requestvariables==True:
        return ['electron/vg_rho','proton/vg_rho']
    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronflow expected a single timestep, but got multiple. Exiting.")
        quit()

    return pass_maps['electron/vg_rho']-pass_maps['proton/vg_rho']

def expr_electronpressure_isothermal(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['rho','temperature']

    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronpressure_isothermal expected a single timestep, but got multiple. Exiting.")
        quit()

    #rho_e = pass_maps['electron/rho'].T
    rho_p = np.ma.masked_less_equal(pass_maps['rho'].T,0) # assumes equal number density for e,p
    temp_p = pass_maps['temperature'].T

    # This version assumes isothermal electrons with upstream beta_i = beta_p
    # i.e. pressure_i = pressure_p, scalar pressure
    # pressure=n 1.0/3.0 * np.ma.sum(PTensorDiagonal,axis=-1)
    # beta= 2.0 * mu_0 * np.ma.divide(Pressure, np.sum(np.asarray(Magneticfield)**2,axis=-1))
    # temperature = pressure/(rho * kb)
    nx,ny = rho_p.shape

    # electron pressure gradient contribution to electric field:
    # E = - nabla dot P_e / (n_e e)
    # pressure = temperature*(rho*kb) = T_0 * n_e * kb
    upstream_x = nx-2
    upstream_y = ny//2
    T_0 = temp_p[upstream_x,upstream_y]
    logging.info("upstream temperature " + str(T_0))
    # E = - (T_0 * kb)/(n_e * e) * nabla dot n_e
    mult = - T_0 * kb / elementalcharge
    gradrho = numgradscalar(rho_p)
    result = mult * np.ma.divide(gradrho, rho_p[:,:,np.newaxis])
    for i in range(3):
        result.mask[:,:,i] = 1.*rho_p.mask
    # expand mask twice to get rid of ionospheric values
    for i in range(2):
        result = expandMask(result)
    result = np.ma.filled(result, 0)
    return np.ma.swapaxes(result,0,1)

def expr_electronpressure_polytropic(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['rho','pressure']

    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronpressure_polytropic expected a single timestep, but got multiple. Exiting.")
        quit()

    #index (5/3 is adiabatic), 1 is isothermal
    index=5./3.
    #index=1
    # pn^-index = const

    #rho_e = pass_maps['electron/rho'].T
    rho_p = np.ma.masked_less_equal(pass_maps['rho'].T,0) # assumes equal number density for e,p
    pres_p = pass_maps['pressure'].T

    # This version assumes polytropic electrons with upstream beta_i = beta_p
    # i.e. pressure_i = pressure_p, scalar pressure
    # pressure=n 1.0/3.0 * np.ma.sum(PTensorDiagonal,axis=-1)
    # beta= 2.0 * mu_0 * np.ma.divide(Pressure, np.sum(np.asarray(Magneticfield)**2,axis=-1))
    # temperature = pressure/(rho * kb)
    # i.e. upstream T_0 is the same
    nx,ny = rho_p.shape

    # electron pressure gradient contribution to electric field:
    # E = - nabla dot P_e / (n_e e)
    # pressure = temperature*(rho*kb) = T_0 * n_e * kb
    upstream_x = nx-2
    upstream_y = ny//2
    P_0 = pres_p[upstream_x,upstream_y]
    n_0 = rho_p[upstream_x,upstream_y]
    logging.info("upstream pressure " + str(P_0) + " upstream density " + str(n_0))
    # find the constant using upstream values
    const = P_0 * np.power(n_0, -index)
    # Now the electron pressure is const*n^index
    pres_e = const * np.power(rho_p, index)
    gradpres = numgradscalar(pres_e)
    result = - np.ma.divide(gradpres,rho_p[:,:,np.newaxis])/elementalcharge
    for i in range(3):
        result.mask[:,:,i] = 1.*rho_p.mask
    # expand mask twice to get rid of ionospheric values
    for i in range(2):
        result = expandMask(result)
    result = np.ma.filled(result, 0)
    return np.ma.swapaxes(result,0,1)

def expr_electronpressure_ratio(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['rho','temperature','e']

    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronpressure_ratio expected a single timestep, but got multiple. Exiting.")
        quit()

    egradpe = expr_electronpressure_isothermal(pass_maps)
    egradpe = np.ma.masked_less_equal(np.linalg.norm(egradpe,axis=-1),0)
    e = np.linalg.norm(pass_maps['e'], axis=-1)
    e = np.ma.masked_less_equal(e,0)
    for i in range(2):
        e = expandMask(e)
    return np.ma.divide(egradpe, e)

def expr_electronpressure_ratioHall(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['rho','temperature','e','b','v']

    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronpressure_ratio expected a single timestep, but got multiple. Exiting.")
        quit()

    egradpe = expr_electronpressure_isothermal(pass_maps)
    egradpe = np.ma.masked_less_equal(np.linalg.norm(egradpe,axis=-1),0)
    e = pass_maps['e']
    b = pass_maps['b']
    v = pass_maps['v']
    emot=-np.cross(b,v)
    ehall = e - emot
    emag= np.linalg.norm(ehall, axis=-1)
    emag = np.ma.masked_less_equal(emag,0)

    for i in range(2):
        emag = expandMask(emag)
    return np.ma.divide(egradpe, emag)

def expr_electronpressure_check(pass_maps, requestvariables=False):
    if requestvariables==True:
        return ['rho','temperature','pressure']

    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expr_electronpressure_ratio expected a single timestep, but got multiple. Exiting.")
        quit()

    egradpe = expr_electronpressure_isothermal(pass_maps)
    egradpe2 = expr_electronpressure_polytropic(pass_maps)
    egradpe = np.ma.masked_less_equal(np.linalg.norm(egradpe,axis=-1),0)
    egradpe2 = np.ma.masked_less_equal(np.linalg.norm(egradpe2,axis=-1),0)
    for i in range(2):
        egradpe = expandMask(egradpe)
    return np.ma.divide(egradpe, egradpe2)


def expr_fgvgbvol(pass_maps, requestvariables=False):
    # expression for verifying gridglue: downsamples (aggregates) fsgrid values to dccrg-grid resolution
    if requestvariables==True:
        return ['vg_b_vol','fg_b_vol','fg_boundarytype','3d']

    # Verify that time averaging wasn't used
    if type(pass_maps) is list:
        logging.info("expression expected a single timestep, but got multiple. Exiting.")
        quit()

    fg = np.ma.masked_invalid(pass_maps['fg_b_vol'])
    # use boundarytype to mask which cells participate in aggregation
    fg_boundarytype = np.ma.masked_values(pass_maps['fg_boundarytype'], 0)
    fg.mask[:,:,:,0] = fg_boundarytype.mask[:,:,:]
    fg.mask[:,:,:,1] = fg_boundarytype.mask[:,:,:]
    fg.mask[:,:,:,2] = fg_boundarytype.mask[:,:,:]
    # generate unity array for counting aggregation source cells
    sums = np.ma.masked_invalid(np.zeros_like(fg)+1)
    sums.mask[:,:,:,0] = fg_boundarytype.mask[:,:,:]
    sums.mask[:,:,:,1] = fg_boundarytype.mask[:,:,:]
    sums.mask[:,:,:,2] = fg_boundarytype.mask[:,:,:]

    vg = pass_maps['vg_b_vol']
    bin_size=2
    (dim1,dim2,dim3,vect1)=fg.shape
    (out1,out2,out3,vect2)=vg.shape
    # reshape, then sum over the new mini-dimensions
    fgavg = fg.reshape((out1//bin_size, bin_size,
                        out2//bin_size, bin_size,
                        out3//bin_size, bin_size, 3))
    fgavg = np.ma.sum(fgavg, axis=1)
    fgavg = np.ma.sum(fgavg, axis=2)
    fgavg = np.ma.sum(fgavg, axis=3)
    # reshape, then sum over the new mini-dimensions
    summs = sums.reshape((out1//bin_size, bin_size,
                        out2//bin_size, bin_size,
                        out3//bin_size, bin_size, 3))
    summs = np.ma.sum(summs, axis=1)
    summs = np.ma.sum(summs, axis=2)
    summs = np.ma.sum(summs, axis=3)
    # divide by sum count to get average
    summs = np.ma.masked_values(summs,0)
    fgavg = np.ma.divide(fgavg,summs)
    # resample back up for plotting
    fgreturn = np.repeat(np.repeat(np.repeat(fgavg, 2, axis=0), 2, axis=1), 2, axis=2)
    return np.abs(fgreturn-vg)


# Contour plot spatial AMR refinement levels
def reflevels(ax, XmeshXY,YmeshXY, extmaps, requestvariables=False):
    if requestvariables==True:
        return ['vg_reflevel']

    # Check if pass_maps has multiple time steps or just one
    if type(extmaps) is list:
        dsteps = [x['dstep'] for x in extmaps]
        curri = dsteps.index(0)
        ref = extmaps[curri]['vg_reflevel']
    else:
        ref = extmaps['vg_reflevel']

    minref = int(np.amin(ref))
    maxref = int(np.amax(ref))
    levels = np.arange(minref,maxref)+0.5
    ax.contour(XmeshXY, YmeshXY, ref, levels, antialiased=False, linewidths=0.1, cmap='gray')

# Contour plot MPI ranks
def ranks(ax, XmeshXY,YmeshXY, extmaps, requestvariables=False):
    if requestvariables==True:
        return ['vg_rank']

    # Check if pass_maps has multiple time steps or just one
    if type(extmaps) is list:
        dsteps = [x['dstep'] for x in extmaps]
        curri = dsteps.index(0)
        rank = extmaps[curri]['vg_rank']
    else:
        rank = extmaps['vg_rank']

    minrank = int(np.amin(rank))
    maxrank = int(np.amax(rank))
    levels = np.arange(minrank,maxrank)+0.5
    ax.contour(XmeshXY, YmeshXY, rank, levels, antialiased=False, linewidths=0.1, cmap='gray')

