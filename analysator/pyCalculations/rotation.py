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

import numpy as np
import logging


def rotateTensorToVector( Tensor, vector ):
   '''
      Rotates a tensor with a rotation matrix that would align vector with the z-axis (E.g. moves Tensor to a coordinate system where z axis points in the same direction as vector

      :param Tensor:          Tensor to be rotated
      :param vector:          Vector for creating the rotation matrix
      :returns: rotated tensor
   '''
   vector_u = np.cross(vector, np.array([0,0,1]))

   if np.linalg.norm(vector_u) == 0.0:
      return Tensor

   vector_u = vector_u / np.linalg.norm(vector_u)
   angle = np.arccos( vector.dot(np.array([0,0,1])) / np.linalg.norm(vector) )
   # A unit vector version of the given vector
   R = rotation_matrix( vector_u, angle )
   # Rotate Tensor
   Tensor_rotated = R.dot(Tensor).dot(R.transpose())
   return Tensor_rotated

def rotateArrayTensorToVector( Tensor, vector ):
   '''
      Rotates an array of tensors with rotation matrices that would align the vectors with the z-axis

      :param Tensor:          n x 3 x 3 element array (n  3x3 Tensors to be rotated)
      :param vector:          n x 3 element array (n Vectors for creating the rotation matrix)
      :returns: n x 3 x 3 element array of rotated tensors
   '''
   vector_u = np.cross(vector, np.array([0.,0.,1.])[np.newaxis,:])
   vector_u_len = np.ma.masked_less_equal(np.linalg.norm(vector_u, axis=-1), 0)
   vector_u = np.ma.divide(vector_u, vector_u_len[:,np.newaxis])
   vector_len = np.ma.masked_less_equal(np.linalg.norm(vector, axis=-1), 0)
   vector_n = np.ma.divide(vector,vector_len[:,np.newaxis])
   angle = np.arccos(vector_n[:,2])
   # A unit vector version of the given vector
   R = rotation_array_matrix( vector_u, angle )
   # Rotate Tensor
   # matmul requires numpy 1.10 or newer. Can use einsum instead.
   #Tensor_rotated = np.matmul(np.matmul(R,Tensor),np.transpose(R,(0,2,1)) )
   Tensor_rotated = np.einsum('...ij,...jk', np.einsum('...ij,...jk',R,Tensor),np.transpose(R,(0,2,1)) )
   return Tensor_rotated

def rotateVectorToVector( vector1, vector2 ):
   ''' Applies rotation matrix that would rotate vector2 to z-axis on vector1 and then returns the rotated vector1

       :param vector1        Vector to be rotated
       :param vector2        Vector for creating the rotation matrix
       :returns rotated vector1 vector

       .. note::

          vector1 and vector2 must be 3d vectors
   '''
   vector_u = np.cross(vector2, np.array([0,0,1]))
   if np.linalg.norm(vector_u) == 0.0:
      return vector1
   else:
      vector_u = vector_u / np.linalg.norm(vector_u)
      angle = np.arccos( vector2.dot(np.array([0,0,1])) / np.linalg.norm(vector2) )
      # A unit vector version of the given vector
      R = rotation_matrix( vector_u, angle )
      # Rotate vector
      vector_rotated = R.dot(vector1.transpose()).transpose()
      return vector_rotated

def rotateVectorToVector_X( vector1, vector2 ):
   ''' Applies rotation matrix that would rotate vector2 to x-axis on vector1 and then returns the rotated vector1

       :param vector1        Vector to be rotated
       :param vector2        Vector for creating the rotation matrix
       :returns rotated vector1 vector

       .. note::

          vector1 and vector2 must be 3d vectors
   '''
   vector_u = np.cross(vector2, np.array([1,0,0]))
   if np.linalg.norm(vector_u) == 0.0:
      return vector1
   else:
      vector_u = vector_u / np.linalg.norm(vector_u)
      angle = np.arccos( vector2.dot(np.array([1,0,0])) / np.linalg.norm(vector2) )
      # A unit vector version of the given vector
      R = rotation_matrix( vector_u, angle )
      # Rotate vector
      vector_rotated = R.dot(vector1.transpose()).transpose()
      return vector_rotated

def rotation_matrix(vector, angle):
   ''' Creates a rotation matrix that rotates around a given vector by a given angle
       :param vector        Some unit vector
       :param angle         Some angle
       :returns a rotation matrix
   '''
   v = vector
   t = angle
   cost = np.cos(t)
   sint = np.sin(t)
   unitymcost = -cost+1
   m = np.array([[cost+v[0]**2*unitymcost,
                  v[0]*v[1]*unitymcost-v[2]*sint, 
                  v[0]*v[2]*unitymcost+v[1]*sint],
                 [v[0]*v[1]*unitymcost+v[2]*sint, 
                  cost+v[1]**2*unitymcost, 
                  v[1]*v[2]*unitymcost-v[0]*sint],
                 [v[0]*v[2]*unitymcost-v[1]*sint, 
                  v[2]*v[1]*unitymcost+v[0]*sint, 
                  cost+v[2]**2*unitymcost]])
   return m


def rotation_array_matrix(vector, angle):
   ''' Creates a rotation matrix that rotates around a given vector by a given angle
       :param vector        n x 3 array of unit vectors
       :param angle         array of angles
       :returns a rotation matrix
   '''
   v = vector
   t = angle
   cost = np.cos(t)
   sint = np.sin(t)
   unitymcost = -cost+1
   m = np.array([[cost+v[:,0]**2*unitymcost, 
                  v[:,0]*v[:,1]*unitymcost-v[:,2]*sint, 
                  v[:,0]*v[:,2]*unitymcost+v[:,1]*sint],
                 [v[:,0]*v[:,1]*unitymcost+v[:,2]*sint, 
                  cost+v[:,1]**2*unitymcost, 
                  v[:,1]*v[:,2]*unitymcost-v[:,0]*sint],
                 [v[:,0]*v[:,2]*unitymcost-v[:,1]*sint, 
                  v[:,2]*v[:,1]*unitymcost+v[:,0]*sint, 
                  cost+v[:,2]**2*unitymcost]])
   return np.transpose(m,(2,0,1))
