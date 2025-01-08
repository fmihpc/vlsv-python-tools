#!/usr/bin/python
#
import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from shapely import geometry
from numpy import linalg as LA
import logging

## This script searches for the x and o points from the 2D simulations. It assumes polar plane. If you use equatorial plane change the z_array to y_array and it's limits.
## It uses the contours of grad(flux_function) to find the extrema and Hessian matrix to define the type of the point (minima, maxima, saddle)
## NOTE that for general use it defines the maxima as the o point. 
## If you want to find the o-points, when they are the minima (e.g. BCV magnetopause reconnection), you have to change the sign, where the points are defined

## just remember to set the paths and file names to you liking
## Have fun! 

## HERE DEFINE run ID
run_id= "BFB"

# Naming of flux files?
flux_file= "bulk."  # BCH=bulk, BCQ=flux, BCV=bulk
if run_id == "BCQ":
   flux_file= "flux." 


## Where can we find the fluxfunction files?
flux_path = "/wrk/group/spacephysics/vlasiator/2D/"+run_id+"/flux/"+flux_file
flux_suffix = ".bin"
## We need one bulk file as well:
bulk_path = "/wrk/group/spacephysics/vlasiator/2D/"+run_id+"/bulk/bulk."
bulk_suffix = ".vlsv"
bulkfile = bulk_path + str(1958).zfill(7) + bulk_suffix

## GIVE PATH WHERE YOU WANT TO SAVE THE POINTS ##
path_to_save='/wrk/group/spacephysics/vlasiator/2D/'+run_id+'/visualization/x_and_o_points/'

RE=6371000

# FUNCTION FOR FINDING THE INTERSECTION POINTS OF TWO CONTOURS
def findIntersection(v1,v2):
  # modified from https://stackoverflow.com/questions/42838190/finding-intersection-of-two-contour-plots-in-python
  #p1 = contour1.collections[0].get_paths()[0]
  #v1 = p1.vertices

  #p2 = contour2.collections[0].get_paths()[0]
  #v2 = p2.vertices
  poly1 = geometry.LineString(v1)
  poly2 = geometry.LineString(v2)

  intersection = poly1.intersection(poly2)
  return intersection

## Which frames to calculate points for?

# for looping over many files
if len(sys.argv)==2:
   indexes = [int(sys.argv[1])]
elif len(sys.argv)==3:
   indexes = np.arange(int(sys.argv[1]),int(sys.argv[2]))
elif len(sys.argv)==4:
   indexes = np.arange(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
else:
   sys.stderr.write("Syntax: size.py <index>\n")
   sys.stderr.write("or: size.py <starting_index> <final_index+1>\n")
   sys.stderr.write("or: size.py <starting_index> <final_index+1>  <index_step> \n")
   sys.exit()

## Open bulkfile, determine sizes ##
vlsvfile = pt.vlsvfile.VlsvReader(bulkfile);
x_cells=int(vlsvfile.get_spatial_mesh_size()[0])
z_cells=int(vlsvfile.get_spatial_mesh_size()[2])
xsize = vlsvfile.read_parameter("xcells_ini")
xmax =  vlsvfile.read_parameter("xmax")
xmin =  vlsvfile.read_parameter("xmin")
zmin =  vlsvfile.read_parameter("zmin")
zmax =  vlsvfile.read_parameter("zmax")
dx = (xmax-xmin)/xsize 

## DEFINE ARRAYS FOR AXIS
x_array=np.array(range(int(xmin), int(xmax), int(dx)))
z_array=np.array(range(int(zmin), int(zmax), int(dx)))

for index in indexes:
   fluxfile = flux_path + str(index).zfill(7) + flux_suffix

   # Open input fluxfile
   flux_function = np.fromfile(fluxfile,dtype='double').reshape(z_cells,x_cells)
   flux_offset = float(index)*0.3535*5e-9*(-7.5e5)

   # Smooth fluxfunction
   kernel_size=5
   fkernel = np.ones((kernel_size,kernel_size))/(kernel_size**2)
   raw_flux_function = flux_function
   flux_function= convolve2d(flux_function, fkernel, 'same')

   # calculate gradient of flux function
   dfdx,dfdz=np.gradient(flux_function)

   #calculate the 0 contours of df/dx and df/dz
   plt.figure(1)
   contour1=plt.contour(x_array,z_array, dfdx, [0])
   contour1_paths=contour1.collections[0].get_paths()
   contour2=plt.contour(x_array,z_array, dfdz, [0])
   contour2_paths=contour2.collections[0].get_paths()

   x_coords=[]
   z_coords=[]

   # find the intersection points of the 
   for path1 in contour1_paths:
      for path2 in contour2_paths:
         if path1.intersects_path(path2) and len(path1)>1 and len(path2)>1:
            intersection=findIntersection(path1.vertices,path2.vertices)
            intersection_points=np.asarray(intersection)
            if len(intersection_points)>0:
               if len(intersection_points.shape)==1:
                  x_coords.append(intersection_points[0])
                  z_coords.append(intersection_points[1])
               else:
                  for i in range(len(intersection_points[:,0])):
                     x_coords.append(intersection_points[i,0])
                     z_coords.append(intersection_points[i,1])


   # DEFINE the type of the gradient(flux)=0 ##

   x_point_location=[]
   o_point_location=[]
   x_point_fluxes=[]
   o_point_fluxes=[]
   minima_location=[]
   flux_function=flux_function.T

   for k in range(len(x_coords)):
      #cellid = 1+i+j*x_cells
      coords=[x_coords[k],0,z_coords[k]]
      cellid=vlsvfile.get_cellid(coords)
      i=int((cellid-1)%x_cells)
      j=(int(cellid)-1)//x_cells

      difference=[]

      ## the limist for i and j have the value 100, to save time, I have not been interested about x's and o's within 100 cell from boundaries.
      ## If you are then change the limits
      if i > 100 and j > 100 and i < x_cells-100 and j < z_cells-100:


         # Hessian matrix using central difference formulas for the second partial derivatives
         deltaPsi_xx= (flux_function[i+1,j]-2*flux_function[i,j]+flux_function[i-1,j])/dx**2
         deltaPsi_zz= (flux_function[i,j+1]-2*flux_function[i,j]+flux_function[i,j-1])/dx**2
         deltaPsi_xz= (flux_function[i+1,j+1]-flux_function[i+1,j-1]-flux_function[i-1,j+1]+flux_function[i-1,j-1])/(4*dx**2)

         Hessian = [[deltaPsi_xx, deltaPsi_xz], [deltaPsi_xz, deltaPsi_zz]]
         DetHess = deltaPsi_xx*deltaPsi_zz-deltaPsi_xz*deltaPsi_xz
         eigvals, eigvectors = LA.eig(Hessian)
         #coords.append(DetHess)
         #if sign_changes == 4:
         #if eigvals[0]*eigvals[1] <0 :

         # Calculate interpolated flux function value
         i_i = int(coords[0]/dx)
         i_f = coords[0]/dx - i_i
         j_i = int(coords[1]/dx)
         j_f = coords[1]
         interpolated_flux = (1.-j_f) * ((1. - i_f) * flux_function[i,j] + i_f * flux_function[i+1,j]) + j_f* ((1. - i_f) * flux_function[i,j+1] + i_f * flux_function[i+1,j+1])

         if DetHess < 0:
            x_point_location.append(coords)           
            x_point_fluxes.append(interpolated_flux)

         ## NOTE if you want the o-points to be local maxima use deltaPsi_xx < 0, if you want them to be local minima use  deltaPsi_xx > 0                  
         #if DetHess > 0 and deltaPsi_xx > 0:
         #   minima_location.append(coords)
         if DetHess > 0 and deltaPsi_xx < 0:
            o_point_location.append(coords)
            o_point_fluxes.append(interpolated_flux)

   plt.close('all')

   np.savetxt(path_to_save+"/o_point_location_"+str(index)+".txt", o_point_location)
   np.savetxt(path_to_save+"/o_point_location_and_fluxes_"+str(index)+".txt", np.concatenate( (o_point_location,np.array(o_point_fluxes)[:,np.newaxis]), axis=1))
   np.savetxt(path_to_save+"/x_point_location_"+str(index)+".txt", x_point_location)
   np.savetxt(path_to_save+"/x_point_location_and_fluxes_"+str(index)+".txt", np.concatenate( (x_point_location,np.array(x_point_fluxes)[:,np.newaxis]), axis=1))

