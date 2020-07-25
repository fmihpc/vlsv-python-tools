#!/usr/bin/env python
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

 

def write_vtk_file( filename, point_data ):
   ''' Writes a line into a VTK file with given points

       :param filename:      Name of the file e.g. "test.vtk"
       :param points:        Points in a line in array format

       .. code-block:: python

          # Example usage:
          import analysator as pt
          filename = "test.vtk"
          point_data = [[0.,0.,0.], [1.,1.,1.], [2.,2.,2.], [3.,3.,3.], [4.,4.,4.]]
          pt.miscellaneous.write_vtk_file( filename=filename, point_data=point_data )

   '''
   import vtk
   
   # Create a line object
   #######################################
   # Create a vtkPoints object and store the points in it
   points = vtk.vtkPoints()
   for point in point_data:
      points.InsertNextPoint(point)
   
   # Create a cell array to store the lines in and add the lines to it
   lines = vtk.vtkCellArray()
   
   for i in range(len(point_data)-1):
     line = vtk.vtkLine()
     line.GetPointIds().SetId(0,i)
     line.GetPointIds().SetId(1,i+1)
     lines.InsertNextCell(line)
   
   # Create a polydata to store everything in
   linesPolyData = vtk.vtkPolyData()
   
   # Add the points to the dataset
   linesPolyData.SetPoints(points)
   
   # Add the lines to the dataset
   linesPolyData.SetLines(lines)
   #######################################
   
   
   # Write the Line object
   #######################################
   polyDataWriter = vtk.vtkPolyDataWriter()
   polyDataWriter.SetFileName(filename)
   polyDataWriter.SetInput(linesPolyData)
   polyDataWriter.Write()
   #######################################

#   # Read and visualize the written line
#   #######################################
#   reader = vtk.vtkPolyDataReader()
#   reader.SetFileName(filename)
#    
#   mapper = vtk.vtkPolyDataMapper()
#   if vtk.VTK_MAJOR_VERSION <= 5:
#       mapper.SetInput(reader.GetOutput())
#   else:
#       mapper.SetInputConnection(reader.GetOutputPort())
#    
#   actor = vtk.vtkActor()
#   actor.SetMapper(mapper)
#   
#   # Create a rendering window and renderer
#   ren = vtk.vtkRenderer()
#   renWin = vtk.vtkRenderWindow()
#   renWin.AddRenderer(ren)
#    
#   # Create a renderwindowinteractor
#   iren = vtk.vtkRenderWindowInteractor()
#   iren.SetRenderWindow(renWin)
#    
#   # Assign actor to the renderer
#   ren.AddActor(actor)
#    
#   # Enable user interface interactor
#   iren.Initialize()
#   renWin.Render()
#   iren.Start()
#   #######################################
