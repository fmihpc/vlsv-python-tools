from traits.api import HasTraits, Instance, Property, Button, Enum
from mayavi.core.ui.engine_view import EngineView
from traits.api import HasTraits, Range, Instance, \
                    on_trait_change
from traitsui.api import View, Item, HGroup, Group
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
                    MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
import vlsvfile
from numpy import mgrid, empty, sin, pi, ravel
import pylab as pl
from tvtk.api import tvtk
import traits.api
import mayavi.api
import mayavi.mlab
import numpy as np
import signal
import threading
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.outline import Outline
from mayavi.modules.surface import Surface
from mayavi.modules.vectors import Vectors

#Catch SIGINT as mayavi (VTK) has disabled the normal signal handler
def SigHandler(SIG, FRM):
    print "Ctrl+C"
    return
signal.signal(signal.SIGINT, SigHandler)

class MayaviGrid(HasTraits):
   ''' This class is used to plot the data in a vlsv file as a mayavi grid The following will bring up a new window and plot the grid in the vlsv file:

   .. code-block:: python

      grid = pt.grid.MayaviGrid(vlsvReader=f, variable="rho", operator='pass', threaded=False)

   Once you have the window open you can use the picker tool in the right-upper corner and use various point-click tools for analyzing data.

   Picker options:
   
   **None** Does nothing upon clicking somewhere in the grid
   
   **Velocity_space** Plots the velocity space at a specific position upon clicking somewhere in the grid Note: If the vlsv file does not have the velocity space at the position where you are clicking, this will not work
   
   **Velocity_space_iso_surface** Plots the velocity space at a specific position upon clicking somewhere in the grid in iso-surface plotting style Note: If the vlsv file does not have the velocity space at the position where you are clicking, this will not work
   
   **Velocity_space_nearest_cellid** Plots the velocity space of the closest cell id to the picking point Note: If the vlsv file does not have velocity space saved at all, this will not work
   
   **Velocity_space_nearest_cellid_iso_surface** Plots the velocity space of the closest cell id to the picking point in iso-surface plotting style Note: If the vlsv file does not have velocity space saved at all, this will not work
   
   **Pitch_angle** Plots the pitch angle distribution at the clicking position Note: If the vlsv file does not have the velocity space at the position where you are clicking, this will not work
   
   **Cut_through** Is used to plot or save the cut-through between two clicking points. This option requires you to use the args section at top-left. To use the args section to plot variables you must write for example: **plot rho B,x E,y** Upon clicking at two points a new window would open with a cut-through plot of rho, x-component of B and y-component of E Alternatively, you can save the cut-through to a variable in the MayaviGrid class by typing instead: **rho B,x E,y** and then going to the terminal and typing
   
   .. code-block:: python

      cut_through_data = grid.cut_through
      print cut_through_data

   '''
   picker = Enum('None',
                 'Velocity_space',
                 "Velocity_space_nearest_cellid",
                 'Velocity_space_iso_surface',
                 'Velocity_space_nearest_cellid_iso_surface',
                 "Pitch_angle",
                 "Cut_through")

   args = ""

   cut_through = []

   scene = Instance(MlabSceneModel, ())

   engine_view = Instance(EngineView)

   current_selection = Property

   # Define the view:
   view = View(
                HGroup(
                   Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                      height=250, width=300, show_label=False, resizable=True),
                   Group(
                      #'cell_pick',
                      'picker',
                      'args',
                      show_labels=True
                   ),
                ),
                resizable=True,
            )


   def __init__(self, vlsvReader, variable, operator="pass", threaded=True, **traits):
      ''' Initializes the class and loads the mayavi grid

          :param vlsvReader:        Some vlsv reader with a file open
          :type vlsvReader:         :class:`vlsvfile.VlsvReader`
          :param variable:          Name of the variable
          :param operator:          Operator for the variable
          :param threaded:          Boolean value for using threads or not using threads to draw the grid (threads enable interactive mode)
      '''
      HasTraits.__init__(self, **traits)
      self.__vlsvReader = vlsvReader
      self.engine_view = EngineView(engine=self.scene.engine)
      self.__engine = self.scene.engine
      self.__picker = []
      self.__mins = []
      self.__maxs = []
      self.__last_pick = []
      self.__structured_figures = []
      self.__unstructured_figures = []
      self.__thread = []
      self.__load_grid( variable=variable, operator=operator, threaded=threaded )

   def __load_grid( self, variable, operator="pass", threaded=True ):
      ''' Creates a grid and inputs scalar variables from a vlsv file
          :param variable:        Name of the variable to plot
          :param operator:        Operator for the variable
          :param threaded:        Boolean value for using threads or not using threads to draw the grid (threads enable interactive mode)
      '''
      # Get the cell params:
      mins = np.array([self.__vlsvReader.read_parameter("xmin"), self.__vlsvReader.read_parameter("ymin"), self.__vlsvReader.read_parameter("zmin")])
      cells = np.array([self.__vlsvReader.read_parameter("xcells_ini"), self.__vlsvReader.read_parameter("ycells_ini"), self.__vlsvReader.read_parameter("zcells_ini")])
      maxs = np.array([self.__vlsvReader.read_parameter("xmax"), self.__vlsvReader.read_parameter("ymax"), self.__vlsvReader.read_parameter("zmax")])
      # Get the variables:
      index_for_cellid_dict = self.__vlsvReader.get_cellid_locations()
      variable_array = self.__vlsvReader.read_variable( name=variable, operator=operator )
      # Sort the dictionary by cell id
      import operator as oper
      sorted_index_for_cellid_dict = sorted(index_for_cellid_dict.iteritems(), key=oper.itemgetter(0))
      # Add the variable values:
      variable_array_sorted = []
      for i in sorted_index_for_cellid_dict:
         variable_array_sorted.append(variable_array[i[1]])
      # Store the mins and maxs:
      self.__mins = mins
      self.__maxs = maxs
      # Draw the grid:
      if threaded == True:
         thread = threading.Thread(target=self.__generate_grid, args=( mins, maxs, cells, variable_array_sorted, variable ))
         thread.start()
      else:
         self.__generate_grid( mins=mins, maxs=maxs, cells=cells, datas=variable_array_sorted, names=variable )

   def __picker_callback( self, picker ):
      """ This gets called when clicking on a cell
      """
      if (self.picker != "Cut_through"):
         # Make sure the last pick is null (used in cut_through)
         self.__last_pick = []

      coordinates = picker.pick_position
      coordinates = np.array([coordinates[0], coordinates[1], coordinates[2]])
      for i in xrange(3):
         if (coordinates[i] < self.__mins[i]) and (coordinates[i] + 15 > self.__mins[i]):
            # Correct the numberical inaccuracy
            coordinates[i] = self.__mins[i] + 1
         if (coordinates[i] > self.__maxs[i]) and (coordinates[i] - 15 < self.__maxs[i]):
            # Correct the values
            coordinates[i] = self.__maxs[i] - 1
      print "COORDINATES:" + str(coordinates)
      cellid = self.__vlsvReader.get_cellid(coordinates)
      print "CELL ID: " + str(cellid)
      # Check for an invalid cell id
      if cellid == 0:
         print "Invalid cell id"
         return

      if (self.picker == "Velocity_space"):
         self.__generate_velocity_grid(cellid)
      elif (self.picker == "Velocity_space_nearest_cellid"):
         # Find the nearest cell id with distribution:
         # Read cell ids with velocity distribution in:
         cell_candidates = self.__vlsvReader.read("SpatialGrid","CELLSWITHBLOCKS")
         # Read in the coordinates of the cells:
         cell_candidate_coordinates = [self.__vlsvReader.get_cell_coordinates(cell_candidate) for cell_candidate in cell_candidates]
         # Read in the cell's coordinates:
         pick_cell_coordinates = self.__vlsvReader.get_cell_coordinates(cellid)
         # Find the nearest:
         from operator import itemgetter
         norms = np.sum((cell_candidate_coordinates - pick_cell_coordinates)**2, axis=-1)**(1./2)
         norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
         # Get the cell id:
         cellid = cell_candidates[i]
         # Generate velocity grid
         self.__generate_velocity_grid(cellid)
      elif (self.picker == "Velocity_space_iso_surface"):
         self.__generate_velocity_grid(cellid, True)
      elif (self.picker == "Velocity_space_nearest_cellid_iso_surface"):
         # Find the nearest cell id with distribution:
         # Read cell ids with velocity distribution in:
         cell_candidates = self.__vlsvReader.read("SpatialGrid","CELLSWITHBLOCKS")
         # Read in the coordinates of the cells:
         cell_candidate_coordinates = [self.__vlsvReader.get_cell_coordinates(cell_candidate) for cell_candidate in cell_candidates]
         # Read in the cell's coordinates:
         pick_cell_coordinates = self.__vlsvReader.get_cell_coordinates(cellid)
         # Find the nearest:
         from operator import itemgetter
         norms = np.sum((cell_candidate_coordinates - pick_cell_coordinates)**2, axis=-1)**(1./2)
         norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
         # Get the cell id:
         cellid = cell_candidates[i]
         # Generate velocity grid
         self.__generate_velocity_grid(cellid, True)
      elif (self.picker == "Pitch_angle"):
         # Plot pitch angle distribution:
         from pitchangle import pitch_angles
         result = pitch_angles( vlsvReader=self.__vlsvReader, cellid=cellid, cosine=True, plasmaframe=True )
         # plot:
         pl.hist(result[0].data, weights=result[1].data, bins=50, log=False)
         pl.show()
      elif (self.picker == "Cut_through"):
         if len(self.__last_pick) == 3:
            from cutthrough import cut_through
            # Get a cut-through
            self.cut_through = cut_through( self.__vlsvReader, point1=self.__last_pick, point2=coordinates )
            # Get cell ids and distances separately
            cellids = self.cut_through[0].data
            distances = self.cut_through[1]
            # Get any arguments from the user:
            args = self.args.split()
            if len(args) == 0:
               #Do nothing
               print "Bad args"
               self.__last_pick = []
               return
            plotCut = False
            # Optimize file read:
            self.__vlsvReader.optimize_open_file()
            variables = []
            # Save variables
            for i in xrange(len(args)):
               # Check if the user has given the plot argument
               if args[i] == "plot":
                  plotCut = True
               else:
                  if args[i].find(",") != -1:
                     _variable = args[i].split(',')[0]
                     _operator = args[i].split(',')[1]
                     variable_info = self.__vlsvReader.read_variable_info( name=_variable, cellids=cellids, operator=_operator )
                     variables.append(variable_info)
                     self.cut_through.append(variable_info)
                  else:
                     variable_info = self.__vlsvReader.read_variable_info( name=args[i], cellids=cellids )
                     variables.append(variable_info)
                     self.cut_through.append(variable_info)
            if plotCut == True:
               from plot import plot_multiple_variables
               fig = plot_multiple_variables( [distances for i in xrange(len(args)-1)], variables, figure=[] )
               pl.show()
            # Close the optimized file read:
            self.__vlsvReader.optimize_close_file()
            # Read in the necessary variables:
            self.__last_pick = []
         else:
            self.__last_pick = coordinates

   
   def __generate_grid( self, mins, maxs, cells, datas, names  ):
      ''' Generates a grid from given data
          :param mins:           An array of minimum coordinates for the grid for ex. [-100, 0, 0]
          :param maxs:           An array of maximum coordinates for the grid for ex. [-100, 0, 0]
          :param cells:          An array of number of cells in x, y, z direction
          :param datas:          Scalar data for the grid e.g. array([ cell1Rho, cell2Rho, cell3Rho, cell4Rho, .., cellNRho ])
          :param names:          Name for the scalar data
      '''
      # Create nodes
      x, y, z = mgrid[mins[0]:maxs[0]:(cells[0]+1)*complex(0,1), mins[1]:maxs[1]:(cells[1]+1)*complex(0,1), mins[2]:maxs[2]:(cells[2]+1)*complex(0,1)]
      
      # Create points for the nodes:
      pts = empty(z.shape + (3,), dtype=float)
      pts[...,0] = x
      pts[...,1] = y
      pts[...,2] = z
      
      # Input scalars
      scalars = np.array(datas)
      
      # We reorder the points, scalars and vectors so this is as per VTK's
      # requirement of x first, y next and z last.
      pts = pts.transpose(2, 1, 0, 3).copy()
      pts.shape = pts.size/3, 3
      scalars = scalars.T.copy()
      
      # Create the dataset.
      sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
      sg.cell_data.scalars = ravel(scalars.copy())
      sg.cell_data.scalars.name = names
      
      
      # Visualize the data
      d = self.scene.mlab.pipeline.add_dataset(sg)
      iso = self.scene.mlab.pipeline.surface(d)

      # Configure traits
      self.configure_traits()

      # Note: This is not working properly -- it seemingly works out at first but it eventually causes segmentation faults in some places
      #self.__thread = threading.Thread(target=self.configure_traits, args=())
      #self.__thread.start()
      

   def __generate_velocity_grid( self, cellid, iso_surface=False ):
      '''Generates a velocity grid from a given spatial cell id
         :param cellid:           The spatial cell's ID
         :param iso_surface:      If true, plots the iso surface
      '''
      # Create nodes
      # Get velocity blocks and avgs:
      blocksAndAvgs = self.__vlsvReader.read_blocks(cellid)
      if len(blocksAndAvgs) == 0:
         print "CELL " + str(cellid) + " HAS NO VELOCITY BLOCK"
         return False
      # Create a new scene
      self.__engine.new_scene()
      mayavi.mlab.set_engine(self.__engine)#CONTINUE
      # Create a new figure
      figure = mayavi.mlab.gcf(engine=self.__engine)
      figure.scene.disable_render = True
      blocks = blocksAndAvgs[0]
      avgs = blocksAndAvgs[1]
      # Get nodes:
      nodesAndKeys = self.__vlsvReader.construct_velocity_cell_nodes(blocks)
      # Create an unstructured grid:
      points = nodesAndKeys[0]
      tets = nodesAndKeys[1]
      tet_type=tvtk.Voxel().cell_type#VTK_VOXEL

      ug=tvtk.UnstructuredGrid(points=points)
      # Set up the cells
      ug.set_cells(tet_type,tets)
      # Input data
      values=np.ravel(avgs)
      ug.cell_data.scalars=values
      ug.cell_data.scalars.name='avgs'

      # Plot B if possible:
      def plot_B( name ):
         ''' Helper function for plotting B vector (name can change from B_vol to B)
             :param name:         Name of the B vector ( "B_vol" or "B" )
         '''
         # Read B vector and plot it:
         B = self.__vlsvReader.read_variable(name=name,cellids=cellid)
         points2 = np.array([[0,0,0]])
         ug2 = tvtk.UnstructuredGrid(points=points2)
         ug2.point_data.vectors = [(B * 8000000000000) / np.linalg.norm( B )]
         ug2.point_data.vectors.name = 'B_vector'
         #src2 = VTKDataSource(data = ug2)
         d2 = mayavi.mlab.pipeline.add_dataset(ug2)
         #mayavi.mlab.add_module(Vectors())
         vec = mayavi.mlab.pipeline.vectors(d2)
         vec.glyph.mask_input_points = True
         vec.glyph.glyph.scale_factor = 100000

      if self.__vlsvReader.check_variable( "B" ) == True:
         plot_B( "B" )
      elif self.__vlsvReader.check_variable( "B_vol" ) == True:
         plot_B( "B_vol" )


      # Visualize
      d = mayavi.mlab.pipeline.add_dataset(ug)
      if iso_surface == False:
         iso = mayavi.mlab.pipeline.surface(d)
      else:
         ptdata = mayavi.mlab.pipeline.cell_to_point_data(d)
         iso = mayavi.mlab.pipeline.iso_surface(ptdata, contours=[1e-15,1e-14,1e-12], opacity=0.3)
      figure.scene.disable_render = False
      self.__unstructured_figures.append(figure)
      # Name the figure
      figure.name = str(cellid)
      return True

   def generate_diff_grid( self, cellid1, cellid2 ):
      ''' Generates a diff grid of given cell ids (shows avgs diff)

          :param cellid1:          The first cell id
          :param cellid2:          The second cell id

          .. code-block:: python

             # Example:
             grid.generate_diff_grid( 29219, 2910 )

          .. note:: If the cell id does not have a certain velocity cell, it is assumed that the avgs value of that cell is 0

      '''
      # Create nodes
      # Get velocity blocks and avgs (of cellid 1)
      blocksAndAvgs1 = self.__vlsvReader.read_blocks(cellid1)
      if len(blocksAndAvgs1) == 0:
         print "CELL " + str(cellid1) + " HAS NO VELOCITY BLOCK"
         return False
      blocks1 = blocksAndAvgs1[0]
      avgs1 = blocksAndAvgs1[1]

      # Get velocity blocks and avgs (of cellid 2)
      blocksAndAvgs2 = self.__vlsvReader.read_blocks(cellid2)
      if len(blocksAndAvgs2) == 0:
         print "CELL " + str(cellid2) + " HAS NO VELOCITY BLOCK"
         return False
      blocks2 = blocksAndAvgs2[0]
      avgs2 = blocksAndAvgs2[1]
      print len(avgs2)
      print len(blocks2)

      # Compare blocks and create a new avgs array values:
      avgs_same = []
      avgs_cellid1 = []
      avgs_cellid2 = []
      blocks_same = []
      blocks_cellid1 = []
      blocks_cellid2 = []
      print np.shape(avgs1[0])
      for i in xrange(len(blocks1)):
         b = blocks1[i]
         # Get index of block
         i2 = np.where(blocks2 == b)[0]
         if len(i2) != 0:
            # Fetch the block:
            #print avgs1[64*i:64*(i+1)]
            #print avgs2[64*i2[0]:64*(i2[0]+1)]
            avgs_same.append(avgs1[i:(i+1)] - avgs2[i2[0]:(i2[0]+1)])
            blocks_same.append(b)
         else:
            avgs_cellid1.append(avgs1[i:(i+1)])
            blocks_cellid1.append(b)
      for i in xrange(len(blocks2)):
         b = blocks2[i]
         if (b in blocks1) == False:
            avgs_cellid2.append(avgs2[i:(i+1)])
            blocks_cellid2.append(b)
      # Make a list for the avgs etc
      avgs = np.zeros(64*(len(avgs_same)+len(avgs_cellid1)+len(avgs_cellid2)))
      #avgs = np.reshape(avgs, (len(avgs_same)+len(avgs_cellid1)+len(avgs_cellid2), 64))
      print np.shape(avgs_same)
      blocks = np.zeros(len(blocks_same)+len(blocks_cellid1)+len(blocks_cellid2))

      index = 0
      avgs[64*index:64*(index + len(blocks_same))] = np.ravel(np.array(avgs_same))
      blocks[index:index + len(blocks_same)] = np.array(blocks_same)

      index = index + len(blocks_same)
      avgs[64*index:64*(index + len(blocks_cellid1))] = np.ravel(np.array(avgs_cellid1))
      blocks[index:index + len(blocks_cellid1)] = np.array(blocks_cellid1)

      index = index + len(blocks_cellid1)
      avgs[64*index:64*(index + len(blocks_cellid2))] = np.ravel(np.array(avgs_cellid2))
      blocks[index:index + len(blocks_cellid2)] = np.array(blocks_cellid2)

      blocks = blocks.astype(int)

      # Get nodes:
      nodesAndKeys = self.__vlsvReader.construct_velocity_cell_nodes(blocks)


      # Create an unstructured grid:
      points = nodesAndKeys[0]
      tets = nodesAndKeys[1]

      # Create a new scene
      self.__engine.new_scene()
      mayavi.mlab.set_engine(self.__engine)#CONTINUE
      # Create a new figure
      figure = mayavi.mlab.gcf(engine=self.__engine)
      figure.scene.disable_render = True
      tet_type=tvtk.Voxel().cell_type#VTK_VOXEL

      ug=tvtk.UnstructuredGrid(points=points)
      #Thissetsupthecells.
      ug.set_cells(tet_type,tets)
      #Attributedata.
      values=np.ravel(avgs)
      ug.cell_data.scalars=values
      ug.cell_data.scalars.name='avgs'
      d = mayavi.mlab.pipeline.add_dataset(ug)
      iso = mayavi.mlab.pipeline.surface(d)
      figure.scene.disable_render = False
      self.__unstructured_figures.append(figure)
      # Name the figure
      figure.name = str(cellid1) + " " + str(cellid2)
      mayavi.mlab.show()
      return True


   def __do_nothing( self, picker ):
      return

   # Trait events:
   @on_trait_change('scene.activated')
   def set_mouse_click( self ):
      # Temporary bug fix (MayaVi needs a dummy pick to be able to remove cells callbacks from picker.. )
      #self.figure.on_mouse_pick( self.__do_nothing, type='world' 
      self.figure = self.scene.mlab.gcf()
      # Cell picker
      func = self.__picker_callback
      typeid = 'world'
      click = 'Left'
      picker = self.figure.on_mouse_pick( func, type='world' )
      self.__picker = [func, typeid, click]
      #picker.tolerance = 0
      # Show legend bar
      manager = self.figure.children[0].children[0]
      manager.scalar_lut_manager.show_scalar_bar = True
      manager.scalar_lut_manager.show_legend = True

    # Not working properly, commenting this out
#   @on_trait_change('scene.closed')
#   def kill_thread( self ):
#      self.__thread.join()





