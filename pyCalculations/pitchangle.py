import numpy as np
import pylab as pl

def pitch_angles( vlsvReader, cellid, cosine=True, plasmaframe=False ):
   ''' Calculates the pitch angle distribution for a given cell
       :param vlsvReader        Some VlsvFile class with a file open
       :param cellid            The cell id whose pitch angle the user wants NOTE: The cell id must have a velocity distribution!
       :param cosine            True if returning the pitch angles as a cosine plot
       :param plasmaframe       True if the user wants to get the pitch angle distribution in the plasma frame
       :returns pitch angles and avgs [pitch_angles, avgs]
       Example usage:
       vlsvReader = VlsvFile("fullf.0001.vlsv")
       result = pitch_angles( vlsvReader=vlsvReader, cellid=1924, cosine=True, plasmaframe=False )
       # Plot the data
       import pylab as pl
       pl.hist(result[0].data, weights=result[1].data, bins=100, log=False)
   '''
   # Read the velocity cells:
   velocity_cell_data = vlsvReader.read_velocity_cells(cellid)
   # Read bulk velocity:
   bulk_velocity = np.array(vlsvReader.read_variable("rho_v", cellid) / vlsvReader.read_variable("rho", cellid), copy=False)
   # Calculate the pitch angles for the data:
   B = vlsvReader.read_variable("B", cellid)
   B_unit = B / np.linalg.norm(B)
   # Get cells:
   vcellids = velocity_cell_data.keys()
   # Get avgs data:
   avgs = velocity_cell_data.values()
   # Get a list of velocity coordinates:
   if plasmaframe == True:
      v = vlsvReader.get_velocity_cell_coordinates(vcellids) / bulk_velocity
   else:
      v = vlsvReader.get_velocity_cell_coordinates(vcellids)
   # Get norms:
   v_norms = np.sum(np.abs(v)**2,axis=-1)**(1./2)
   # Get the angles:
   if cosine == True:
      pitch_angles = v.dot(B_unit) / v_norms
      units = "radian"
   else:
      pitch_angles = np.arccos(v.dot(B_unit) / v_norms) / (2*np.pi) * 360
      units = "degree"
   # Return the pitch angles and avgs values:
   from output import output_1d
   return output_1d([pitch_angles, avgs], ["Pitch_angle", "avgs"], [units, ""])
   #pl.hist(pitch_angles, weights=avgs, bins=bins, log=log)


