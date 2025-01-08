import numpy as np
import logging

# finds the cell ids which are needed to plot a 2d cut through out of a 3d mesh # WARNING! works only if cellids is sorted
def ids3d(cellids, depth, reflevel,
          xsize, ysize, zsize,
          xmin=None, xmax=None,
          ymin=None, ymax=None,
          zmin=None, zmax=None):

  # is the cut in x, y or z direction
  if xmin != None and xmax != None:
    imin = xmin; imax = xmax; size = xsize; sett = 0
  elif ymin != None and ymax != None:
    imin = ymin; imax = ymax; size = ysize; sett = 1
  elif zmin != None and zmax != None:
    imin = zmin; imax = zmax; size = zsize; sett = 2

  # find the cut through depths (index) for each refinement level
  pro = depth/(imax-imin)
  depths = []
  for i in range(reflevel+1):
    depth = int(pro*size*2**i) + 1 # goes from 1 to Nmax
    if depth > int(size*2**i):
      logging.info("depth error, depth = " +str(depth) +"; i = "+str(i))
      depth -= 1
    depths.append(depth)

  #############
  # find the ids
  length = 0
  cellids = np.array(cellids)
  idlist = np.array([]); indexlist = np.array([])
  cells = int(xsize*ysize*zsize); cellsum = cells; cellsumII = 0 # cellsum = (the number of cells up to refinement
  # level i); cellsumII = (the number of cells up to refinement level i-1)
  for i in range(reflevel+1):
    # cell ids at the refinement level i
    ids = cellids[(cellsumII < cellids) & (cellids <= cellsum)]

    # compute every cell ids x, y and z coordinates
    z = (ids - cellsumII - 1)//(xsize*ysize*4**i) +1 # goes from 1 to NZ
    #z = np.where(np.isclose(z, z.astype(int)), z, z + 1)
    z = z.astype(int)

    # cellsumB = (cellsumII + the number of ids up to the coordinate z in the refinement level i)
    cellsumB = ((z-1)*xsize*ysize*4**i).astype(int) + cellsumII

    y = (ids - cellsumB - 1)//(xsize*2**i) +1 # goes from 1 to NY
    #y = np.where(y - y.astype(int) == 0, y, y + 1)
    y = y.astype(int)

    cellsumC = ((y-1)*xsize*2**i)
    x = (ids - cellsumB - cellsumC).astype(int) # goes from 1 to NX

    # checks is the cut throughs normal in x, y or z direction
    if sett == 0:
      xyz = x
    elif sett == 1:
      xyz = y
    elif sett == 2:
      xyz = z

    # finds the needed elements to create the asked cut throuhg and puts the results in the indexlist and the idlist
    elements = xyz == depths[i]
    indexlist = np.append(indexlist, np.arange(length, length+len(xyz))[elements])
    idlist = np.append(idlist, ids[elements])

    # update these values to match refinement level i+1
    length += len(xyz)
    cellsumII = cellsum
    cellsum += cells*8**(i+1)
  # returns a list of ids (idlist) and a equivalent list of indices (indexlist)
  return idlist.astype(int), indexlist.astype(int)

# creates 2d grid for ploting
def idmesh3d(idlist, data, reflevel, xsize, ysize, zsize, xyz, datadimension):
  # is the cut in x, y or z direction
  if xyz == 0:
    dims = np.array([ysize, zsize]) * 2**reflevel
  elif xyz == 1:
    dims = np.array([xsize, zsize]) * 2**reflevel
  elif xyz == 2:
    dims = np.array([xsize, ysize]) * 2**reflevel
  dims = dims.astype(int)

  # datadimension is None for scalar,
  # N for vector, and (N,M) for tensor data
  if datadimension is None:
    dpoints = np.zeros(dims)
  elif np.ndim(datadimension) == 0:
    dpoints = np.zeros(np.append(dims, datadimension).astype(int))
  elif np.ndim(datadimension) == 1:
    dpoints = np.zeros(np.append(dims, (datadimension[0], datadimension[1])).astype(int))
  else:
    logging.info("Error finding data dimension in idmesh3d")
    return -1

  ######################
  # create the plot grid
  cells = int(xsize*ysize*zsize)
  cellsum = cells # the number of cells up to refinement level i
  cellsumII = 0 # the number of cells up to refinement level i-1
  for i in range(reflevel+1):
    # the cell ids and the data at the refinement level i
    ids = idlist[(cellsumII < idlist) & (idlist <= cellsum)]
    if datadimension is None:
      dat = data[(cellsumII < idlist) & (idlist <= cellsum)]
    elif np.ndim(datadimension) == 0:
      dat = data[(cellsumII < idlist) & (idlist <= cellsum),:]
    elif np.ndim(datadimension) == 1:
      dat = data[(cellsumII < idlist) & (idlist <= cellsum),:,:]

    # compute every cell ids x, y and z coordinates
    z = (ids - cellsumII - 1)//(xsize*ysize*4**i) +1 # goes from 1 to NZ
    #z = np.where(np.isclose(z, z.astype(int)), z, z + 1)
    z = z.astype(int)

    # cellsumB = (cellsumII + the number of ids up to the coordinate z in the refinement level i)
    cellsumB = ((z-1)*xsize*ysize*4**i).astype(int) + cellsumII

    y = (ids - cellsumB - 1)//(xsize*2**i) +1 # goes from 1 to NY
    #y = np.where(y - y.astype(int) == 0, y, y + 1)
    y = y.astype(int)

    cellsumC = ((y-1)*xsize*2**i)
    x = (ids - cellsumB - cellsumC).astype(int) # goes from 1 to NX

    # gets the correct coordinate values and the widths for the plot
    if xyz == 0:
      a = y
      b = z
    elif xyz == 1:
      a = x
      b = z
    elif xyz == 2:
      a = x
      b = y

    # inserts the data values into dpoints
    # Broadcasting magic for vectorization! Not quite a oneliner
    # Skipping the declaration is actually faster but looks like hot garbage
    coords = np.add(((np.stack((a, b)) - 1) * 2**(reflevel - i))[:, :, np.newaxis], np.array(np.meshgrid(np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)))).reshape(2, 1, -1))
    dpoints[coords[0,:,:], coords[1, :, :]] = dat[:, np.newaxis]
    #dpoints[np.add(((np.stack((a, b)) - 1) * 2**(reflevel - i))[:, :, np.newaxis], np.array(np.meshgrid(np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)))).reshape(2, 1, -1))[0,:,:], np.add(((np.stack((a, b)) - 1) * 2**(reflevel - i))[:, :, np.newaxis], np.array(np.meshgrid(np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)))).reshape(2, 1, -1))[1, :, :]] = dat[:, np.newaxis]
    #for j in range(2**(reflevel-i)):
    #  for k in range(2**(reflevel-i)):
    #    if datadimension is None:
    #      dpoints[(a - 1) * 2**(reflevel - i) + j, (b - 1) * 2**(reflevel - i) + k - 1] = dat
    #    elif np.ndim(datadimension) == 0:
    #      dpoints[(a - 1) * 2**(reflevel - i) + j - 1, (b - 1) * 2**(reflevel - i) + k - 1, :] = dat
    #    elif np.ndim(datadimension) == 1:
    #      dpoints[(a - 1) * 2**(reflevel - i) + j - 1, (b - 1) * 2**(reflevel - i) + k - 1, :, :] = dat

    # update these values to match refinement level i+1
    cellsumII = cellsum
    cellsum += cells*8**(i+1)

  return np.swapaxes(dpoints, 0, 1)

# creates 3d grid for ploting
def idmesh3d2(idlist, data, reflevel, xsize, ysize, zsize, datadimension):
  # is the cut in x, y or z direction
  dims = np.array([xsize, ysize, zsize]) * 2**reflevel
  dims = dims.astype(int)
  
  # datadimension is None for scalar,
  # N for vector, and (N,M) for tensor data
  if datadimension is None:
    dpoints = np.zeros(dims)
  elif np.ndim(datadimension) == 0:
    dpoints = np.zeros(np.append(dims, datadimension).astype(int))
  elif np.ndim(datadimension) == 1:
    dpoints = np.zeros(np.append(dims, (datadimension[0], datadimension[1])).astype(int))
  else:
    logging.info("Error finding data dimension in idmesh3d")
    return -1

  ######################
  # create the plot grid
  cells = int(xsize*ysize*zsize)
  cellsum = cells # the number of cells up to refinement level i
  cellsumII = 0 # the number of cells up to refinement level i-1
  for i in range(reflevel+1):
    # the cell ids and the data at the refinement level i
    ids = idlist[(cellsumII < idlist) & (idlist <= cellsum)]
    if datadimension is None:
      dat = data[(cellsumII < idlist) & (idlist <= cellsum)]
    elif np.ndim(datadimension) == 0:
      dat = data[(cellsumII < idlist) & (idlist <= cellsum),:]
    elif np.ndim(datadimension) == 1:
      dat = data[(cellsumII < idlist) & (idlist <= cellsum),:,:]

    # compute every cell ids x, y and z coordinates
    z = (ids - cellsumII - 1)//(xsize*ysize*4**i) +1 # goes from 1 to NZ
    #z = np.where(np.isclose(z, z.astype(int)), z, z + 1)
    z = z.astype(int)

    # cellsumB = (cellsumII + the number of ids up to the coordinate z in the refinement level i)
    cellsumB = ((z-1)*xsize*ysize*4**i).astype(int) + cellsumII

    y = (ids - cellsumB - 1)//(xsize*2**i) +1 # goes from 1 to NY
    #y = np.where(y - y.astype(int) == 0, y, y + 1)
    y = y.astype(int)

    cellsumC = ((y-1)*xsize*2**i)
    x = (ids - cellsumB - cellsumC).astype(int) # goes from 1 to NX

    # inserts the data values into dpoints
    # Broadcasting magic for vectorization! Not quite a oneliner
    # Skipping the declaration is actually faster but looks like hot garbage
    coords = np.add(((np.stack((x, y, z)) - 1) * 2**(reflevel - i))[:, :, np.newaxis], np.array(np.meshgrid(np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)))).reshape(3, 1, -1))
    dpoints[coords[0, :, :], coords[1, :, :], coords[2, :, :]] = dat[:, np.newaxis]
    #dpoints[np.add(((np.stack((a, b)) - 1) * 2**(reflevel - i))[:, :, np.newaxis], np.array(np.meshgrid(np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)))).reshape(2, 1, -1))[0,:,:], np.add(((np.stack((a, b)) - 1) * 2**(reflevel - i))[:, :, np.newaxis], np.array(np.meshgrid(np.arange(2**(reflevel-i)), np.arange(2**(reflevel-i)))).reshape(2, 1, -1))[1, :, :]] = dat[:, np.newaxis]
    #for j in range(2**(reflevel-i)):
    #  for k in range(2**(reflevel-i)):
    #    if datadimension is None:
    #      dpoints[(a - 1) * 2**(reflevel - i) + j, (b - 1) * 2**(reflevel - i) + k - 1] = dat
    #    elif np.ndim(datadimension) == 0:
    #      dpoints[(a - 1) * 2**(reflevel - i) + j - 1, (b - 1) * 2**(reflevel - i) + k - 1, :] = dat
    #    elif np.ndim(datadimension) == 1:
    #      dpoints[(a - 1) * 2**(reflevel - i) + j - 1, (b - 1) * 2**(reflevel - i) + k - 1, :, :] = dat

    # update these values to match refinement level i+1
    cellsumII = cellsum
    cellsum += cells*8**(i+1)

  return dpoints #np.swapaxes(dpoints, 0, 1)

# find the highest refinement level
def refinement_level(xsize, ysize, zsize, bigid):
  cells = int(xsize*ysize*zsize)
  reflevel = 0
  cellsum = cells
  while bigid > cellsum:
    reflevel += 1
    cellsum += int(cells*8**reflevel)

  # return the highest refinement level
  return reflevel


def ids3d_box(cellids, low, up, reflevel,
          xsize, ysize, zsize,
          spatial_mesh_extent):
    """Returns lists of CellIDs and the corresponding indices inside a rectangular 3D box.
    :param cellids:             List of cellids in the simulation box
    :param low:                 An array holding the lower boundaries of the requested box
    :param up:                  An array holding the upper boundaries of the requested box
    :param reflevel:            Highest refinement level in the simulation
    :param xsize:               Number of cells in x direction
    :param ysize:               Number of cells in y direction
    :param zsize:               Number of cells in z direction
    :param spatial_mesh_extent: Smallest and largest values of each spatial component
    """

    [xmin, ymin, zmin, xmax, ymax, zmax] = spatial_mesh_extent

    # find the edge depths (index) for each refinement level
    prox_min = (low[0]-xmin)/(xmax-xmin)
    proy_min = (low[1]-ymin)/(ymax-ymin)
    proz_min = (low[2]-zmin)/(zmax-zmin)
    
    prox_max = (up[0]-xmin)/(xmax-xmin)
    proy_max = (up[1]-ymin)/(ymax-ymin)
    proz_max = (up[2]-zmin)/(zmax-zmin)
    
    depths_x = []
    depths_y = []
    depths_z = []

    for i in range(reflevel+1):
        depth_xmin = int(prox_min*xsize*2**i) + 1   # goes from 1 to Nmax
        depth_ymin = int(proy_min*ysize*2**i) + 1 
        depth_zmin = int(proz_min*zsize*2**i) + 1 
        
        depth_xmax = int(prox_max*xsize*2**i) + 1
        depth_ymax = int(proy_max*ysize*2**i) + 1 
        depth_zmax = int(proz_max*zsize*2**i) + 1
        
        
        depths_x.append([depth_xmin, depth_xmax])
        depths_y.append([depth_ymin, depth_ymax])
        depths_z.append([depth_zmin, depth_zmax])


    # find the ids
    length = 0; 
    cellids = np.array(cellids); 
    idlist = np.array([]); indexlist = np.array([])
    cells = int(xsize*ysize*zsize); cellsum = cells; cellsumII = 0 # cellsum = (the number of cells up to refinement
    # level i); cellsumII = (the number of cells up to refinement level i-1)
    
    for i in range(reflevel+1):
        # cell ids at the refinement level i
        ids = cellids[(cellsumII < cellids) & (cellids <= cellsum)]

        # compute every cell ids x, y and z coordinates
        z = (ids - cellsumII - 1)//(xsize*ysize*4**i) +1 # goes from 1 to NZ
        z = z.astype(int)

        # cellsumB = (cellsumII + the number of ids up to the coordinate z in the refinement level i)
        cellsumB = ((z-1)*xsize*ysize*4**i).astype(int) + cellsumII

        y = (ids - cellsumB - 1)//(xsize*2**i) +1 # goes from 1 to NY
        y = y.astype(int)

        cellsumC = ((y-1)*xsize*2**i)
        x = (ids - cellsumB - cellsumC).astype(int) # goes from 1 to NX

        # finds the needed elements to create the asked box and puts the results in the indexlist and the idlist
        
        x_elements = np.logical_and(x >= depths_x[i][0], x < depths_x[i][1])
        y_elements = np.logical_and(y >= depths_y[i][0], y < depths_y[i][1])
        z_elements = np.logical_and(z >= depths_z[i][0], z < depths_z[i][1])

        elements = x_elements & y_elements & z_elements     # elements within all the limits
        
        indexlist = np.append(indexlist, np.arange(length, length+len(elements))[elements])
        idlist = np.append(idlist, ids[elements])

        # update these values to match refinement level i+1
        length += len(x)
        cellsumII = cellsum
        cellsum += cells*8**(i+1)
    # returns a list of ids (idlist) and a equivalent list of indices (indexlist)
    return idlist.astype(int), indexlist.astype(int)
