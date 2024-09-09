
import os
import socket
import sys
from operator import attrgetter

import matplotlib.pylab as plt
import numpy as np
import pytools as pt
import yt
from mpl_toolkits import mplot3d
from skimage import measure
from yt.visualization.api import Streamlines

import ids3d
import logging

'''
Finds the magnetopause in 3D-run and plots it with matplotlib
Accepts 2 (or 0) command line arguments: 1) how many streamlines to ignore,  2) how many streamlines to count after the ignored ones

output:
saves png-image to the current folder 
'''




arcDeg = 10 # 360/arcDeg must be an integer. Best at 10, ok at 5, 10: 36 slices, 5: 72 slices


class Point: #a 3D-point

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


        self.x, self.r, self.phi = np.array(cartesian_to_polar(x,y,z))

    def print(self):
        print(self.x, self.y, self.z)


class YZSlice: #a slice in yz-plane

    def __init__(self, x, max_deg):
        self.x = x
        self.max_deg = max_deg
        self.min_deg = max_deg-arcDeg
        self.points = [] #contains point in slice

    def get_min_deg(self):
        return self.min_deg

    def get_max_deg(self):
        return self.max_deg

    def add_point(self, p):
        p_phi = p.phi
        if p_phi == 0: p_phi = 360

        if (p_phi <= self.max_deg) and (p_phi >= self.min_deg):
            self.points.append(p)
        else:
            logging.info("point has wrong polar angle! (x, max_deg, p_phi):", self.x, " ", self.max_deg, " ", p_phi)
            exit()


    def get_mean_point(self, ignore, count):
    #get mean of points in slice s ignoring first 'ignore' points
    #and counting the average of the next 'count' points

        if len(self.points) < (ignore+count): #if there are no enough points in slice just ignore it
            return 0

        pnts = sorted(self.points, key=attrgetter('r')) #sort points by radius to the x-axis

        m = []
        i = 0
        for p in pnts:
            if i < ignore:
                i = i + 1
                continue
            elif i < (ignore+count):
                m.append(p.r)
            else:
                break
            i = i + 1


        mean_r = np.mean(m)

        #make (x,y,z)-coordinates of the mean point
        coord_mean = (polar_to_cartesian(self.x, mean_r, (self.max_deg-(arcDeg/2))))
        return coord_mean

class YZPlane: #an yz-plane at a certain x

    def __init__(self, x):
        self.x = x
        self.slices = []

        for i in range (0, int(360/arcDeg)): #initialize slices list
            s = YZSlice(self.x, (i*arcDeg+arcDeg))
            self.slices.append(s)


    def add_point(self, p):

        if (p.x == self.x):
            ind = get_slice_index(p.phi)
            sl = self.slices[ind]
            sl.add_point(p)
        else:
            logging.info("point is in the wrong yz-plane!")



def get_bulk(): #fetch bulk file name

    fileLocation="/wrk/group/spacephysics/vlasiator/3D/EGE/bulk/"
    fileN = "bulk.0002193.vlsv"

    return (fileLocation+fileN)


def get_slice_index(deg): #aux function for indexing arcDeg-degree slices

    if deg == 0: #goes into last index slice
        return int((360/arcDeg)-1)

    if deg < 0:
        deg = 360 + deg

    ind = 0
    j = 0
    for i in range (arcDeg, 360+arcDeg, arcDeg):
        if (deg <= i) and (deg > j):
            return ind
        j = i
        ind = ind + 1

    logging.info("Nope, degrees: " + str(deg) + " ind: " + str(ind)) #something's wrong
    exit()

def to_Re(m): #meters to Re
    Re = 6371000
    return m/Re


#Coordinate transforms
def cartesian_to_polar(x, y, z): #cartesian y,z coordinates to polar
    x = x
    r = np.sqrt(y**2 + z**2)
    phi_0 = np.arctan2(z, y)
    phi = np.degrees(phi_0) #angle in degrees

    if phi < 0: phi = phi+360
    return(x, r, phi)

def polar_to_cartesian(x, r, phi): #polar r, phi coordinates to cartesian
    x = x
    phi_0 = np.radians(phi)
    y = r * np.cos(phi_0)
    z = r * np.sin(phi_0)
    return(x, y, z)






def get_poynting_vectors(coordinate_points): #TODO
    '''
    input: coordinate points in form [[x0,y0,z0],[x1,y1,z1],...]
    returns Poynting vectors at input points
    DOESN'T WORK
    '''

    #coordinates to m
    Re = 6371000
    coordinate_points = [[coord*Re for coord in arr] for arr in coordinate_points]


    mu_0 = 1.25663706212e-06 #vacuum permeability, unit N

    name = get_bulk()
    f = pt.vlsvfile.VlsvReader(file_name=name)


    #The non-working part, no interpolation for fsgrid variables yet
    B = f.read_interpolated_variable("fg_b", coordinate_points)
    E = f.read_interpolated_variable("fg_e", coordinate_points)

    #Poyntnig vectors pseudocode
    P = (1/mu_0) * np.cross(E,B)

    return P




def make_streamlines():

    #boxre=[-30, 30, -30, 30, -30, 30]
    boxre = [0]

    # bulk file
    name = get_bulk()
    f = pt.vlsvfile.VlsvReader(file_name=name)

    #get box coordinates from data
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    [xsizefs, ysizefs, zsizefs] = f.get_fsgrid_mesh_size()
    simext_m =[xmin,xmax,ymin,ymax,zmin,zmax]
    sizes = np.array([xsize,ysize,zsize])
    sizesfs = np.array([xsizefs,ysizefs,zsizefs])


    simext = list(map(to_Re, simext_m))


    #set box coordnates
    if len(boxre)==6:
        boxcoords=boxre
    else:
        boxcoords=list(simext)


    # If box extents were provided manually, truncate to simulation extents
    boxcoords[0] = max(boxcoords[0],simext[0]) #xmax
    boxcoords[1] = min(boxcoords[1],simext[1]) #xmin
    boxcoords[2] = max(boxcoords[2],simext[2]) #ymax
    boxcoords[3] = min(boxcoords[3],simext[3]) #ymin
    boxcoords[4] = max(boxcoords[4],simext[4]) #zmax
    boxcoords[5] = min(boxcoords[5],simext[5]) #zmin


    cellids = f.read_variable("CellID")
    indexids = cellids.argsort()
    cellids = cellids[indexids]

    reflevel = ids3d.refinement_level(xsize, ysize, zsize, cellids[-1])



    ##############

    #Read the data from vlsv-file
    V = f.read_variable("vg_v")

    #from m to Re
    V = np.array([[to_Re(i) for i in j ]for j in V])
    #logging.info(V)

    V = V[indexids]

    Vdpoints = ids3d.idmesh3d2(cellids, V, reflevel, xsize, ysize, zsize, 3)


    Vxs = Vdpoints[:,:,:,0]
    Vys = Vdpoints[:,:,:,1]
    Vzs = Vdpoints[:,:,:,2]

    data=dict(Vx=Vxs,Vy=Vys,Vz=Vzs)


    #Create streamline seeds (starting points for streamlines)
    streamline_seeds = []

    #range: np.arange(from, to, step)
    for i in np.arange(-15.0, 15.0, 0.5):
        for j in np. arange(-15.0, 15.0, 0.5):
            streamline_seeds.append([20, i, j])


    streamline_seeds = np.array(streamline_seeds)

    #dataset in yt-form
    yt_dataset = yt.frontends.stream.load_uniform_grid(
        data,
        sizesfs,
        bbox=np.array([[boxcoords[0], boxcoords[1]],
                    [boxcoords[2],boxcoords[3]],
                    [boxcoords[4],boxcoords[5]]]))


    #data, seeds, dictionary positions, lenght of lines
    streamlines_pos = yt.visualization.api.Streamlines(yt_dataset, streamline_seeds,
                                                    "Vx", "Vy", "Vz", length=60, direction=1)

    #where the magic happens
    streamlines_pos.integrate_through_volume()

    #get rid of lines that don't work
    #for s in np.arange(0, len(streamlines_pos.streamlines)):
    #    logging.info(streamlines_pos.streamlines[s])
    #    stream_pos = streamlines_pos.streamlines[s]
    #    stream_pos = stream_pos[(np.all(stream_pos != 0.0) | (np.linalg.norm(stream_pos, axis = 1) > 4.7)) & (stream_pos[0,2] != 0.0)]


    return streamlines_pos #returns yt streamlines object


def interpolate(YTstream, x_points):
    '''
    :kword YTstream:    A single streamline in yt Streamline -form
    :kword x_points:    Array of x-axis points where coordinates of streamline passing YZ-plane are wanted
    '''

    streamline = YTstream.positions.to_value()

    arr = np.array(streamline)

    #maybe needs to be sorted???
    xp = arr[:,0] #x-coordinates of streamline
    yp = arr[:,1] #y-coordinates
    zp = arr[:,2] #z-coordinates

    #reverse arrays so interpolation works
    xp = xp[::-1]
    yp = yp[::-1]
    zp = zp[::-1]


    #interpolate z from xz-plane
    z_points = np.interp(x_points, xp, zp, left=float('inf'), right=float('inf'))

    #interpolate y from xy-plane
    y_points = np.interp(x_points, xp, yp, left=float('inf'), right=float('inf'))

    #if z or y -coordinate is inf at some point scrap the whole streamline from data
    for y, z in zip(y_points, z_points):
        if z == float('inf') or y == float('inf'):
            return None

    return [x_points, y_points, z_points]


def get_magnetopause(streams, ignore, count):

    #wanted points in x-axis (YZ-plane places)
    x_points = np.array(np.arange(8, -30, -0.5))

    #planes
    planes = []
    for x in x_points:
        planes.append(YZPlane(x))

    #get coordinates of every stream at given x points (goes through all streamlines on by one)
    for s in np.arange(0, len(streams.streamlines)):
        YTstream = streams.path(s)

        stream_points = interpolate(YTstream, x_points)
        if stream_points is None: #discard ugly streamlines (those that end too soon for example)
            continue

        #add interpolated points to yz-planes
        for i in range(0, len(x_points)):
            if stream_points[0][i] != x_points[i]:
                logging.info("Something's very wrong")
                exit()
            p = Point(stream_points[0][i], stream_points[1][i], stream_points[2][i])
            planes[i].add_point(p)



    #Now we (should) have every interpolated Point saved

    #where to save the magnetopause coordinates by slice
    pause_coords_by_slice = []
    for i in range(0, int(360/arcDeg)):
        pause_coords_by_slice.append([])

    #where to save the magnetopause coordinates by x-value
    pause_coords_by_x = []

    #for every yz-plane
    for p in planes:
        p_ind = len(pause_coords_by_x)
        pause_coords_by_x.append([])

        #for every slice get mean point
        for i in range(0, len(p.slices)):
            s = p.slices[i]
            mean_point = s.get_mean_point(ignore, count)
            if mean_point != 0:
                pause_coords_by_slice[i].append(mean_point)
                pause_coords_by_x[p_ind].append(mean_point)

    pause_coords_by_slice = np.array(pause_coords_by_slice)

    return (pause_coords_by_slice, pause_coords_by_x)  #returns streamlines and array of magnetopause positions

def plot_all(ax, XmeshXY=None, YmeshXY=None, pass_maps=None): #external function for plot_colormap

    #command line args
    if len(sys.argv) != 2:
        ignore, count = 3, 3
    else:
        ignore = int(sys.argv[1])
        count = int(sys.argv[2])

    streams, magnetopause_coords = get_magnetopause(ignore, count)

    #plot streamlines
    for s in np.arange(0, len(streams.streamlines)):
        stream_pos = streams.streamlines[s]
        ax.plot(stream_pos[:,0], stream_pos[:,2], color='#C4C4C4', lw=0.3)

    #plot magnetopause
    #for every slice
    for sl in magnetopause_coords:
        if not sl: #if slice is empty(works on nested lists?)
            continue

        sl = np.array(sl)
        ax.plot3D(sl[:,0], sl[:1], sl[:,2], 'k', lw = 0.4)

def make_surface(coords):

    '''
    Defines surface constructed of input coordinates as triangles
    Returns list of verts and vert indices of surface triangles

    coordinates must be in form [...[c11, c21, c31, ... cn1,[c12, c22, c32, ... cn2],...
    where cij = [xij, yij, zij], i marks slice, j marks yz-plane (x_coord) index

    How it works:
    Three points make a triangle, triangles make the surface.
    For every two planes next to each other:
        take every other point from plane1, every other from plane2 (in order!)
        from list of points: every three points closest to each other make a surface

        Example:
        plane 1: [v1, v2, v3, v4]
        plane 2: [v5, v6, v7, v8]

        -> list: [v1, v5, v2, v6, v3,...]
        -> triangles:
            v1 v5 v2
            v5 v2 v6
            v2 v6 v3
            .
            .
            .

    '''
    verts = [] #points
    faces = [] #surface triangles

    slices_in_plane = len(coords[0])
    planes = len(coords)


    #get points
    for plane in coords:
        for vert in plane:
            verts.append(vert)



    #get triangle (face) indices

    #Let's consider the area between the first two planes
    #ind1, ind2, ind3 for triangle indices
    ind1 = 0 #starts from plane 1
    ind2 = slices_in_plane #starts from plane 2
    ind3 = 1 #starts from plane 1
    first_triangles = []

    while len(first_triangles) < (2*slices_in_plane):
        first_triangles.append([ind1, ind2, ind3])
        ind1 = ind2
        ind2 = ind3
        if (ind3 == (slices_in_plane*2)-1): #max index, go over to plane 1 first index
            ind3 = 0
        elif (ind3 == 0): #last round, go to plane 2 first index
            ind3 = slices_in_plane
        else:
            ind3 = ind1 + 1

    first_triangles = np.array(first_triangles)

    #Now the rest of the triangles are just the first triangles + (index of area * slices_in_plane)
    #maybe could be done better?
    for area_index in range(planes-1):
        next_triangles = [x + slices_in_plane*area_index for x in first_triangles]
        faces.extend(next_triangles)

    #faces = np.array(faces, dtype=np.int32, order='C')
    #verts = np.array(verts)
    #logging.info(faces.max())
    #logging.info(len(verts[:,0]))

    return verts, faces #verts is a list of coordinates, faces an array of arrays including indices of verts that make the triangles



def triangle(A,B,C):
    #A, B, C are 3D points defining the triangle
    #NOTE: if input points are straight from make_surface(), every other normal will be outwards and every other inwards!

    #Centre point of the triangle (centroid)
    c_x = (A[0]+B[0]+C[0])/3
    c_y = (A[1]+B[1]+C[1])/3
    c_z = (A[2]+B[2]+C[2])/3

    centre = [c_x, c_y, c_z]


    #Area of the triangle:
    #vetors with A as starting point
    AB = [(B[0]-A[0]), (B[1]-A[1]), (B[2]-A[2])]
    AC = [(C[0]-A[0]), (C[1]-A[1]), (C[2]-A[2])]

    #Cross product (non-unit length normal vector)
    cross = np.cross(AB, AC)
    area = np.linalg.norm(cross)/2

    #unit-length normal vector
    normal = cross / np.linalg.norm(cross)

    # vector area = area * normal vector
    vector_area = area * normal

    #check that area vector points inwards (TODO)

    #Poynting vector at the centre (not working yet)
    P = get_poynting_vectors(centre)

    flow = np.dot(P, vector_area)

    return flow




def main():
    '''
    command line arguments: how many inner streamlines to ignore and how many to count in
    '''

    #command line args
    if len(sys.argv) != 2:
        ignore, count = 3, 3
    else:
        ignore = int(sys.argv[1])
        count = int(sys.argv[2])

    #What to plot:
    #[streamlines, magnetopause, poynting vectors, surface]
    plotting = [False, False, False, True]


    streams = make_streamlines()
    pause_by_slice, pause_by_x = get_magnetopause(streams, ignore, count)

    ax = plt.axes(projection='3d', xlabel='X (Re)', ylabel='Y (Re)', zlabel='Z (Re)')



    if plotting[0]: #plot streamlines
        for s in np.arange(0, len(streams.streamlines)):
            stream_pos = streams.streamlines[s]
            ax.scatter(stream_pos[:,0], stream_pos[:,1], stream_pos[:,2])


    if plotting[1]: #plot magnetopause by coordinates

        coords = []
        for sl in pause_by_slice:
            if len(sl) == 0: continue #if slice is empty(works on nested lists?)
            sl = np.array(sl)
            ax.plot3D(sl[:,0], sl[:,1], sl[:,2])

            coords.append([sl[:,0], sl[:,1], sl[:,2]])

        for plane in pause_by_x:
            plane = np.array(plane)
            ax.plot3D(plane[:,0], plane[:,1], plane[:,2])


    if plotting[2]: #plot Poynting vetors (not working!)
        raise NotImplementedError("Poynting vector plotting not implemented!")
        # coords = np.array(coords)
        # poynting = np.array(poynting)

        # ax.quiver(coords[:,0], coords[:,1], coords[:,2], P[:,0], P[:,1], P[:,2])


    if plotting[3]: #plot surface
        logging.info('Making the surface...')
        verts, faces = make_surface(pause_by_x)
        verts = np.array(verts)

        ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                linewidth=0.2, antialiased=True)

    plt.savefig('magnetopause3d.png')

    #save 2d projection
    #ax.view_init(azim=45, elev=0)
    #plt.savefig('magnetopause3dprojection.png')


    logging.info('Ready!')





if __name__ == "__main__":
    main()
