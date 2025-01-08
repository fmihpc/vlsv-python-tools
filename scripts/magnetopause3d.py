'''
Finds the magnetopause position by tracing steamines of the plasma flow for three-dimensional Vlasiator runs. Needs the yt package.
'''
from pyCalculations import ids3d
import matplotlib.pyplot as plt
import numpy as np
import analysator as pt
import plot_colormap3dslice
import yt
import math
from mpl_toolkits import mplot3d
from yt.visualization.api import Streamlines




def to_Re(m): #meters to Re
    return m/6371000


def cartesian_to_polar(cartesian_coords): # for segments of plane
    y,z = cartesian_coords[0], cartesian_coords[1]
    r = np.sqrt(z**2 + y**2)
    phi = np.arctan2(z, y)
    phi = np.rad2deg(phi) #angle in degrees
    if phi < 0: phi = phi+360
    return(r, phi)

def polar_to_cartesian(r, phi):
    phi = np.deg2rad(phi)
    y = r * np.cos(phi)
    z = r * np.sin(phi)
    return(y, z)


def interpolate(streamline, x_points):
    arr = np.array(streamline)

    # set arrays for interpolation
    xp = arr[:,0][::-1]
    yp = arr[:,1][::-1]
    zp = arr[:,2][::-1]

    #interpolate z from xz-plane
    z_points = np.interp(x_points, xp, zp, left=np.NaN, right=np.NaN)
    #interpolate y from xy-plane
    y_points = np.interp(x_points, xp, yp, left=np.NaN, right=np.NaN)


    return np.array([x_points, y_points, z_points])




def make_surface(coords):

    '''
    Defines surface constructed of input coordinates as triangles
    Returns list of verts and vert indices of surface triangles

    coordinates must be in form [...[c11, c21, c31, ... cn1],[c12, c22, c32, ... cn2],...
    where cij = [xij, yij, zij], i marks slice, j marks yz-plane (x_coord) index 

    How it works:
    Three points make a triangle, triangles make the surface.
    For every two planes next to each other:
    - take every other point from plane1, every other from plane2 (in order!)
    - from list of points: every three points closest to each other make a surface

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
    for area_index in range(planes-1):
        next_triangles = [x + slices_in_plane*area_index for x in first_triangles]
        faces.extend(next_triangles)

    return verts, faces



def make_streamlines(vlsvFileName):
    ## make streamlines
    boxre = [0]
                            
    # bulk file
    f = pt.vlsvfile.VlsvReader(file_name=vlsvFileName)

    #get box coordinates from data
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    [xsizefs, ysizefs, zsizefs] = f.get_fsgrid_mesh_size()
    simext_m =[xmin,xmax,ymin,ymax,zmin,zmax]
    sizes = np.array([xsize,ysize,zsize])
    sizesfs = np.array([xsizefs,ysizefs,zsizefs])

    simext = list(map(to_Re, simext_m))
    boxcoords=list(simext)

    cellids = f.read_variable("CellID")
    indexids = cellids.argsort()
    cellids = cellids[indexids]

    reflevel = ids3d.refinement_level(xsize, ysize, zsize, cellids[-1])


    #Read the data from vlsv-file
    V = f.read_variable("vg_v")

    #from m to Re
    V = np.array([[to_Re(i) for i in j ]for j in V])


    V = V[indexids]

    if np.ndim(V)==1:
        shape = None
    elif np.ndim(V)==2: # vector variable
        shape = V.shape[1]
    elif np.ndim(V)==3:  # tensor variable
        shape = (V.shape[1], V.shape[2])


    Vdpoints = ids3d.idmesh3d2(cellids, V, reflevel, xsize, ysize, zsize, shape)


    Vxs = Vdpoints[:,:,:,0]
    Vys = Vdpoints[:,:,:,1]
    Vzs = Vdpoints[:,:,:,2]

    data=dict(Vx=Vxs,Vy=Vys,Vz=Vzs)

    #Create streamline seeds (starting points for streamlines)
    seedN = 50 #seeds per row, final seed count will be seedN*seedN !
    streamline_seeds = np.zeros([seedN**2, 3])

    #range: np.arange(from, to, step)
    t = np.linspace(-5, 5, seedN)
    k = 0
    for i in t:
        for j in t:
            streamline_seeds[k] = [20, i, j]
            k = k+1


    #dataset in yt-form
    yt_dataset = yt.load_uniform_grid(
        data,
        sizesfs,
        bbox=np.array([[boxcoords[0], boxcoords[1]],
                    [boxcoords[2],boxcoords[3]],
                    [boxcoords[4],boxcoords[5]]]))
                                    

    # data, seeds, dictionary positions, length of streamlines
    streamlines_pos = yt.visualization.api.Streamlines(yt_dataset, streamline_seeds,
                                                    "Vx", "Vy", "Vz", length=50, direction=1)

    # make the streamlines
    streamlines_pos.integrate_through_volume()

    return np.array(streamlines_pos.streamlines)


def make_magnetopause(streams):
    streampoints = np.reshape(streams, (streams.shape[0]*streams.shape[1], 3)) #all the points in one array
    
    ## find the subsolar dayside point in the x-axis
    ## do this by finding a stremline point on positive x axis closest to the Earth
    x_axis_points = streampoints[(np.floor(streampoints[:,1])==0) & (np.floor(streampoints[:,2])==0)]
    subsolar_x =np.min(x_axis_points[:,0])

    ## define points in the x axis where to find magnetopause points on the yz-plane
    x_points = np.arange(subsolar_x, -15, -0.5)
    
    ## interpolate more exact points for streamlines at exery x_point
    new_streampoints = np.zeros((len(x_points), len(streams), 2)) # new array for keeping interpolated streamlines in form new_streampoints[x_point, streamline, y and z -coordinates] 
    i=0 # streamline
    for stream in streams:
        interpolated_streamline = interpolate(stream, x_points)

        if type(interpolated_streamline) is np.ndarray: # don't use 'discarded' streamlines, see function interpolate()
            for j in range(0, len(x_points)):
                y,z = interpolated_streamline[1,j], interpolated_streamline[2,j]
                new_streampoints[j, i,:] = np.array([y,z])
        i += 1



    ## create a list of streamline points in polar coordinates (for every x_point)
    polar_coords = np.zeros_like(new_streampoints)
    for i in range(0,new_streampoints.shape[0]):
        for j in range(0,new_streampoints.shape[1]):
            polar_coords[i,j,:] = cartesian_to_polar(new_streampoints[i,j])


    ## now start making the magnetopause
    ## in each x_point, divide the plane into sectors and look for the closest streamline to x-axis in the sector
    sector_n = 36

    ## if given sector number isn't divisible by 4, make it so because we want to have magnetopause points at exactly y=0 and z=0 for 2d slices of the whole thing
    while sector_n%4 != 0:
        sector_n +=1

    sector_width = 360/sector_n
    magnetopause = np.zeros((len(x_points), sector_n, 3))

    for i,x_point in enumerate(x_points): #loop over every chosen x-axis point
        # divide the yz-plane into sectors
        for j, mean_sector_angle in enumerate(np.arange(0, 360, sector_width)):
            min_angle = mean_sector_angle-sector_width/2
            max_angle = mean_sector_angle+sector_width/2

            # find points that are in the sector
            if mean_sector_angle == 0: # special case as the first sector needs streamlines around phi=0
                min_angle = min_angle+360
                # divide into phi<360 and phi>0
                sector1 = polar_coords[i, (polar_coords[i,:,1] <= 360)*(polar_coords[i,:,1] > min_angle)]
                sector2 = polar_coords[i, (polar_coords[i,:,1] <= max_angle)*(polar_coords[i,:,1] >= 0)]
                sector_points = np.concatenate((sector1, sector2))

            else:
                sector_points = polar_coords[i, (polar_coords[i,:,1] <= max_angle)*(polar_coords[i,:,1] > min_angle)]

            # discard 'points' with r=0 and check that there's at least one streamline point in the sector
            sector_points = sector_points[sector_points[:,0] != 0.0]
            if sector_points.size == 0:
                raise ValueError('No streamlines found in the sector')

            # find the points closest to the x-axis
            closest_point_radius = sector_points[sector_points[:,0].argmin(), 0] # smallest radius
            
            # return to cartesian coordinates and save as a magnetopause point at the middle of the sector
            y,z = polar_to_cartesian(closest_point_radius, mean_sector_angle)
            magnetopause[i,j,:] = [x_point, y, z]


    # make a tip point for the magnetopause for prettier 3d plots 
    tip = np.array([subsolar_x, 0, 0])
    tips = np.tile(tip, (magnetopause.shape[1],1))
    magnetopause = np.vstack(([tips], magnetopause))
    
    return magnetopause




def main():

    ## get bulk data
    run =  'EGI'
    fileLocation="/wrk-vakka/group/spacephysics/vlasiator/3D/"+run+"/bulk/"
    fileN = "bulk5.0000070.vlsv" 

    ## STREAMLINES
    streams = make_streamlines(fileLocation+fileN)
    ## MAGNETOPAUSE
    magnetopause = make_magnetopause(streams)


    ## PLOTTING
    outdir=""

    ## take separate arrays for different 2d slice plots
    slices = magnetopause.shape[1]
    quarter_slice = int(slices/4)
    # xy plane: z=0
    xy_slice = np.concatenate((magnetopause[:,0][::-1], magnetopause[:,2*quarter_slice]))
    # xz plane: y=0
    xz_slice = np.concatenate((magnetopause[:,quarter_slice][::-1], magnetopause[:,3*quarter_slice]))


    #2D plots
    # analysator 3dcolormapslice y=0
    if True:
        def external_plot(ax,XmeshXY=None, YmeshXY=None, pass_maps=None, requestvariables=False):
            if requestvariables==True:
                return ['vg_v']
            ax.plot(xz_slice[:,0], xz_slice[:,2],  color='limegreen', linewidth=1.5)


        plot_colormap3dslice.plot_colormap3dslice(
        filename=fileLocation+fileN,
        outputdir=outdir,
        run=run,
        nooverwrite=None,
        boxre = [-21,21,-21,21],
        title=None,
        draw=None, usesci=True, Earth=True,
        external = external_plot,
        colormap='inferno',
        normal='y'
        )

    # analysator 3dcolormapslice z=0
    if True:
        def external_plot(ax,XmeshXY=None, YmeshXY=None, pass_maps=None, requestvariables=False):
            if requestvariables==True:
                return ['vg_v']
            ax.plot(xy_slice[:,0], xy_slice[:,1], color='limegreen', linewidth=1.5)

        plot_colormap3dslice.plot_colormap3dslice(
        filename=fileLocation+fileN,
        outputdir=outdir,
        run=run,
        nooverwrite=None,
        boxre = [-21,21,-21,21],
        title=None,
        draw=None, usesci=True, Earth=True,
        external = external_plot,
        colormap='inferno',
        normal='z'
        )

    # 3D plot
    # matplotlib 3d surface plot for single image
    if False:
        verts, faces = make_surface(magnetopause)
        verts = np.array(verts)
        
        fig = plt.figure()
        ax2 = fig.add_subplot(projection='3d')
        ax2.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
        ax2.view_init(azim=-60, elev=5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        fig.tight_layout()
        plt.savefig(outdir+run+'_3d_magnetopause.png')


if __name__ == "__main__":
    main()
