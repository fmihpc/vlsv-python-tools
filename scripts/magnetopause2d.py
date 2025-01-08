'''
Finds the magnetopause position by tracing steamines of the plasma flow for two-dimensional Vlasiator runs. Needs the yt package.
'''

import numpy as np
import analysator as pt
import plot_colormap
import yt
from yt.visualization.api import Streamlines




def interpolate(streamline, x_points):

    arr = np.array(streamline)

    # set arrays for interpolation
    xp = arr[:,0][::-1]
    zp = arr[:,2][::-1]

    #interpolate z coordinates
    z_points = np.interp(x_points, xp, zp, left=np.NaN, right=np.NaN)

    return np.array([x_points, z_points])


def make_streamlines(vlsvFileName):
    ## make streamlines
    boxre = [0,0]
                            
    # bulk file
    f = pt.vlsvfile.VlsvReader(file_name=vlsvFileName)

    #get box coordinates from data
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    [xsize, ysize, zsize] = f.get_spatial_mesh_size()
    simext_m =[xmin,xmax,ymin,ymax,zmin,zmax]
    sizes = np.array([xsize,ysize,zsize])

    def to_Re(m): #meters to Re
        return m/6371000

    simext = list(map(to_Re, simext_m))
    boxcoords=list(simext)

    cellids = f.read_variable("CellID")

    #Read the data from vlsv-file
    Vx = f.read_variable("v", operator="x")
    Vz = f.read_variable("v", operator="z")


    #Re-shape variable data
    Vxs=Vx[np.argsort(cellids)].reshape(f.get_spatial_mesh_size(), order="F")
    Vys = np.zeros_like(Vxs)
    Vzs=Vz[np.argsort(cellids)].reshape(f.get_spatial_mesh_size(), order="F")

    data=dict(Vx=Vxs,Vy=Vys,Vz=Vzs)

    #Create streamline seeds (starting points for streamlines)
    seedN = 200 # number of streamlines wanted
    streamline_seeds = np.array([[20, 0 ,i] for i in  np.linspace(-4, 4, seedN)])

    #streamline_seeds = np.array(streamline_seeds)
    #dataset in yt-form
    yt_dataset = yt.load_uniform_grid(
        data,
        sizes,
        bbox=np.array([[boxcoords[0], boxcoords[1]],
                    [boxcoords[2],boxcoords[3]],
                    [boxcoords[4],boxcoords[5]]]))
                                    

    #data, seeds, dictionary positions, lenght of lines
    streamlines_pos = yt.visualization.api.Streamlines(yt_dataset, streamline_seeds,
                                                    "Vx", "Vy", "Vz", length=40, direction=1)

    #where the magic happens
    streamlines_pos.integrate_through_volume()
    return np.array(streamlines_pos.streamlines)

def make_magnetopause(streams):

    streampoints = np.reshape(streams, (streams.shape[0]*streams.shape[1], 3)) #all the points in one array
    
    ## find the subsolar dayside point in the positive x-axis
    ## do this by finding a stremline point on positive x axis closest to the Earth
    x_axis_points = streampoints[np.floor(streampoints[:,2])==0]
    x_axis_points[x_axis_points<0] = 800
    subsolar_x =np.min(x_axis_points[:,0])

    ## define points in the x axis where to find magnetopause points on the yz-plane
    x_points = np.arange(subsolar_x, -10, -0.2)
    
    ## interpolate more exact points for streamlines at exery x_point
    new_streampoints = np.zeros((len(x_points), len(streams), 1)) # new array for keeping interpolated streamlines in form streamlines_new[x_point, streamline, z-coordinate] 
    i=0
    for stream in streams:
        interpolated_streamline = interpolate(stream, x_points)
        for j in range(0, len(x_points)):
            new_streampoints[j, i,:] = interpolated_streamline[1,j]
        i += 1


    ## start making the magnetopause
    ## in each x_point, find the closest streamline to x-axis in the positive and negative z-axis

    pos_z_mpause = np.zeros((len(x_points), 2))
    neg_z_mpause =  np.zeros((len(x_points), 2))

    for i, x_point in enumerate(x_points):
        pos = new_streampoints[i, new_streampoints[i,:] > 0]
        neg = new_streampoints[i, new_streampoints[i,:] < 0]

        if (pos.size == 0) or (neg.size == 0):
            raise ValueError('No streamlines found for x axis point, try adding streamlines or checking the x_points')
        
        # find points closest to x-axis and save found points
        pos_z_mpause[i] = [x_point, pos[pos.argmin()]]
        neg_z_mpause[i] = [x_point, neg[neg.argmax()]]

    magnetopause = np.concatenate((pos_z_mpause[::-1], np.array([[subsolar_x, 0]]),  neg_z_mpause))

    return magnetopause


def polar_to_cartesian(r, phi):
    phi = np.deg2rad(phi)
    y = r * np.cos(phi)
    z = r * np.sin(phi)
    return(y, z)



def main():

    ## get bulk data
    run = 'BFD'
    num =  '2000'

    fileLocation="/wrk-vakka/group/spacephysics/vlasiator/2D/"+run+"/bulk/"
    fileN = "bulk.000"+num+".vlsv"
    
    
    ## STREAMLINES
    streams = make_streamlines(fileLocation+fileN)

    ## MAGNETOPAUSE
    magnetopause = make_magnetopause(streams)
    

    ## PLOTTING
    outdir=""

    # plot the magnetopause (and streamlines if needed) on top of colormap
    def external_plot(ax,XmeshXY=None, YmeshXY=None, pass_maps=None):
            plot_magnetopause = True
            plot_streamlines = False
        
            if plot_streamlines:
                for stream in streams:
                    ax.plot(stream[:,0], stream[:,2], color='paleturquoise', alpha=0.2)

            if plot_magnetopause:
                ax.plot(magnetopause[:,0], magnetopause[:,1], color='cyan', linewidth=1.0) 


    plot_colormap.plot_colormap(
        filename=fileLocation+fileN,
        outputdir=outdir,
        nooverwrite=None,
        var="rho", #var="beta_star",
        boxre = [-21,21,-21,21],
        title=None,
        draw=None, usesci=True, Earth=True,
        external = external_plot,
        run=run,
        colormap='inferno',
        )



if __name__ == "__main__":
    main()
