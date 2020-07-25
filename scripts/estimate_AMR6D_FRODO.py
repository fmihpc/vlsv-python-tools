import analysator as pt
import sys, os, socket
import numpy as np
import math
import scipy

# Script for attempting to estimate AMR 6D run costs for the FRODO GC run
# MCB 2.5.2018

cellsize = 1000.e3 # Cell size best refinement level (L2)

Ri = 5. # ionosphere boundary

R2 = 15. # Level 2 refinement sphere radius

R1 = 30. # Level 1 refinement area extent in cylindrical y-z-coordinates
parab_p1 = 20. # +x directional nose point of Level 1 parabola
parab_p2 = -0.010 # Level 1 parabola curvature
xrangep=-20. # Level 1 -x boundary

# Simulation outer extent (L0 boundary)
xrange=[-60.,40.]
yrange=[-50.,50.]
zrange=[-50.,50.]



# Helper function for drawing on existing panel                                                                         
def AMRcontours(ax, XmeshXY,YmeshXY, pass_maps):

    # Generate AMR regions
    circlerad = np.arange(0.,360.,1.)*(math.pi/180.)
    Rix = Ri * np.cos(circlerad)
    Riy = Ri * np.sin(circlerad)

    R2x = R2 * np.cos(circlerad)
    R2y = R2 * np.sin(circlerad)

    Paraby = np.arange(-R1,R1,0.1)
    Parabx = parab_p1 + parab_p2 * (Paraby**2)

    ax.plot(Rix,Riy,lw=1.0, color='k')
    ax.plot(R2x,R2y,lw=1.0, color='k')

    ax.plot(Parabx,Paraby,lw=1.0, color='k')
    ax.plot([Parabx[0],xrangep,xrangep,Parabx[-1]], [Paraby[0],-R1,R1,Paraby[-1]]
,lw=1.0, color='k')
    
    #ax.text(-4,0,'ionosphere')
    ax.annotate('ionosphere',xy=(0,0), xytext=(-40,-40), fontsize=10,
                arrowprops=dict(facecolor='black', width=0.5, headwidth=1.5))
    ax.text(-8,6,'level 2', fontsize=10)
    ax.text(-18,18,'level 1', fontsize=10)
    ax.text(15,38,'level 0', fontsize=10)

fileLocation="/proj/vlasov/2D/BCH/bulk/"
outputLocation=outputdir=os.path.expandvars('$HOME/BCH/')

timetot = [4070]
for j in timetot:
    # Source data file                                                                                                  
    bulkname = "bulk."+str(j).rjust(7,'0')+".vlsv"
    print(bulkname)
    BCHf = pt.vlsvfile.VlsvReader(fileLocation+bulkname)

    pt.plot.plot_colormap(vlsvobj=BCHf,
                          run="BCH",
                          colormap='viridis',
                          step=j,
                          outputdir=outputLocation,
                          var="Blocks",
                          external=AMRcontours,
                          boxre=[-60,40,-50,50],
                          title='', thick=1.2)


    # Save blocks and cells for use in regional averaging counts later on

    datamap = BCHf.read_variable("Blocks")
    [xsize, ysize, zsize] = BCHf.get_spatial_mesh_size()
    [xmin, ymin, zmin, xmax, ymax, zmax] = BCHf.get_spatial_mesh_extent()

    simext=[xmin,xmax,zmin,zmax]
    sizes=[xsize,zsize]
    cellids = BCHf.read_variable("CellID")
    datamap = datamap[cellids.argsort()].reshape([sizes[1],sizes[0]])
    [XmeshXY,YmeshXY] = scipy.meshgrid(np.linspace(simext[0],simext[1],num=sizes[0]),np.linspace(simext[2],simext[3],num=sizes[1]))

Re = 6.371e6
# V = 4/3 pi r^3

# Ionosphere volume (5 Re -> 1.354e23 m^3)
# Now calculate in Re
#V_iono = (4./3.)*math.pi*(Ri*Re)**3
V_iono = (4./3.)*math.pi*(Ri**3)

# level 2 volume (25 Re -> 1.693e25 m^3)
#V_l2 = (4./3.)*math.pi*(R2*Re)**3
V_l2 = (4./3.)*math.pi*(R2**3)
V_l2c = V_l2 - V_iono

# Level 1 volume: integrate volume within parabola of x = 30-0.012*(y^2+z^2)
# (where parab_p1 = 30, parab_p2 = -0.012
# over volume of xrangep to parabola and within sqrt(y^2+z^2) < R1
# This utilises some fancy indefinite integrals from Schaum

V_l1 = ((parab_p1-xrangep)*(R1**2) + 0.5*parab_p2*(R1**4))*math.pi
V_l1c = V_l1 - V_l2

# Level 0 refinement region (pristine solar wind, far flanks and tail)
V_l0 = (xrange[1]-xrange[0])*(yrange[1]-yrange[0])*(zrange[1]-zrange[0])
V_l0c = V_l0 - V_l1

cw_l2 = cellsize # cell width refinement x2 per level
cw_l1 = cw_l2 * 2
cw_l0 = cw_l1 * 2

cs_l2 = (cw_l2/Re)**3
cs_l1 = (cw_l1/Re)**3
cs_l0 = (cw_l0/Re)**3

cells_l2 = V_l2c/cs_l2
cells_l1 = V_l1c/cs_l1
cells_l0 = V_l0c/cs_l0

# Guesstimates from BCH
#avgblocks_l0 = 470.
#avgblocks_l1 = 4000.
#avgblocks_l2 = 14000.

# with vAMR could be something like this?
#avgblocks_l0 = 270.
#avgblocks_l1 = 2000.
#avgblocks_l2 = 4000.

# Alternative: calculate based on regions from BCH
Blocks_L2 = 0
Blocks_L1 = 0
Blocks_L0 = 0
counts_L2 = 0
counts_L1 = 0
counts_L0 = 0

if True: # A bit of a slow calculation
    for xi in np.arange(XmeshXY.shape[1]):
        for yi in np.arange(XmeshXY.shape[0]):

            Blocks = datamap[yi,xi]
            ix = XmeshXY[yi,xi] / Re
            iy = YmeshXY[yi,xi] / Re

            if ((ix**2 + iy**2) < R2**2): # L2 region
                Blocks_L2 = Blocks_L2 + Blocks
                counts_L2 = counts_L2 + 1
            elif ((ix < xrangep) or (abs(iy) > R1) or (ix > (parab_p1 + parab_p2*(iy**2)))):
                Blocks_L0 = Blocks_L0 + Blocks
                counts_L0 = counts_L0 + 1
            else:
                Blocks_L1 = Blocks_L1 + Blocks
                counts_L1 = counts_L1 + 1
                
    avgblocks_l2 = Blocks_L2*1./counts_L2
    avgblocks_l1 = Blocks_L1*1./counts_L1
    avgblocks_l0 = Blocks_L0*1./counts_L0
else:
    # saved values from previous calculation
    avgblocks_l0 = 3892.
    avgblocks_l1 = 8549.
    avgblocks_l2 = 6407.
       
# Note: BCH values were generated with:
# [sparse]
# minValue = 1.0e-15
# dynamicAlgorithm = 1
# dynamicBulkValue1 = 1.0e4
# dynamicBulkValue2 = 1.0e5
# dynamicMinValue1 = 1.0e-17
# dynamicMinValue2 = 1.0e-15
# So with adjusting the sparsity, block counts can be decreased somewhat even if vAMR isn't available.

print("   Volumes       c-widths    c-counts  blocks pc  tot blocks")
print("   [R_E^3]           [km]")
print("L0 %10.5e %10d %10.5e %10d %10.5e " % (V_l0c, cw_l0*1.e-3, cells_l0, avgblocks_l0, cells_l0*avgblocks_l0))
print("L1 %10.5e %10d %10.5e %10d %10.5e " % (V_l1c, cw_l1*1.e-3, cells_l1, avgblocks_l1, cells_l1*avgblocks_l1))
print("L2 %10.5e %10d %10.5e %10d %10.5e " % (V_l2c, cw_l2*1.e-3, cells_l2, avgblocks_l2, cells_l2*avgblocks_l2))

blockcount = (cells_l0*avgblocks_l0 + cells_l1*avgblocks_l1 + cells_l2*avgblocks_l2)
psccount = blockcount*64 # phase-space cells
memuse = psccount*22*(1./(1024**3)) # estimated memory use due to ghost cells and intermediate arrays
nodes = memuse / 60 # each sisu node has about 60 GB for use
print("Total block count %10.5e " % blockcount)
print("Total psc count %10.5e " % psccount)
print("Total memory use [GiB] %10.5f " % memuse)
print("Sisu nodes required %10.5f " % nodes)


