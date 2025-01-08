import analysator as pt
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps
import matplotlib
import os


# Avoids opening a figure window
if str(matplotlib.get_backend()) != 'Agg':
    plt.switch_backend('Agg') 

fileLocation="/proj/vlasov/2D/BCH/bulk/"
fluxLocation="/proj/vlasov/2D/BCH/flux/"
outputLocation=outputdir=os.path.expandvars('$HOME/Plots/')

# Earth radius
Re = 6371e3

# ===== Example of usage of the axes and cbaxes parameters in plot_colormap =====


# Geometry of the figure layout:
#     1 rectangular top panel for a colormap, with its colorbar
#     3 square bottom panels for VDFs, colorbar next to the third panel
mh = 0.10 # horizontal margin
mv = 0.14 # vertical margin
ecb = 0.015 # space between panel and colorbar
lcb = 0.02 # colorbar width
l = 0.175 # panel and colorbar height
ev = 0.1 # vertical spacing between panels
h = (1.-1.5*mv-ev)/2. # lower panel width
eh = (1.-3.*l-2.*mh-ecb-lcb)/2. # horizontal spacing between lower panels
L = 3.*l + 2.*eh # upper panel width
figH = l/h # figure width/height ratio

# Creating the sets of axes
fig=plt.figure(figsize=(10.,10.*figH)) 
ax1 = fig.add_axes([mh,mv+h+ev,L,h])
cax1 = fig.add_axes([mh+L+ecb,mv+h+ev,lcb,h])
ax2 = fig.add_axes([mh,mv,l,h])
ax3 = fig.add_axes([mh+l+eh,mv,l,h])
ax4 = fig.add_axes([mh+2.*(l+eh),mv,l,h])
cax4 = fig.add_axes([mh+3.*l+2.*eh+ecb,mv,lcb,h])

# Font size scaling factor
fontscale = 1.25

# Time step to plot
step=3532

# cellID for VDF plots
cellid = 3601601



# -- Top panel: colormap of Vx in the magnetotail --

filename = fileLocation+"bulk.000{}.vlsv".format(step)
xmin = -24.
xmax = -4.
ylim = (xmax-xmin)/2.*(h/L)*figH # to ensure an equal aspect ratio

pt.plot.plot_colormap(axes=ax1,cbaxes=cax1,filename=filename, var='V',op='x',run='BCH',outputdir=outputLocation,boxre=[xmin,xmax,-ylim,ylim], vmin=-1.5e6,vmax=1.5e6,colormap='RdBu_r',scale=fontscale,tickinterval=1.,fluxfile=fluxLocation+"bulk.000{}.bin".format(step),fluxlines=7)

# Adding extra information on the panel (VDF location, B-field direction)
vlsvReader=pt.vlsvfile.VlsvReader(filename)
xCid,yCid,zCid = vlsvReader.get_cell_coordinates(cellid)
VDFcoord = [xCid/Re,yCid/Re,zCid/Re]
Bvect = vlsvReader.read_variable("B", cellid)
Bvect = Bvect / np.sqrt(np.dot(Bvect,Bvect))
# Add dot at VDF location
ax1.scatter(VDFcoord[0], VDFcoord[2], color='black',marker='o',s=20)
ax1.scatter(VDFcoord[0], VDFcoord[2], color='white',marker='o',s=2)
# Add arrow for B-field direction
dx,dy,dz = Bvect[0],Bvect[1],Bvect[2]
ax1.arrow(xCid/Re+dx/15.,zCid/Re+dz/15.,dx,dz,color='#222222')



# -- Bottom panels: 3 slices of the VDF at selected location in magnetic frame --

vdflim=2.5e6 # m/s

# In (vB,vBxV) plane (no colorbar)
pt.plot.plot_vdf(axes=ax2,filename=filename,cellids=[cellid],box=[-vdflim,vdflim,-vdflim,vdflim],bpara1=1,fmin=1e-15,fmax=1e-11,colormap='nipy_spectral',title='',scale=fontscale,cbulk=1,nocb=1)

# In (vB,vBx(BxV)) plane (no colorbar)
pt.plot.plot_vdf(axes=ax3,filename=filename,cellids=[cellid],box=[-vdflim,vdflim,-vdflim,vdflim],bpara=1,fmin=1e-15,fmax=1e-11,colormap='nipy_spectral',title='',scale=fontscale,cbulk=1,nocb=1)

# In (vBxV,vBx(BxV)) plane (with common colorbar for the three panels)
pt.plot.plot_vdf(axes=ax4,cbaxes=cax4,filename=filename,cellids=[cellid],box=[-vdflim,vdflim,-vdflim,vdflim],bperp=1,fmin=1e-15,fmax=1e-11,colormap='nipy_spectral',title='',scale=fontscale,cbulk=1)
    


# -- Saving the figure --
figname = 'example_multi_panels_BCH_'+str(step)+'.png'
    
print('Saving figure as '+outputLocation+figname)
plt.savefig(outputLocation+figname,dpi=300)

