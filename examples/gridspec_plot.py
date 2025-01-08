import analysator as pt
import sys, os, socket
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from string import ascii_lowercase

# Example to showcase gridspec plotting

outputfile='gridspec.png'
bulkfile = "/wrk/group/spacephysics/vlasiator/2D/BCH/bulk/bulk.0003901.vlsv"

# We open the bulk file once to speed up reading
vf = pt.vlsvfile.VlsvReader(bulkfile)

#first create figure
fig=mpl.pyplot.figure(figsize=(17.,5.))  # width, height

# initialize gridspec with 16 rows and 42 columns of "pixels"
# pixels need not be square, but it can help with visualisation
# width-space and height-space indicate breathing room between pixels
gs = mpl.gridspec.GridSpec(16,42, wspace=0.25, hspace=0.125)

# main panel: 16 pixels wide, 14 pixels tall. Note that first we set rows (Y), and we leave one empty row before the image.
fig.add_subplot(gs[1:15, 0:16])

# Note: You can also index from the gridspec using e.g. [-3:-1, :] as per standard numpy indexing

# VDF panels: each is 6x6 "pixels"
# extra breather space left between rows of VDF plots but not colums
VDFos=[20,0] # Offset to start from (in Xoffset, Yoffset)
VDFsize=[6,6] # (Xsize, Ysize)
Ybreather=2
Xbreather=0

# we add 2 rows and 3 columns of VDF plots
NXVDF=3
NYVDF=2
for i in range(NYVDF):
    for j in range(NXVDF):
        # The gridspec array of "pixels" is accessed as [rows, columns]]
        row_start = VDFos[1]+i*(VDFsize[1]+Ybreather)
        column_start = VDFos[0]+j*(VDFsize[0]+Xbreather)
        fig.add_subplot(gs[row_start:row_start+VDFsize[1],
                        column_start:column_start+VDFsize[0]])

#colorbar
CBos=[VDFos[0]+NXVDF*VDFsize[0]+(NXVDF-1)*Xbreather, VDFos[1]+Ybreather] # Offset (X,Y) to start from
CBYsize = 2*VDFsize[1]
CBXsize = 1
fig.add_subplot(gs[CBos[1]:CBos[1]+CBYsize, CBos[0]:CBos[0]+CBXsize])

ax = fig.get_axes()


# Now plot main figure
pt.plot.plot_colormap(vlsvobj=vf, axes=ax[0])

# Plot VDFs
CellIDs=[4201801,4201851,4201901,4201951,4202001,4202051]

scale=1.5
fmin=1e-16
fmax=1e-11
boxsize=[-2e6,2e6,-2e6,2e6]

for i in range(2):
    for j in range(3):
        # Only plot colorbar once
        nocb=1
        cbaxes=None
        if j==0 and i==0:
            nocb=None
            cbaxes=ax[-1]
            
        cid = CellIDs[3*i+j]
        axes=ax[1+3*i+j]

        pt.plot.plot_vdf(vlsvobj=vf,cellids=[cid],axes=axes,center='bulk', scale=scale,
		         contours=10, title='', cbaxes=cbaxes, nocb=nocb, box=boxsize, axisunit=6, 
		         fmin=fmin,fmax=fmax, bpara1=1) 

mpl.pyplot.savefig(outputfile, dpi=300, bbox_inches='tight')
