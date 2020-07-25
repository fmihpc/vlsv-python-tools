import analysator as pt
import sys, os
import numpy as np
import traceback

runs = []
runs.append( { 'name': 'ABC',
                 'verifydir': 'testpackage_colormap/ABC/', 
                 'fileLocation': '/proj/vlasov/2D/ABC/bulk/',
                 'fluxLocation': '/proj/vlasov/2D/ABC/flux/',
                 'pops': ['avgs'],
                 'time': 1000,
                 'singletime': False,
                 'filename': None,
                 'nosubpops': False, # backstreaming / non-backstreaming
                 'vlasiator5': False,
                 'cavitonparams': [6.6e6,2.64e6,4.e-9,10] } )
runs.append( { 'name': 'BCQ',
                 'verifydir': 'testpackage_colormap/BCQ/', 
                 'fileLocation': '/proj/vlasov/2D/BCQ/bulk/',
                 'fluxLocation': '/proj/vlasov/2D/BCQ/flux/',
                 'pops': ['avgs'],
                 'time': 2000,
                 'singletime': False,
                 'filename': None,
                 'nosubpops': False, # backstreaming / non-backstreaming
                 'vlasiator5': False,
                 'cavitonparams': [2.0e6,0.8e6,4.e-9,10] } )
runs.append( { 'name': 'BFD',
                 'verifydir': 'testpackage_colormap/BFD/', 
                 'fileLocation': '/proj/vlasov/2D/BFD/bulk/',
                 'fluxLocation': '/proj/vlasov/2D/BFD/fluxfunction/',
                 'fluxprefix': 'bulk.',
                 'pops': ['proton','helium'],
                 'time': 2000,
                 'singletime': False,
                 'filename': None,
                 'nosubpops': False, # backstreaming / non-backstreaming
                 'vlasiator5': False,
                 'cavitonparams': [2.0e6,0.8e6,4.e-9,10] } )
runs.append( { 'name': 'BCQr',
                 'verifydir': 'testpackage_colormap/BCQr/', 
                 'fileLocation': '/proj/vlasiato/BCQ/',
                 'fluxLocation': None,
                 'pops': ['avgs'],
                 'time': 0,
                 'singletime': False,
                 'filename': 'restart.0001361.vlsv',
                 'nosubpops': False, # backstreaming / non-backstreaming
                 'vlasiator5': False,
                 'cavitonparams': [2.0e6,0.8e6,4.e-9,10] } )
runs.append( { 'name': 'AFC',
                 'verifydir': 'testpackage_colormap/AFC/', 
                 'fileLocation': '/proj/vlasov/2D/AFC/production_halfres/',
                 'fluxLocation': '/proj/vlasov/2D/AFC/production_halfres/flux/',
                 'pops': ['proton'],
                 'time': 1000,
                 'singletime': False,
                 'filename': None,
                 'nosubpops': False, # backstreaming / non-backstreaming
                 'vlasiator5': False,
                 'cavitonparams': [24.0e6,9.6e6,11.27e-9,10] } )
runs.append( { 'name': 'AFCr',
                 'verifydir': 'testpackage_colormap/AFCr/', 
                 'fileLocation': '/proj/vlasov/2D/AFC/production_halfres/',
                 'fluxLocation': None,
                 'pops': ['proton'],
                 'time': 0,
                 'singletime': False,
                 'filename': 'restart.0000592.2019-04-19_07-58-12.vlsv',
                 'nosubpops': False, # backstreaming / non-backstreaming
                 'vlasiator5': False,
                 'cavitonparams': [24.0e6,9.6e6,11.27e-9,10] } )
# runs.append( { 'name': 'BFDr',
#                  'verifydir': 'testpackage_colormap/BFDr/', 
#                  'fileLocation': '/proj/vlasov/2D/BFD/restart/',
#                  'fluxLocation': None,
#                  'pops': ['avgs'],
#                  'time': 0,
#                  'singletime': False,
#                  'filename': 'restart.0001126.2018-06-03_21-34-16.vlsv',
#                  'nosubpops': False, # backstreaming / non-backstreaming
#                  'vlasiator5': False,
#                  'cavitonparams': [2.0e6,0.8e6,4.e-9,10] } )
runs.append( { 'name': 'BFH',
                 'verifydir': 'testpackage_colormap/BFH/', 
                 'fileLocation': '/proj/vlasov/2D/BFH/zero_ehall_layers_23/',
                 'fluxLocation': None,
                 'pops': ['proton'],
                 'time': 380,
                 'singletime': True, # neighboring bulk files not available
                 'filename': None,
                 'vlasiator5': True,
                 'nosubpops': True, # thermal / non-thermal
                 'cavitonparams': [2.0e6,0.8e6,4.e-9,10] } )
                     
# Custom expression function                                                               
def exprMA_cust(exprmaps, requestvariables=False):
    if vlasiator5 is True:
        reqs = ['vg_va']
    else:
        reqs = ['va']
    if requestvariables==True:
        return reqs
    # exprmaps is a dictionary of numpy arrays
    # Each array has 2 dimensions [xsize, ysize]
    # or 3 dimensions [xsize, ysize, components]
    # here the function returns the M_A with a preset bulk velocity
    custombulkspeed=1500000. # m/s
    va = exprmaps[reqs[0]][:,:]
    MA = np.ma.divide(custombulkspeed,va)
    return MA

# Second example of a more involved custom expression function
def expr_cav_cust(pass_maps, requestvariables=False):
    if vlasiator5 is True:
        reqs= ['vg_rho','vg_b_vol','vg_beta']
    else:
        reqs= ['rho', 'B', 'beta']
    if requestvariables==True:
        return reqs
    # pass_maps is a dictionary of numpy arrays
    # Each array has 2 dimensions [ysize, xsize]
    # or 3 dimensions [ysize, xsize, components]

    # for time averaging, it's a list of dictionaries
    # Each dictionary contains an entry called 'dstep'
    # Which contains the relative time step position, i.e.
    # a value of 0 indicates no time offset.

    # This custom expression returns a map with values of
    # either 0 (solar wind), 0.5 (caviton), or 1.0 (SHFA), calculated against
    # time-averaged background values. This version doesn't do intelligent checks for the
    # format of the incoming data.
    if type(pass_maps) is not list:
        # Not a list of time steps, calculating this value does not make sense.
        print("expr_cav_cust expected a list of timesteps to average from, but got a single timestep. Exiting.")
        quit()

    # Multiple time steps were found
    ntimes = len(pass_maps)
    dsteps = [x['dstep'] for x in pass_maps]
    curri = dsteps.index(0)
    thesemaps = pass_maps[curri]
    
    thisrho = np.ma.masked_less_equal(thesemaps[reqs[0]][:,:], 0)
    thisB = np.ma.masked_less_equal(thesemaps[reqs[1]],0)
    thisbeta = np.ma.masked_less_equal(thesemaps[reqs[2]],0)
    thisBmag = np.linalg.norm(thisB, axis=-1)
        
    avgrho = np.zeros(np.array(thisrho.shape))
    avgBmag = np.zeros(np.array(thisrho.shape))
    # avgbeta = np.zeros(np.array(thisrho.shape))
    
    for i in range(ntimes):
        if i==curri: # Exclude current frame from background value averaging
            continue
        nowmaps = pass_maps[i]
        print(nowmaps['dstep'])
        avgrho = np.add(avgrho, nowmaps[reqs[0]])
        avgBcalc = np.linalg.norm(nowmaps[reqs[1]], axis=-1)
        avgBmag = np.add(avgBmag, avgBcalc)
        # avgbeta = np.add(avgbeta, nowmaps[2])

    avgrho = np.divide(np.ma.masked_less_equal(avgrho,0), np.array([ntimes-1]))
    avgBmag = np.divide(np.ma.masked_less_equal(avgBmag,0), np.array([ntimes-1]))
    #avgbeta = np.divide(np.ma.masked_less_equal(avgbeta,0), np.array([ntimes-1]))

    rhoratio = np.ma.divide(thisrho, avgrho)
    Bmagratio = np.ma.divide(thisBmag, avgBmag)
    #betaratio = np.divide(thisbeta, avgbeta)

    rhoratioreq = 0.9
    bmagratioreq = 0.9
    betashfareq = level_beta_SHFA

    # Calculations using masked arrays proved problematic so a less-than elegant method is used here.
    empty = np.zeros(np.array(thisrho.shape))+0.0
    half = empty + 0.5
    one = empty + 1.0
    caviton = np.add(empty, one, where=(rhoratio<rhoratioreq))
    print("sum of cavitons rho ",caviton.sum())
    caviton = np.add(caviton, one, where=(Bmagratio<bmagratioreq))
    print("sum of cavitons Bmag ",caviton.sum())
    shfa = np.add(caviton, one, where=(thisbeta>betashfareq))
    print("sum of SHFA ",shfa.sum())

    combo = np.add(empty, half, where=(caviton>1.5))
    print("sum of combo ",combo.sum())
    combo2 = np.add(empty, half, where=(shfa>2.5))
    print("sum of combo2 ",combo2.sum())
    combo3 = combo+combo2
    print("sum of combo3 ",combo3.sum())

    # Mask out anything that is inside the bow shock
    combo3 = np.ma.masked_where(thisrho>level_bow_shock, combo3)
    print("sum of combo3 upstream ",combo3.sum())

    return combo3


# Second example of a more involved custom expression function
def timesmooth(pass_maps):
    # pass_maps is a dictionary of numpy arrays
    # Each array has 2 dimensions [ysize, xsize]
    # or 3 dimensions [ysize, xsize, components]

    # for time averaging, it's a list of dictionaries
    # Each dictionary contains an entry called 'dstep'
    # Which contains the relative time step position, i.e.
    # a value of 0 indicates no time offset.

    # consists of a single [ysize,xsize] array to smooth over time

    if type(pass_maps) is not list:
        # Not a list of time steps, calculating this value does not make sense.
        print("timesmooth expected a list of timesteps to average from, but got a single timestep. Exiting.")
        quit()
    ntimes = len(pass_maps)
    print("this many time steps ",ntimes)
    # Select first valid variable
    listofkeys = iter(pass_maps[0])
    while True:
        var = next(listofkeys)
        if var!="dstep": break
    
    # Multiple time steps were found
    avg = np.zeros(pass_maps[0][var].shape)
    for i in range(ntimes):
        avg = np.add(avg, pass_maps[i][var])
    return np.divide(avg, np.array([ntimes]))

# Helper function for drawing on existing panel
def extcontour(ax, XmeshXY,YmeshXY, extmaps, requestvariables=False):
    if vlasiator5 is True:
        reqs= ['vg_rho','vg_b_vol','vg_beta']
    else:
        reqs= ['rho', 'B', 'beta']
    if requestvariables==True:
        return reqs

    rho = extmaps[reqs[0]]
    beta = extmaps[reqs[2]]
    # take magnitude of B
    B = np.linalg.norm(extmaps[reqs[1]],axis=-1)

    # Colours to use
    color_cavitons = '#924900'
    color_SHFAs    = '#B66DFF'
    color_BS       = '#FFFF6D'

    # mask cavitons
    cavitons = np.ma.masked_greater_equal(B,level_B_caviton)
    cavitons.mask[rho > level_n_caviton] = True
    cavitons.fill_value = 0.
    cavitons[cavitons.mask == False] = 1.

    # mask SHFAs
    SHFAs = np.ma.masked_greater_equal(B,level_B_caviton)
    SHFAs.mask[rho > level_n_caviton] = True
    SHFAs.mask[beta < level_beta_SHFA] = True
    SHFAs.fill_value = 0.
    SHFAs[SHFAs.mask == False] = 1.
 
    # draw contours
    contour_shock = ax.contour(XmeshXY,YmeshXY,rho,[level_bow_shock], 
                               linewidths=1.2, colors=color_BS,label='Bow shock')
    contour_cavitons = ax.contour(XmeshXY,YmeshXY,cavitons.filled(),[0.5], linewidths=1.5, colors=color_cavitons)  
    contour_SHFAs = ax.contour(XmeshXY,YmeshXY,SHFAs.filled(),[0.5], linewidths=1.5, colors=color_SHFAs)           


regularcalls = [
# Input and output methods, nooverwrite
"pt.plot.plot_colormap(filename=fileLocation+bulkname, outputdir=outputLocation+'/'+REPLACEINDEX+'_')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX)",
"pt.plot.plot_colormap(vlsvobj=f, outputfile=outputLocation+REPLACEINDEX+'_outputfiletest.png', nooverwrite=1)",
"pt.plot.plot_colormap(vlsvobj=f, outputfile=outputLocation+REPLACEPREVINDEX+'_outputfiletest.png', nooverwrite=1)",

# Thickness, scale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=2.0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=2.0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, highres=True)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=2.0, highres=True)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=2.0, highres=True)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, highres=3)",

# Tick interval
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=1)",

# msec musec titles
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='msec')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='musec')",

# Watermarks
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmarkb=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='NE')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='NW')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='SE')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='SW')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, Earth=True)",

# title, axes, noborders
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title=r'$\mathcal{Title}$ and so forth $\odot$', cbtitle=r'$\mathcal{Color}$')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noxlabels=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noxlabels=1,noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1,noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1,noxlabels=1,noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,nocb=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,internalcb=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,internalcb=1,highres=True)",

# Variables, operators, colormaps, usesci, lin, vscale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', colormap='nipy_spectral')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', colormap='nipy_spectral', vscale=1e9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='x', colormap='bwr')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='z', colormap='bwr')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='V', colormap='warhol',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='V', colormap='warhol',lin=1, vscale=1e-3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='V', op='x', colormap='PuOr')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='V', op='y', colormap='PuOr',symlog=0, usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='V', op='z', colormap='PuOr',symlog=0, usesci=0)"]

nonrestartcalls = [
# Overplots and flux lines
"pt.plot.plot_colormap(filedir=fileLocation, step=REPLACETIME, run=verifydir+REPLACEINDEX)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, fsaved=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, fsaved=1, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, fluxfile=fluxLocation+fluxname, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, fluxdir=fluxLocation, step=REPLACETIME, fluxthick=0.5, fluxlines=10)",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, fluxdir=fluxLocation, fluxthick=5, fluxlines=2)",

"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='E', colormap='hot_desaturated')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='E', colormap='hot_desaturated', vscale=1e3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='E', op='x', colormap='RdBu',symlog=0, usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='E', op='y', colormap='RdBu',symlog=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='E', op='z', colormap='RdBu',symlog=0, usesci=0)",

"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='RhoBackstream', colormap='jet')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='beta',lin=1, usesci=0, colormap='viridis',vmax=50)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='MA',lin=1,usesci=0,vmin=2,vmax=40, colormap='inferno_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Mms',lin=1,usesci=0, vmin=0, colormap='magma')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='va',lin=1,usesci=0, colormap='magma_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vms',lin=1, colormap='plasma_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vs',lin=1, colormap='viridis_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='max_v_dt', vscale=1e6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='max_v_dt', vscale=1e3)",

# Vscale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vmin=7.e3, vmax=7.e6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vmin=7.e-3, vmax=7.e0, vscale=1e-6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vmin=7.e6, vmax=7.e9, vscale=1e3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Temperature', colormap='plasma', vscale=1e-6,lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Pressure', vscale=1e9)",

# Symlog and vscale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='y', colormap='bwr',symlog=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='y', colormap='bwr',symlog=1e-9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='y', colormap='bwr',symlog=1e-12)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='y', colormap='bwr',symlog=0,vscale=1e9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='y', colormap='bwr',symlog=1,vscale=1e9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='B', op='y', colormap='bwr',symlog=1e-3,vscale=1e9)",

# Zoom and units
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxre=[-10,10,5,50],axisunit=3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxre=[-10,10,5,50],axisunit=6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxm=[-10e6,50e6,-5e6,15e6])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxm=[-10e6,50e6,-5e6,15e6],axisunit=0)",

# Externals and expressions
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, external=extcontour, pass_vars=['rho','B','beta'])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, external=extcontour, boxre=[0,30,-15,15])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, expression=exprMA_cust, pass_vars=['va'], vmin=1, vmax=20,lin=1,usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, expression=exprMA_cust, boxre=[0,30,-15,15], vmin=1, vmax=20,lin=1,usesci=0)",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=expr_cav_cust, pass_times=3, pass_vars=['rho','B','beta'],lin=1,colormap='bwr',usesci=0)",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=expr_cav_cust, pass_times=3,lin=1,colormap='bwr',usesci=0, boxre=[0,30,-15,15])",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=timesmooth, pass_times=[7,0], pass_vars=['rho'], boxre=[0,30,-15,15])",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=timesmooth, pass_times=[7,0], pass_vars=['beta'])",

# Everything at once
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, external=extcontour, boxre=[0,30,-15,15], expression=exprMA_cust, vmin=1, vmax=20,lin=1,usesci=0, fsaved=1, fluxfile=fluxLocation+fluxname)",

# Streamlines, vectors
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B', vectorcolormap='viridis')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B', vectorcolormap='magma')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B',vectordensity=20)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B',vectordensity=400)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B',vectordensity=20, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='B',vectordensity=400, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=20)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=400)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=20, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=400, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=20,vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=400,vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',boxre=[-10,10,5,50],vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=20, boxre=[-10,10,5,50],vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=400, boxre=[-10,10,5,50],vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=20,vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=400,vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',boxre=[-10,10,5,50],vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=20, boxre=[-10,10,5,50],vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='E',vectordensity=400, boxre=[-10,10,5,50],vectorsize=0.5)",


"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B', streamlinecolor='black')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B', streamlinecolor='gray')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B', streamlinedensity=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B', streamlinedensity=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B', streamlinedensity=0.5, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='B', streamlinedensity=2, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=0.5,streamlinethick=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinethick=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=2,streamlinethick=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=0.5,streamlinethick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinethick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=2,streamlinethick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=0.5, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='V', streamlinedensity=2, boxre=[-10,10,5,50])",

# More data reducers
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VBackstream',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VNonBackstream',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VParallel', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VPerpendicular',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VParallelBackstream', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VPerpendicularBackstream',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VParallelNonBackstream', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='VPerpendicularNonBackstream',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Pressure')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PParallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PPerpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PPerpOverPar', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PParallelBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PPerpendicularBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PPerpOverParBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PParallelNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PPerpendicularNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='PPerpOverParNonBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Pdyn',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Pdynx',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Temperature')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TParallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TPerpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TParallelBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TPerpendicularBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TParallelNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TPerpendicularNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TPerpOverPar', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TPerpOverParBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='TPerpOverParNonBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='betaPerpOverPar', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='betaPerpOverParBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='betaPerpOverParNonBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='beta')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='betaParallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='betaPerpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Rmirror',lin=1,vmin=0.5,vmax=1.5,usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vBeam',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vBeamRatio')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Thermalvelocity',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='Blocks')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='aGyrotropy')"]


multipopcalls = [
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/VParallel', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/VPerpendicular',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Pressure')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PParallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PPerpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PPerpOverPar', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PParallelBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PPerpendicularBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PPerpOverParBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PParallelNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PPerpendicularNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/PPerpOverParNonBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Pdyn',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Pdynx',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Temperature')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TParallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TPerpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TParallelBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TPerpendicularBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TParallelNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TPerpendicularNonBackstream')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TPerpOverPar', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TPerpOverParBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/TPerpOverParNonBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/betaPerpOverPar', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/betaPerpOverParBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/betaPerpOverParNonBackstream', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/beta')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/betaParallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/betaPerpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Rmirror',lin=1,vmin=0.5,vmax=1.5,usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Thermalvelocity',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/Blocks')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/aGyrotropy')"]

v5regularcalls = [
# Input and output methods, nooverwrite
"pt.plot.plot_colormap(filename=fileLocation+bulkname, outputdir=outputLocation+'/'+REPLACEINDEX+'_')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX)",
"pt.plot.plot_colormap(filedir=fileLocation, step=REPLACETIME, run=verifydir+REPLACEINDEX)",
"pt.plot.plot_colormap(vlsvobj=f, outputfile=outputLocation+REPLACEINDEX+'_outputfiletest.png', nooverwrite=1)",
"pt.plot.plot_colormap(vlsvobj=f, outputfile=outputLocation+REPLACEPREVINDEX+'_outputfiletest.png', nooverwrite=1)",

# Thickness, scale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=5.0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=2.0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, highres=True)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=2.0, highres=True)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=2.0, highres=True)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, highres=3)",

# Tick interval
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=1)",

# msec musec titles
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='msec')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='musec')",

# Watermarks
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmarkb=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='NE')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='NW')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='SE')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='SW')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, Earth=True)",

# title, axes, noborders
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title=r'$\mathcal{Title}$ and so forth $\odot$', cbtitle=r'$\mathcal{Color}$')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noxlabels=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noxlabels=1,noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1,noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1,noxlabels=1,noborder=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,nocb=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,internalcb=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,internalcb=1,highres=True)",

# Variables, operators, colormaps, usesci, lin, vscale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', colormap='nipy_spectral')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', colormap='nipy_spectral', vscale=1e9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='x', colormap='bwr')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='z', colormap='bwr')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v', colormap='warhol',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v', colormap='warhol',lin=1, vscale=1e-3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v', op='x', colormap='PuOr')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v', op='y', colormap='PuOr',symlog=0, usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v', op='z', colormap='PuOr',symlog=0, usesci=0)"]

v5nonrestartcalls = [
# Overplots and flux lines
"pt.plot.plot_colormap(filedir=fileLocation, step=REPLACETIME, run=verifydir+REPLACEINDEX)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, fsaved=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, fsaved=1, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, fluxfile=fluxLocation+fluxname, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, fluxdir=fluxLocation, step=REPLACETIME, fluxthick=0.5, fluxlines=10)",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, fluxdir=fluxLocation, fluxthick=5, fluxlines=2)",

# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_e_vol', colormap='hot_desaturated')",
# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_e_vol', colormap='hot_desaturated', vscale=1e3)",
# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_e_vol', op='x', colormap='RdBu',symlog=0, usesci=0)",
# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_e_vol', op='y', colormap='RdBu',symlog=0)",
# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_e_vol', op='z', colormap='RdBu',symlog=0, usesci=0)",

"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_rho_nonthermal', colormap='jet')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta',lin=1, usesci=0, colormap='viridis',vmax=50)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_MA',lin=1,usesci=0,vmin=2,vmax=40, colormap='inferno_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_Mms',lin=1,usesci=0, vmin=0, colormap='magma')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_va',lin=1,usesci=0, colormap='magma_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_vms',lin=1, colormap='plasma_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_vs',lin=1, colormap='viridis_r')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_maxdt_acceleration', vscale=1e6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_maxdt_acceleration', vscale=1e3)",

# Vscale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vmin=7.e3, vmax=7.e6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vmin=7.e-3, vmax=7.e0, vscale=1e-6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vmin=7.e6, vmax=7.e9, vscale=1e3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_temperature', colormap='plasma', vscale=1e-6,lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_pressure', vscale=1e9)",

# Symlog and vscale
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='y', colormap='bwr',symlog=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='y', colormap='bwr',symlog=1e-9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='y', colormap='bwr',symlog=1e-12)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='y', colormap='bwr',symlog=0,vscale=1e9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='y', colormap='bwr',symlog=1,vscale=1e9)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_b_vol', op='y', colormap='bwr',symlog=1e-3,vscale=1e9)",

# Zoom and units
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxre=[-10,10,5,50],axisunit=3)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxre=[-10,10,5,50],axisunit=6)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxm=[-10e6,50e6,-5e6,15e6])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, boxm=[-10e6,50e6,-5e6,15e6],axisunit=0)",

# Externals and expressions
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, external=extcontour, pass_vars=['vg_rho','vg_b_vol','vg_beta'])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, external=extcontour, boxre=[0,30,-15,15])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, expression=exprMA_cust, pass_vars=['vg_va'], vmin=1, vmax=20,lin=1,usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, expression=exprMA_cust, boxre=[0,30,-15,15], vmin=1, vmax=20,lin=1,usesci=0)",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=expr_cav_cust, pass_times=3, pass_vars=['vg_rho','vg_b_vol','vg_beta'],lin=1,colormap='bwr',usesci=0)",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=expr_cav_cust, pass_times=3,lin=1,colormap='bwr',usesci=0, boxre=[0,30,-15,15])",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=timesmooth, pass_times=[7,0], pass_vars=['vg_rho'], boxre=[0,30,-15,15])",
"pt.plot.plot_colormap(filename=fileLocation+bulkname, run=verifydir+REPLACEINDEX, expression=timesmooth, pass_times=[7,0], pass_vars=['vg_beta'])",

# Everything at once
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, external=extcontour, boxre=[0,30,-15,15], expression=exprMA_cust, vmin=1, vmax=20,lin=1,usesci=0, fsaved=1, fluxfile=fluxLocation+fluxname)",

# Streamlines, vectors
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol', vectorcolormap='viridis')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol', vectorcolormap='magma')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol',vectordensity=20)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol',vectordensity=400)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol',vectordensity=20, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_b_vol',vectordensity=400, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=20)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=400)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=20, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=400, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=20,vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=400,vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',boxre=[-10,10,5,50],vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=20, boxre=[-10,10,5,50],vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=400, boxre=[-10,10,5,50],vectorsize=1.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=20,vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=400,vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',boxre=[-10,10,5,50],vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=20, boxre=[-10,10,5,50],vectorsize=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, vectors='vg_v',vectordensity=400, boxre=[-10,10,5,50],vectorsize=0.5)",

"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol', streamlinecolor='black')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol', streamlinecolor='gray')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol', streamlinedensity=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol', streamlinedensity=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol', streamlinedensity=0.5, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_b_vol', streamlinedensity=2, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=0.5,streamlinethick=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinethick=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=2,streamlinethick=2)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=0.5,streamlinethick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinethick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=2,streamlinethick=0.5)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=0.5, boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', boxre=[-10,10,5,50])",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, streamlines='vg_v', streamlinedensity=2, boxre=[-10,10,5,50])",

# More data reducers
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_nonthermal',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_thermal',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_parallel', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_perpendicular',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_parallel_nonthermal', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_perpendicular_nonthermal',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_parallel_thermal', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_v_perpendicular_thermal',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_pressure')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_parallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_perpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_anisotropy', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_parallel_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_perpendicular_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_anisotropy_nonthermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_parallel_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_perpendicular_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_p_anisotropy_thermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_pdyn',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_pdynx',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_temperature')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_parallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_perpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_parallel_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_perpendicular_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_parallel_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_perpendicular_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_anisotropy', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_anisotropy_nonthermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_t_anisotropy_thermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta_anisotropy', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta_anisotropy_nonthermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta_anisotropy_thermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta_parallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_beta_perpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_rmirror',lin=1,vmin=0.5,vmax=1.5,usesci=0)",
# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_vBeam',lin=1)",
# "pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_vBeamRatio')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='vg_agyrotropy')"]

v5multipopcalls = [
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_v_parallel', op='magnitude',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_v_perpendicular',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_pressure')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_parallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_perpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_anisotropy', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_parallel_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_perpendicular_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_anisotropy_nonthermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_parallel_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_perpendicular_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_p_anisotropy_thermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_pdyn',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_pdynx',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_temperature')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_parallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_perpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_parallel_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_perpendicular_nonthermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_parallel_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_perpendicular_thermal')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_anisotropy', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_anisotropy_nonthermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_t_anisotropy_thermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_beta_anisotropy', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_beta_anisotropy_nonthermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_beta_anisotropy_thermal', vmin=0.1, vmax=10)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_beta')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_beta_parallel')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_beta_perpendicular')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_rmirror',lin=1,vmin=0.5,vmax=1.5,usesci=0)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_thermalvelocity',lin=1)",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_blocks')",
"pt.plot.plot_colormap(vlsvobj=f, run=verifydir+REPLACEINDEX, var='REPLACEPOP/vg_agyrotropy')"]

# Construct test list
calls = []
callrunids = []
callrunindex = []
for i,run in enumerate(runs):
    # bulk and restart files
    vlasiator5 = run['vlasiator5']
    filename = run['filename']
    fileLocation = run['fileLocation']
    singletime = run['singletime']
    nosubpops = run['nosubpops']
    fluxLocation = run['fluxLocation']

    callindex = 0
    if vlasiator5:
        for call in v5regularcalls:
            if not filename is None: 
                call = call.replace("var='V'","var='restart_V'")
            callrunids.append(i)
            calls.append(call)
            callrunindex.append(callindex)
            callindex += 1
    else:
        for call in regularcalls:
            if not filename is None: 
                call = call.replace("var='V'","var='restart_V'")
            callrunids.append(i)
            calls.append(call)                
            callrunindex.append(callindex)
            callindex += 1
    # non-restart files
    if filename is None:
        if vlasiator5:
            for call in v5nonrestartcalls:
                # Skip flux function calls if no flux files
                if "flux" in call and fluxLocation is None:
                    continue
                # skip time integration if only one file available
                if "pass_times" in call and singletime:
                    continue
                # thermal / non-thermal subpopulations
                if (("_thermal" in call) or ("_nonthermal" in call)) and nosubpops:
                    continue
                callrunids.append(i)
                calls.append(call)
                callrunindex.append(callindex)
                callindex += 1
            for pop in run['pops']:
                if pop is not 'avgs':
                    for call in v5multipopcalls:
                        # Skip flux function calls if no flux files
                        if "flux" in call and fluxLocation is None:
                            continue
                        # skip time integration if only one file available
                        if "pass_times" in call and singletime:
                            continue
                        # thermal / non-thermal subpopulations
                        if (("_thermal" in call) or ("_nonthermal" in call)) and nosubpops:
                            continue
                        mpcall = call.replace('REPLACEPOP',pop)
                        callrunids.append(i)
                        calls.append(mpcall)
                        callrunindex.append(callindex)
                        callindex += 1
        else:
            for call in nonrestartcalls:
                # Skip flux function calls if no flux files
                if "flux" in call and fluxLocation is None:
                    continue
                # skip time integration if only one file available
                if "pass_times" in call and singletime:
                    continue
                # thermal / non-thermal subpopulations
                if (("_backstream" in call) or ("_nonbackstream" in call)) and nosubpops:
                    continue
                callrunids.append(i)
                calls.append(call)
                callrunindex.append(callindex)
                callindex += 1
            for pop in run['pops']:
                if pop is not 'avgs':
                    for call in multipopcalls:
                        # Skip flux function calls if no flux files
                        if "flux" in call and fluxLocation is None:
                            continue
                        # skip time integration if only one file available
                        if "pass_times" in call and singletime:
                            continue
                        # thermal / non-thermal subpopulations
                        if (("_backstream" in call) or ("_nonbackstream" in call)) and nosubpops:
                            continue
                        mpcall = call.replace('REPLACEPOP',pop)
                        callrunids.append(i)
                        calls.append(mpcall)
                        callrunindex.append(callindex)
                        callindex += 1

nteststot = len(callrunids)



# How many jobs? 
jobcount=int(sys.argv[1])
jobcurr=int(sys.argv[2])
increment = int(nteststot/jobcount)
remainder = nteststot - jobcount * increment
start=jobcurr * increment
end=start + increment
# Remainder frames are divvied out evenly among tasks
if jobcurr < remainder:
    start = start + jobcurr
    end = end + jobcurr + 1
else:
    start = start + remainder
    end = end + remainder


# Perform call
for j in range(start,end):
    # Calculate which run
    jrun = callrunindex[j]
    runid = callrunids[j]
    call = calls[j]

    runname = runs[runid]['name']
    verifydir = runs[runid]['verifydir']
    fileLocation = runs[runid]['fileLocation']
    fluxLocation = runs[runid]['fluxLocation']
    pops = runs[runid]['pops']
    time = runs[runid]['time']
    filename = runs[runid]['filename']
    vlasiator5 = runs[runid]['vlasiator5']
    singletime = runs[runid]['singletime']

    level_bow_shock = runs[runid]['cavitonparams'][0]
    level_n_caviton = runs[runid]['cavitonparams'][1]
    level_B_caviton = runs[runid]['cavitonparams'][2]
    level_beta_SHFA = runs[runid]['cavitonparams'][3]

    outputLocation=os.path.expandvars('$HOME/Plots/'+verifydir)
    
    # Source data files
    if filename is None:
        bulkname = "bulk."+str(time).rjust(7,'0')+".vlsv"
    else:
        bulkname = filename
    if 'fluxprefix' in runs[runid]:
        fluxname = runs[runid]['fluxprefix']+str(time).rjust(7,'0')+".bin"
    else:
        fluxname = "flux."+str(time).rjust(7,'0')+".bin"
           
    call = call.replace('REPLACEPREVINDEX',"'"+str(jrun-1).rjust(4,'0')+"'")
    call = call.replace('REPLACEINDEX',"'"+str(jrun).rjust(4,'0')+"'")
    call = call.replace('REPLACETIME',"'"+str(time)+"'")

    # Many different plots
    print(j, runid, jrun, call)
    f = pt.vlsvfile.VlsvReader(fileLocation+bulkname)
    try:
        exec(call)
    except Exception as e:
        print("FAILURE IN CALL: \n",repr(e))
        traceback.print_exc()
