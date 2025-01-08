import analysator as pt
import sys, os
import numpy as np
import traceback

runs = []
runs.append( { 'name': 'ABC',
                 'verifydir': 'testpackage_vdf/ABC/', 
                 'fileLocation': '/proj/vlasov/2D/ABC/distributions/',
                 'pops': ['avgs'],
                 'time': 100,
                 'filename': None} )
runs.append( { 'name': 'BCQ',
                 'verifydir': 'testpackage_vdf/BCQ/', 
                 'fileLocation': '/proj/vlasov/2D/BCQ/bulk/',
                 'pops': ['avgs'],
                 'time': 1600,
                 'filename': None } )
runs.append( { 'name': 'BED', 
                 'verifydir': 'testpackage_vdf/BED/', 
                 'fileLocation': '/proj/vlasov/2D/BED/bulk/',
                 'pops': ['avgs'],
                 'time': 2000,
                 'filename': None } )
runs.append( { 'name': 'BFD',
                 'verifydir': 'testpackage_vdf/BFD/', 
                 'fileLocation': '/proj/vlasov/2D/BFD/bulk/',
                 'pops': ['proton','helium'],
                 'time': 1000,
                 'filename': None } )
runs.append( { 'name': 'BCQr',
                 'verifydir': 'testpackage_vdf/BCQr/', 
                 'fileLocation': '/proj/vlasiato/BCQ/',
                 'pops': ['avgs'],
                 'time': 0,
                 'filename': 'restart.0001361.vlsv' } )
runs.append( { 'name': 'BFDr',
                 'verifydir': 'testpackage_vdf/BFDr/', 
                 'fileLocation': '/proj/vlasov/2D/BFD/restart/',
                 'pops': ['avgs'],
                 'time': 0,
                 'filename': 'restart.0001126.2018-06-03_21-34-16.vlsv' } )
                     
regularcalls = [
# Input and output methods, nooverwrite
"pt.plot.plot_vdf(filename=fileLocation+bulkname, outputdir=outputLocation+'/'+REPLACEINDEX+'_', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(filedir=fileLocation, step=REPLACETIME, run=verifydir+REPLACEINDEX, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, outputfile=outputLocation+REPLACEINDEX+'_outputfiletest.png', nooverwrite=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, outputfile=outputLocation+REPLACEPREVINDEX+'_outputfiletest.png', nooverwrite=1, coordre=REPLACECOORDRE)",

# cellids, coordinates
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, cellids=REPLACECELLID)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, coordinates=REPLACECOORDINATES)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, cellids=REPLACEMULTIPLECELLID)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, coordinates=REPLACEMULTIPLECOORDINATES)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, coordre=REPLACEMULTIPLECOORDRE)",


# Thickness, scale
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=0.5, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, thick=2.0, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=0.5, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, scale=2.0, coordre=REPLACECOORDRE)",

# Tick interval
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=1000, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=500, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=0.5,axisunit=6, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, tickinterval=1,axisunit=6, coordre=REPLACECOORDRE)",

# msec musec titles
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='msec', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='musec', coordre=REPLACECOORDRE)",

# B vector
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bvector=1, coordre=REPLACECOORDRE)",

# Zoom and units
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, box=[-2e6,2e6,-2e6,2e6], coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, box=[-2e6,2e6,-2e6,2e6],axisunit=0, coordre=REPLACECOORDRE)",

# Watermarks
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, wmarkb=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='NE', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='NW', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='SE', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, wmark='SW', coordre=REPLACECOORDRE)",

# Biglabels
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, biglabel='A', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, biglabel='B', biglabloc=0, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, biglabel='C', biglabloc=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, biglabel='D', biglabloc=2, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, biglabel='E', biglabloc=3, coordre=REPLACECOORDRE)",

# title, axes, noborders
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title=r'$\mathcal{Title}$ and so forth $\odot$', cbtitle=r'$\mathcal{Color}$', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noborder=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noxlabels=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noxlabels=1,noborder=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1,noborder=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',cbtitle='',noylabels=1,noxlabels=1,noborder=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,nocb=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, title='',noylabels=1,noxlabels=1,noborder=1,internalcb=1, coordre=REPLACECOORDRE)",

# slicethick
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, slicethick=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, slicethick=0, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, slicethick=2, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, slicethick=4, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, slicethick=1e3, coordre=REPLACECOORDRE)",

# cellsize
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, cellsize=0.5, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, cellsize=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, cellsize=2, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, cellsize=4, coordre=REPLACECOORDRE)",

# fmin, fmax, setThreshold
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, fmin=1.e-14, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, fmax=1.e-12, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, fmin=1.e-14,fmax=1.e-12, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, setThreshold=1.e-20, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, setThreshold=1.e-15, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, setThreshold=0, coordre=REPLACECOORDRE)",

# colormaps 
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='nipy_spectral', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='jet', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='hot_desaturated', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='hot_desaturated_r', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='viridis', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='plasma', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='magma', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='warhol', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='bwr', coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, colormap='PuOr', coordre=REPLACECOORDRE)",

# cbulk, center, bvector
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bvector=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bvector=1, cbulk=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bvector=1, center=[-7e5,0,0], coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bvector=1, center=[2e5,2e5,2e5], coordre=REPLACECOORDRE)",

# wflux
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, wflux=1, coordre=REPLACECOORDRE)",

# directions
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, xy=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, xy=1,coordswap=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, normal=[0,0,5], coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, xz=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, xz=1,coordswap=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, normal=[0,1,0], coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, yz=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, yz=1,coordswap=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, normal=[-1,0,0], coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bpara=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bpara=1,coordswap=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bperp=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, bperp=1,coordswap=1, coordre=REPLACECOORDRE)",
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, normal=[1,1,1], coordre=REPLACECOORDRE)",
]

multipopcalls = [
"pt.plot.plot_vdf(vlsvobj=f, run=verifydir+REPLACEINDEX, pop='REPLACEPOP', coordre=REPLACECOORDRE)"]



# count how many tests to run in total
ntests = []
for i,run in enumerate(runs):
    n = len(regularcalls)
    for pop in run['pops']:
        if pop!='avgs':
            n += len(multipopcalls)
    ntests.append(n)
nteststot = np.sum(np.array(ntests))

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

#print("nteststot",nteststot)
#print("jobcount",jobcount,"jobcurr",jobcurr,"start",start,"end",end)

for j in range(start,end):
    # Calculate which run
    jrun = j
    runid = 0
    while True:
        if jrun<ntests[runid]:
            break
        else:
            jrun -= ntests[runid]
            runid+=1

    run = runs[runid]['name']
    verifydir = runs[runid]['verifydir']
    fileLocation = runs[runid]['fileLocation']
    pops = runs[runid]['pops']
    time = runs[runid]['time']
    filename = runs[runid]['filename']

    outputLocation=os.path.expandvars('$HOME/Plots/'+verifydir)
    
    # Source data files
    bulkname = "bulk."+str(time).rjust(7,'0')+".vlsv"
    if run=="ABC":
        bulkname = "distributions."+str(time).rjust(7,'0')+".vlsv"
   

    # Special case for restart files
    if not filename is None:
        bulkname = filename

    if jrun<len(regularcalls):
        call = regularcalls[jrun]
    else:
        jrunmp = jrun-len(regularcalls)
        popid = int(jrunmp/len(multipopcalls))
        jrunmp = jrunmp % len(multipopcalls)
        call = multipopcalls[jrunmp].replace('REPLACEPOP',pops[popid])
    
    call = call.replace('REPLACEPREVINDEX',"'"+str(jrun-1).rjust(4,'0')+"'")
    call = call.replace('REPLACEINDEX',"'"+str(jrun).rjust(4,'0')+"'")
    call = call.replace('REPLACETIME',"'"+str(time)+"'")

    call = call.replace('REPLACECELLID','1')
    call = call.replace('REPLACECOORDRE','[10,0,0]')
    call = call.replace('REPLACECOORDINATES','[6.371e7,0,0]')
    call = call.replace('REPLACEMULTIPLECELLID','[1,51,101]')
    call = call.replace('REPLACEMULTIPLECOORDRE','[[10,0,0],[15,0,0],[20,0,0]]')
    call = call.replace('REPLACEMULTIPLECOORDINATES','[[6.371e7,0,0],[9.5565e7,0,0],[12.742e7,0,0]]')

    # Special case for restart files
    if not filename is None:
        if "step=" in call:
            continue

    # Special case for ABC distribution name construction
    if run=="ABC":
        if "step=" in call:
            continue
        

    # Many different plots
    print(j, runid, jrun, call)
    f = pt.vlsvfile.VlsvReader(fileLocation+bulkname)

    try:
        exec(call)
    except Exception as e:
        print("FAILURE IN CALL: \n",repr(e))
        traceback.print_exc()
