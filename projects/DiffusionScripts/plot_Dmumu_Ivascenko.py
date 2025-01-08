import analysator as pt
import numpy as np
import sys,os
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal

run    = sys.argv[1]
folder = sys.argv[2]

colours = ['blue','red']

bulkStart = int(sys.argv[3])
bulkEnd   = int(sys.argv[4])
interval  = int(sys.argv[5])

timetot = range(bulkStart, bulkEnd-2*interval+1,1)

#path_bulk = "/wrk-vakka/users/markusb/weirddiffusion/"+run+'/'+folder+'/'
path_bulk = "/scratch/project_2000203/dubartma/"+run+'/'+folder+'/bulk/'
path_save = '/scratch/project_2000203/dubartma/data/'+run+'/'+folder+'/'
path_fig  = '/users/dubartma/analysis/fig/'+run+'/'+folder+'/'

isExist = os.path.exists(path_fig)
if not isExist:
    os.makedirs(path_fig)
    print('Created '+ path_fig)

# Figure layout
xmargin   = 0.12
ymargin   = 0.11
vdfwidth  = 0.25
vdfheight = 0.35
axwidth   = 0.45
axheight  = 0.39
cbmargin  = 0.015
cbwidth   = 0.02
axspacew  = 0.05
axspaceh  = 0.06

Dmumu = np.load(path_save+'Dmumu/DmumuSplit_' + str(interval) +'.npy')

for j in timetot:

    bulks = [j, j + interval] 

    fig = plt.figure(figsize=(15,8))
    
    ax_fmu      = fig.add_axes([xmargin,ymargin+axheight+axspaceh*2-0.08,axwidth,axheight])
    ax_dmumu    = fig.add_axes([xmargin,ymargin,axwidth,axheight])
    ax_vdf_300  = fig.add_axes([xmargin+axwidth+axspacew*1.5,ymargin+vdfheight+axspaceh*2,vdfwidth,vdfheight])
    ax_vdf_diff = fig.add_axes([xmargin+axwidth+axspacew*1.5,ymargin,vdfwidth,vdfheight])
    vdf_cax     = fig.add_axes([xmargin+axwidth+axspacew*1.5+vdfwidth+cbmargin,ymargin,cbwidth,vdfheight*2+axspaceh*2])
    
    ax_fmu.set_xlim(-1,1)
    ax_fmu.set_ylim(0.0,3.0e6)
    #ax_fmu.set_ylim(0.1,80)
    #ax_fmu.set_ylim(3,6)
    #ax_fmu.set_yscale('log')
    ax_fmu.set_ylabel('$f(\\mu)$ [m$^{-3}$]',fontsize=20,weight='bold')    
    
    ax_fmu.xaxis.set_ticklabels([])
    ax_fmu.xaxis.set_tick_params(labelsize=18)
    ax_fmu.yaxis.set_tick_params(labelsize=19)
    
    #if run != 'Markus':
    #   ax_fmu.ticklabel_format(axis='y',style='sci',scilimits=(5,5))
    
    ax_vdf_300.xaxis.label.set_color(colours[0])
    ax_vdf_300.yaxis.label.set_color(colours[0])
    ax_vdf_300.tick_params(axis='both',colors=colours[0])
    ax_vdf_300.spines['top'].set_color(colours[0])
    ax_vdf_300.spines['bottom'].set_color(colours[0])
    ax_vdf_300.spines['left'].set_color(colours[0])
    ax_vdf_300.spines['right'].set_color(colours[0])
    
    ax_vdf_diff.xaxis.label.set_color(colours[1])
    ax_vdf_diff.yaxis.label.set_color(colours[1])
    ax_vdf_diff.tick_params(axis='both',colors=colours[1])
    ax_vdf_diff.spines['top'].set_color(colours[1])
    ax_vdf_diff.spines['bottom'].set_color(colours[1])
    ax_vdf_diff.spines['left'].set_color(colours[1])
    ax_vdf_diff.spines['right'].set_color(colours[1])
    
    
    ax_dmumu.set_xlim(-1,1)
    ax_dmumu.set_ylim(-0.005,0.1)
    ax_dmumu.set_xlabel('$\\mu$',fontsize=20,weight='bold')
    ax_dmumu.set_ylabel('D$_{\\mu\\mu}$ [s$^{-1}$]',fontsize=20,weight='bold')
    
    ax_dmumu.xaxis.set_tick_params(labelsize=17)
    ax_dmumu.yaxis.set_tick_params(labelsize=19)
    
    labels = ax_fmu.get_xticklabels() + ax_fmu.get_yticklabels() + ax_dmumu.get_xticklabels() + ax_dmumu.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    bulkname_start = 'bulk.'+str(bulks[0]).rjust(7,'0')+'.vlsv'
    bulkname_end   = 'bulk.'+str(bulks[1]).rjust(7,'0')+'.vlsv'
    
    f_start = pt.vlsvfile.VlsvReader(path_bulk + bulkname_start)
    f_end   = pt.vlsvfile.VlsvReader(path_bulk + bulkname_end)
    
    time_start = f_start.read_parameter('time')
    time_end   = f_end.read_parameter('time')
    
    dt = time_end - time_start
    
    CellID = 1
    
    # plot fmu and Dmumu
    fmu_start = np.mean(f_start.read_variable('vg_1dmuspace'),axis=0)
    fmu_end   = np.mean(f_end  .read_variable('vg_1dmuspace'),axis=0)

    mu_mid = np.linspace(-1.0,1.0,len(fmu_start))

    ax_fmu.plot(mu_mid,fmu_start,linewidth=2,color=colours[0],label='t = '+str(round(time_start,1))+'s')
    
    ax_fmu.plot(mu_mid,fmu_end,linewidth=2,color=colours[1],label='t = '+str(round(time_end,1))+'s')
    
    fmu_avg = (fmu_start + fmu_end)/2.0
    
    ax_fmu.ticklabel_format(axis='y',style='sci',scilimits=(6,6))

    # Ivascenko (savgol filter)
    Dmumu_int = Dmumu[j,:]
 
    # Analytical
    m     = 1.67262192e-27
    kB    = 1.380649e-23

    TparaStart  = np.mean(f_start.read_variable('vg_t_parallel'))
    TperpStart  = np.mean(f_start.read_variable('vg_t_perpendicular'))
    nStart      = np.mean(f_start.read_variable('vg_rho'))
    TanisoStart = np.mean(f_start.read_variable('vg_t_anisotropy'))

    TparaEnd  = np.mean(f_end.read_variable('vg_t_parallel'))
    TperpEnd  = np.mean(f_end.read_variable('vg_t_perpendicular'))
    nEnd      = np.mean(f_end.read_variable('vg_rho'))
    TanisoEnd = np.mean(f_end.read_variable('vg_t_anisotropy'))

    Tpara  = (TparaEnd + TparaStart)/2.0
    Tperp  = (TperpEnd + TperpStart)/2.0
    n      = (nEnd     + nStart)/2.0
    Taniso = (TanisoEnd + TanisoStart)/2.0

    dtTaniso = (TanisoEnd - TanisoStart)/dt
    dtn      = (nEnd - nStart)/dt

    DmumuEq29 = - (np.sqrt(Tpara)*Tperp)/(3.0*dt) * ( np.sqrt(TanisoEnd)/np.sqrt(1 + mu_mid**2 * (TanisoEnd - 1.0)) - np.sqrt(TanisoStart)/np.sqrt(1 + mu_mid**2 * (TanisoStart - 1.0)) )/( (1.0/Tpara - 1.0/Tperp) * (mu_mid**2/Tpara + (1.0-mu_mid**2)/Tperp)**(-5.0/2.0) )

    ax_dmumu.plot(mu_mid,DmumuEq29,linewidth=2,color='purple',label='Analytical')

    epsilonFit = 0.0 #min(Dmumu_int[indxA:indxB])
    nu0Fit     = np.arange(0.0,1.0,0.01)

    DmumuFit = np.empty([len(mu_mid),len(nu0Fit)])
    MSE      = np.empty(len(nu0Fit))
    for i in range(0,len(nu0Fit)):
        DmumuFit[:,i] = nu0Fit[i]/2 * (abs(mu_mid)/(1+abs(mu_mid)) + epsilonFit) * (1 - mu_mid**2)
        MSE[i]        = 1/len(Dmumu_int) * np.sum((DmumuFit[:,i] - Dmumu_int)**2)
        ax_dmumu.plot(mu_mid,DmumuFit[:,i],linewidth=2,color='green',alpha=0.2)

    minMSE = np.argmin(MSE)

    ax_dmumu.plot(mu_mid,Dmumu_int,linewidth=2,color='orange',label='Ivascenko et al.')
    ax_dmumu.plot(mu_mid,DmumuFit[:,minMSE],linewidth=2,color='green',label='Fit')

    pt.plot.plot_vdf(filename   = path_bulk+bulkname_start,
                     cellids    = [CellID],
                     xy         = 1,
                     box        = [-2.0e6,2.0e6,-2.0e6,2.0e6],
                     fmin       = 1e-11, fmax = 1e-7,
                     cbulk      = None,
                     axes       = ax_vdf_300,nocb=1,
                     slicethick = 1,
                     axisunit   = 6,
                     title      = '',
                     scale      = 2.5,
                     cbtitle    = '',
                     colormap   = 'viridis')
        

    pt.plot.plot_vdf(filename   = path_bulk+bulkname_end,
                     cellids    = [CellID],
                     xy         = 1,
                     box        = [-2.0e6,2.0e6,-2.0e6,2.0e6],
                     fmin       = 1e-11, fmax = 1e-7,
                     cbulk      = None,
                     axes       = ax_vdf_diff,cbaxes=vdf_cax,
                     slicethick = 1,
                     axisunit   = 6,
                     title      = '',
                     scale      = 2.5,
                     cbtitle    = '',
                     colormap   = 'viridis')
    
    cb_text = r'$f($v$)$ [m$^{-6}$ s$^{3}$]'
    ax_vdf_diff.text(2.0,-3.0,cb_text,fontsize=20,weight='bold')
    
    ax_fmu.legend(loc = 'lower center',fontsize=20)
    ax_dmumu.legend(loc = 'upper center',fontsize=13,ncol=3)
    
    ax_vdf_300.text(-1.9,1.6,'(c)',fontsize = 22, weight ='bold')
    ax_vdf_diff.text(-1.9,1.6,'(d)',fontsize = 22, weight ='bold')
    
    ax_fmu.text(0.01,0.9,'(a)',transform=ax_fmu.transAxes,fontsize=22,weight='bold')
    ax_dmumu.text(0.01,0.9,'(b)',transform=ax_dmumu.transAxes,fontsize=22,weight='bold')
    
    plt.suptitle('t = '+str(round((time_start+time_end)/2,2))+' s, $\\Delta$ t = 1.1s',fontsize=18,weight='bold')
    plt.savefig(path_fig+'Dmumu_'+run+'_'+folder+'_'+str(j).rjust(7,'0')+'.png',dpi=300)
    print(path_fig+'Dmumu_'+run+'_'+folder+'_'+str(j).rjust(7,'0')+'.png')
    plt.close()
