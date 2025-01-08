import numpy as np
import analysator as pt
import matplotlib
import matplotlib.pyplot as plt
import sys,os
import scipy.signal

run    = sys.argv[1]
folder = sys.argv[2]

path_bulk  = '/scratch/project_2000203/dubartma/'+run+'/'+folder+'/bulk/'
path_mu    = '/scratch/project_2000203/dubartma/data/'+run+'/'+folder+'/mu/'
path_dmumu = '/scratch/project_2000203/dubartma/data/'+run+'/'+folder+'/Dmumu/'
path_var   = '/scratch/project_2000203/dubartma/data/'+run+'/'+folder+'/variables/'

try:
    os.mkdir('/scratch/project_2000203/dubartma/'+run+'/'+folder)
except:
    pass
try:
    os.mkdir(path_mu)
except:
    pass
try:
    os.mkdir(path_dmumu)
except:
    pass
try:
    os.mkdir(path_var)
except:
   pass

bulkStart = int(sys.argv[3]) # 0 
bulkEnd   = int(sys.argv[4]) # 70/0.25 = 280

interval = int(sys.argv[5]) # 4
if interval < 1:
    raise Exception("Interval has to be at least 1")

twindow = int(sys.argv[6]) # +- 1

timetot = range(bulkStart,bulkEnd-2*interval+1,1)

nu0Analytic = np.empty(len(timetot))
nu0Fit      = np.empty(len(timetot))
nu0Mean     = np.empty(len(timetot))

Tpara  = np.load(path_var+'Tpara_' +run+'_'+folder+'_'+str(interval)+'.npy')
Tperp  = np.load(path_var+'Tperp_' +run+'_'+folder+'_'+str(interval)+'.npy')
n      = np.load(path_var+'n_'     +run+'_'+folder+'_'+str(interval)+'.npy')
Taniso = np.load(path_var+'Taniso_'+run+'_'+folder+'_'+str(interval)+'.npy')

DmumuL2R   = np.empty([len(timetot),30])
DmumuR2L   = np.empty([len(timetot),30])
DmumuSplit = np.empty([len(timetot),30])

for j in timetot:

    try:

        if (j-twindow) < 0:
            startbulks = np.arange(0,2*twindow+1,1,dtype=int) 
        else:
            startbulks = np.arange(j-twindow,j+twindow+1,1,dtype=int)

        if (j+interval+twindow) > (timetot[-1]+interval):
            endbulks = np.arange(timetot[-1]+interval-2*twindow,timetot[-1]+interval+1,1,dtype=int)
        else:
            endbulks = np.arange(j+interval-twindow,j+interval+twindow+1,1,dtype=int)

        print(startbulks)
        print(endbulks)

        startFile = pt.vlsvfile.VlsvReader(path_bulk+'bulk.'+str(j).rjust(7,'0')+'.vlsv')
        endFile   = pt.vlsvfile.VlsvReader(path_bulk+'bulk.'+str(j+interval).rjust(7,'0')+'.vlsv')

        tStart = startFile.read_parameter('time')
        tEnd   = endFile  .read_parameter('time')
        dt = abs(tStart - tEnd)
        print('dt = '+str(dt)+'s')

        muBins = startFile.read_variable('vg_1dmuspace')
        nbins  = muBins.shape[1]

        fStart = np.empty([len(startbulks),nbins])
        fEnd   = np.empty([len(endbulks),nbins])

        for i in range(0,len(startbulks)):
           fileStart   = pt.vlsvfile.VlsvReader(path_bulk+'bulk.'+str(startbulks[i]).rjust(7,'0')+'.vlsv')
           fStart[i,:] = np.mean(fileStart.read_variable('vg_1dmuspace'),axis=0)
           fileEnd     = pt.vlsvfile.VlsvReader(path_bulk+'bulk.'+str(endbulks[i]).rjust(7,'0')+'.vlsv')
           fEnd[i,:]   = np.mean(fileEnd.read_variable('vg_1dmuspace'),axis=0)

        start = np.mean(fStart,axis=0)
        end   = np.mean(fEnd,  axis=0)

        dmu = 2.0/len(start)
        mu  = np.linspace(-1.0,1.0,len(start))

        # dfdt
        dfdt = (end - start) / dt

        half = int(nbins/2)

        dfdmuStart = np.empty(nbins)
        dfdmuEnd   = np.empty(nbins)
        dfdmuStart[0]  = (start[1]  - start[0]) /dmu
        dfdmuStart[-1] = (start[-1] - start[-2])/dmu
        dfdmuEnd[0]    = (end[1]  - end[0]) /dmu
        dfdmuEnd[-1]   = (end[-1] - end[-2])/dmu
        for i in range(1,nbins-1):
            dfdmuStart[i] = (start[i+1] - start[i-1])/(2*dmu)
            dfdmuEnd[i]   = (end[i+1]   - end[i-1]  )/(2*dmu)

        dfdmu = (dfdmuStart + dfdmuEnd)/2

        integral_L2R   = np.empty(nbins)
        integral_R2L   = np.empty(nbins)
        integral_Split = np.empty(nbins)

        for i in range(0,nbins):
            integral_L2R[i]             = sum(dfdt[:i+1])*dmu # Left to Right
            integral_R2L[(nbins-1) - i] = -sum(dfdt[(nbins-i-1):nbins])*dmu # Right to Left

        # Split at 0
        for i in range(0,int(nbins/2)):
            integral_Split[i]             = sum(dfdt[:i+1])*dmu # Left to Right
            integral_Split[(nbins-1) - i] = -sum(dfdt[(nbins-i-1):nbins])*dmu # Right to Left

        DmumuL2R[j,:half-1] = integral_L2R[:half-1] / dfdmu[:half-1]
        DmumuL2R[j,half+1:] = integral_L2R[half+1:] / dfdmu[half+1:]
        DmumuL2R[j,:] = np.clip(DmumuL2R[j,:],0,None)

        DmumuR2L[j,:half-1] = integral_R2L[:half-1] / dfdmu[:half-1]
        DmumuR2L[j,half+1:] = integral_R2L[half+1:] / dfdmu[half+1:]
        DmumuR2L[j,:] = np.clip(DmumuR2L[j,:],0,None)
        
        DmumuSplit[j,:half-1] = integral_Split[:half-1] / dfdmu[:half-1]
        DmumuSplit[j,half+1:] = integral_Split[half+1:] / dfdmu[half+1:]
        DmumuSplit[j,:] = np.clip(DmumuSplit[j,:],0,None)

        epsilonFit = 0.0 #min(Dmumu_int[indxA:indxB])
        nu0FitAll  = np.arange(0.0,1.0,0.01)

        DmumuFit = np.empty([len(mu),len(nu0FitAll)])
        MSE      = np.empty(len(nu0FitAll))
        for i in range(0,len(nu0FitAll)):
            DmumuFit[:,i] = nu0FitAll[i]/2 * (abs(mu)/(1+abs(mu)) + epsilonFit) * (1 - mu**2)
            MSE[i]        = 1/len(DmumuSplit) * np.sum((DmumuFit[:,i] - DmumuSplit)**2)

        minMSE = np.argmin(MSE)
        nu0Fit[j] = nu0FitAll[minMSE]

        # Analytical
        m     = 1.67262192e-27
        kB    = 1.380649e-23

        TparaStart  = Tpara[j]
        TperpStart  = Tperp[j]
        nStart      = n[j]
        TanisoStart = Taniso[j]

        TparaEnd  = Tpara[j+interval]
        TperpEnd  = Tperp[j+interval]
        nEnd      = n[j+interval]
        TanisoEnd = Taniso[j+interval]

        TparaMean  = (TparaEnd + TparaStart)/2.0
        TperpMean  = (TperpEnd + TperpStart)/2.0
        nMean      = (nEnd     + nStart)/2.0

        DmumuMax = - (np.sqrt(TparaMean)*TperpMean)/(3.0*dt) * ( np.sqrt(TanisoEnd)/np.sqrt(1 + 0.5**2 * (TanisoEnd - 1.0)) - np.sqrt(TanisoStart)/np.sqrt(1 + 0.5**2 * (TanisoStart - 1.0)) )/( (1.0/TparaMean - 1.0/TperpMean) * (0.5**2/TparaMean + (1.0-0.5**2)/TperpMean)**(-5.0/2.0) )
        #DmumuMax = max(DmumuEq29)
        if DmumuMax < 0.0:
            DmumuMax = 0.0

        nu0Analytic[j] = 2*DmumuMax / ( (0.5/(1+0.5) + epsilonFit) * (1-0.5**2) )

        nu0Mean[j] = (nu0Analytic[j] + nu0Fit[j]) / 2

    except Exception as e:
        print(e)
        continue

np.save(path_dmumu + "DmumuL2R_"+str(interval)+".npy",DmumuL2R)
print  (path_dmumu + "DmumuL2R_"+str(interval)+".npy")

np.save(path_dmumu + "DmumuR2L_"+str(interval)+".npy",DmumuR2L)
print  (path_dmumu + "DmumuR2L_"+str(interval)+".npy")

np.save(path_dmumu + "DmumuSplit_"+str(interval)+".npy",DmumuSplit)
print  (path_dmumu + "DmumuSplit_"+str(interval)+".npy")

np.save(path_dmumu+'nu0Analytic_'+run+'_'+folder+'_'+str(interval)+'.npy',nu0Analytic)
print  (path_dmumu+'nu0Analytic_'+run+'_'+folder+'_'+str(interval)+'.npy')

np.save(path_dmumu+'nu0Fit_'+run+'_'+folder+'_'+str(interval)+'.npy',nu0Fit)
print  (path_dmumu+'nu0Fit_'+run+'_'+folder+'_'+str(interval)+'.npy')

np.save(path_dmumu+'nu0Mean_'+run+'_'+folder+'_'+str(interval)+'.npy',nu0Mean)
print  (path_dmumu+'nu0Mean_'+run+'_'+folder+'_'+str(interval)+'.npy')
