import analysator as pt
import numpy as np
import sys,os

run    = sys.argv[1]
folder = sys.argv[2]
bulkStart = int(sys.argv[3])
bulkEnd   = int(sys.argv[4])
interval  = int(sys.argv[5])

timetot = range(bulkStart, bulkEnd-interval+1,1)

path_bulk = "/scratch/project_2000203/dubartma/"+run+'/'+folder+'/bulk/'
path_save = '/scratch/project_2000203/dubartma/data/'+run+'/'+folder+'/variables/'

try:
    os.mkdir('/scratch/project_2000203/dubartma/data/'+run+'/'+folder)
except:
    pass
try:
    os.mkdir(path_save)
except:
    pass

betaPara = np.empty(len(timetot))
Taniso   = np.empty(len(timetot))
Tpara    = np.empty(len(timetot))
Tperp    = np.empty(len(timetot))
n        = np.empty(len(timetot))

for j in timetot:

    try:

        bulks = np.arange(j, j + interval + 1, 1)
        print(bulks)

        BetaParaTmp = np.empty(len(bulks))
        TanisoTmp   = np.empty(len(bulks))
        TparaTmp    = np.empty(len(bulks))
        TperpTmp    = np.empty(len(bulks))
        nTmp        = np.empty(len(bulks))

        for i in range(0,len(bulks)):
            bulkname = 'bulk.'+str(bulks[i]).rjust(7,'0')+'.vlsv'
            f = pt.vlsvfile.VlsvReader(path_bulk+bulkname)

            BetaParaTmp[i] = np.mean(f.read_variable('vg_beta_parallel'))
            TanisoTmp[i]   = np.mean(f.read_variable('vg_t_anisotropy'))
            TparaTmp[i]    = np.mean(f.read_variable('vg_t_parallel'))
            TperpTmp[i]    = np.mean(f.read_variable('vg_t_perpendicular'))
            nTmp[i]        = np.mean(f.read_variable('vg_rho'))

        betaPara[j] = np.mean(BetaParaTmp)
        Taniso[j]   = np.mean(TanisoTmp)
        Tpara[j]    = np.mean(TparaTmp)
        Tperp[j]    = np.mean(TperpTmp)
        n[j]        = np.mean(nTmp)

    except Exception as e:
        print(e)
        continue

np.save(path_save+'BetaPara_'+run+'_'+folder+'_'+str(interval)+'.npy',betaPara)
np.save(path_save+'Taniso_'  +run+'_'+folder+'_'+str(interval)+'.npy',Taniso)
np.save(path_save+'Tpara_'   +run+'_'+folder+'_'+str(interval)+'.npy',Tpara)
np.save(path_save+'Tperp_'   +run+'_'+folder+'_'+str(interval)+'.npy',Tperp)
np.save(path_save+'n_'       +run+'_'+folder+'_'+str(interval)+'.npy',n)

print(path_save+'BetaPara_'+run+'_'+folder+'_'+str(interval)+'.npy')
print(path_save+'Taniso_'  +run+'_'+folder+'_'+str(interval)+'.npy')
print(path_save+'Tpara_'   +run+'_'+folder+'_'+str(interval)+'.npy')
print(path_save+'Tperp_'   +run+'_'+folder+'_'+str(interval)+'.npy')
print(path_save+'n_'       +run+'_'+folder+'_'+str(interval)+'.npy')
