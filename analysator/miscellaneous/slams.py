# 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2019 University of Helsinki
# 
# For details of usage, see the COPYING file and read the "Rules of the Road"
# at http://www.physics.helsinki.fi/vlasiator/
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# 

import numpy as np
import analysator as pt
import os
import copy
import matplotlib.pyplot as plt

m_p  = 1.672621898e-27
r_e  = 6.371e+6
k    = 1.3806488e-23
mu_0 = 1.2566370614359173e-06

'''
This file should contain all functions and scripts related to finding, sorting, tracking and statistically analysing SLAMS.
'''

def bow_shock_r(runid,t):

    r0_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[11.7851239669,10.3130434783,11.9669421488,9.9652173913,12.4938271605]))
    v_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[0.0089345544,0.0044131524,0.0089722231,0.0054675004,0.0053351551]))

    return r0_dict[runid]+v_dict[runid]*(t-290)

def get_cell_area(vlsvobj):
    # returns area of one cell

    # get spatial extent of simulation and the number of cells in each direction
    simextent = vlsvobj.get_spatial_mesh_extent().reshape((2,3))
    simsize = vlsvobj.get_spatial_mesh_size()

    # calculate DX,DY,DZ
    cell_sizes = (simextent[1]-simextent[0])/simsize
    
    # calculate area of one cell
    dA = cell_sizes[0]*cell_sizes[1]

    return dA

def get_neighbors(vlsvobj,c_i,neighborhood_reach=[1,1]):
    # finds the neighbors of the specified cells within the maximum offsets in neighborhood_reach

    simsize = vlsvobj.get_spatial_mesh_size()

    # initialise array of neighbors
    neighbors = np.array([],dtype=int)

    # range of offsets to take into account
    x_r = range(-1*neighborhood_reach[0],neighborhood_reach[0]+1)
    y_r = range(-1*neighborhood_reach[1],neighborhood_reach[1]+1)

    
    if simsize[1] == 1:
        for n in c_i:

            # append cellids of neighbors, cast as int, to array of neighbors
            for a in x_r:
                for b in y_r:
                    neighbors = np.append(neighbors,int(vlsvobj.get_cell_neighbor(cellid=n,offset=[a,0,b],periodic=[0,0,0])))
    else:
        for n in c_i:

            # append cellids of neighbors, cast as int, to array of neighbors
            for a in x_r:
                for b in y_r:
                    neighbors = np.append(neighbors,int(vlsvobj.get_cell_neighbor(cellid=n,offset=[a,b,0],periodic=[0,0,0])))

    # discard invalid cellids
    neighbors = neighbors[neighbors != 0]

    # discard duplicate cellids
    neighbors = np.unique(neighbors)

    return neighbors

def var_pars_list(var):

    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    label_list = ["$Duration~[s]$",
    "$Radial~size~[R_{e}]$","$Tangential~size~[R_{e}]$","$Radial~size/Tangential~size$",
    "$P_{dyn,vmax}~[P_{dyn,sw}]$","$P_{dyn,avg}~[P_{dyn,sw}]$","$P_{dyn,med}~[P_{dyn,sw}]$","$P_{dyn,max}~[P_{dyn,sw}]$",
    "$n_{max}~[n_{sw}]$","$n_{avg}~[n_{sw}]$","$n_{med}~[n_{sw}]$","$n_{v,max}~[n_{sw}]$",
    "$v_{max}~[v_{sw}]$","$v_{avg}~[v_{sw}]$","$v_{med}~[v_{sw}]$",
    "$B_{max}~[B_{IMF}]$","$B_{avg}~[B_{IMF}]$","$B_{med}~[B_{IMF}]$",
    "$\\beta _{max}~[\\beta _{sw}]$","$\\beta _{avg}~[\\beta _{sw}]$","$\\beta _{med}~[\\beta _{sw}]$","$\\beta _{v,max}~[\\beta _{sw}]$",
    "$T_{avg}~[MK]$","$T_{med}~[MK]$","$T_{max}~[MK]$",
    "$T_{Parallel,avg}~[MK]$","$T_{Parallel,med}~[MK]$","$T_{Parallel,max}~[MK]$",
    "$T_{Perpendicular,avg}~[MK]$","$T_{Perpendicular,med}~[MK]$","$T_{Perpendicular,max}~[MK]$",
    "$Area~[R_{e}^{2}]$",
    "$(r_{v,max}-r_{BS})~at~time~of~death~[R_{e}]$"]

    xmin_list=[10,
    0,0,0,
    1.25,1.25,1.25,1.25,
    1,1,1,1,
    0.6,0.6,0.6,
    1.25,1.25,1.25,
    1,1,1,1,
    0,0,0,
    0,0,0,
    0,0,0,
    0,
    8]

    xmax_list=[60,
    3,1,7,
    3,3,3,3,
    3,3,3,3,
    1.2,1.2,1.2,
    6,6,6,
    1000,1000,1000,1000,
    25,25,25,
    25,25,25,
    25,25,25,
    1.5,
    18]

    step_list = [2,
    0.25,0.05,0.2,
    0.05,0.05,0.05,0.05,
    0.1,0.1,0.1,0.1,
    0.05,0.05,0.05,
    0.25,0.25,0.25,
    100,100,100,100,
    1,1,1,
    1,1,1,
    1,1,1,
    0.05,
    0.5]

    tickstep_list = [20,
    0.5,0.5,1,
    1,1,1,1,
    2,2,2,2,
    0.2,0.2,0.2,
    1,1,1,
    100,100,100,100,
    5,5,5,
    5,5,5,
    5,5,5,
    1,
    2]

    return [label_list[key_list.index(var)],xmin_list[key_list.index(var)],xmax_list[key_list.index(var)],step_list[key_list.index(var)],tickstep_list[key_list.index(var)]]

def bow_shock_finder(vlsvobj,rho_sw=1.0e+6,v_sw=750e+3):
    # returns cells outside the bow shock

    # If file has separate populations, find proton population
    if vlsvobj.check_population("proton"):
        try:
            rho = vlsvobj.read_variable("proton/rho")
        except:
            rho = vlsvobj.read_variable("rho")
    else:
        rho = vlsvobj.read_variable("rho")
        
    cellids = vlsvobj.read_variable("CellID")

    simdim = vlsvobj.get_spatial_mesh_size()

    # Create mask
    bs = np.ma.masked_less(rho,1.85*rho_sw)

    # Find IDs of masked cells
    masked_ci = np.ma.array(cellids,mask=~bs.mask).compressed()

    return masked_ci

def xyz_reconstruct_old(vlsvobj):
    # reconstructs coordinates based on spatial mesh parameters

    # read UNSORTED cell ids
    ci = vlsvobj.read_variable("CellID")

    # get simulation extents and dimension sizes
    simextent = vlsvobj.get_spatial_mesh_extent()
    simsize = vlsvobj.get_spatial_mesh_size()

    # discard 3rd dimension
    simdim = simsize[simsize!=1]

    # reconstruct SORTED X
    X = np.linspace(simextent[0],simextent[3],simdim[0]+1)[:-1]
    X = np.pad(X,(0,simdim[0]*(simdim[1]-1)),"wrap")

    # reconstruct SORTED Y
    Y = np.linspace(simextent[1],simextent[4],simdim[1]+1)[:-1]
    Y = np.pad(Y,(0,simdim[1]*(simdim[0]-1)),"wrap")
    Y = np.reshape(Y,(simdim[0],simdim[1]))
    Y = Y.T
    Y = Y.flatten()

    # reconstruct SORTED Z
    Z = np.linspace(simextent[2],simextent[5],simdim[1]+1)[:-1]
    Z = np.pad(Z,(0,simdim[1]*(simdim[0]-1)),"wrap")
    Z = np.reshape(Z,(simdim[0],simdim[1]))
    Z = Z.T
    Z = Z.flatten()

    # UNSORT XYZ
    X = X[ci-1]
    Y = Y[ci-1]
    Z = Z[ci-1]

    return np.array([X,Y,Z])

def xyz_reconstruct(vlsvobj,cellids=-1):

    if type(cellids) == int:
        if cellids == -1:
            ci = vlsvobj.read_variable("CellID")
        else:
            ci = np.asarray([cellids])
    else:
        ci = np.asarray(cellids)

    coords = np.array([vlsvobj.get_cell_coordinates(cell) for cell in ci])

    coords = coords.T

    return coords

def restrict_area(vlsvobj,boxre):
    # find cellids of cells that correspond to X,Y-positions within the specified limits

    cellids = vlsvobj.read_variable("CellID")

    # If X doesn't exist, reconstruct X,Y,Z, otherwise read X,Y,Z
    if vlsvobj.check_variable("X"):
        X,Y,Z = vlsvobj.read_variable("X"),vlsvobj.read_variable("Y"),vlsvobj.read_variable("Z")
    else:
        X,Y,Z = xyz_reconstruct(vlsvobj)

    # Get the simulation size
    simsize = vlsvobj.get_spatial_mesh_size()
    
    # if polar run, replace Y with Z
    if simsize[1] == 1:
        Y = Z

    # mask the cellids within the specified limits
    msk = np.ma.masked_greater_equal(X,boxre[0]*r_e)
    msk.mask[X > boxre[1]*r_e] = False
    msk.mask[Y < boxre[2]*r_e] = False
    msk.mask[Y > boxre[3]*r_e] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(cellids,mask=~msk.mask).compressed()

    return masked_ci

def sw_par_dict(runid):
    # List of runs incomplete, please add parameters for additional runs manually as needed

    # Returns solar wind parameters for specified run
    # Output is 0: density, 1: velocity, 2: IMF strength 3: dynamic pressure 4: plasma beta

    runs = ["ABA","ABC","AEA","AEC","BFD"]
    sw_rho = [1e+6,3.3e+6,1.0e+6,3.3e+6,1.0e+6]
    sw_v = [750e+3,600e+3,750e+3,600e+3,750e+3]
    sw_B = [5.0e-9,5.0e-9,10.0e-9,10.0e-9,5.0e-9]
    sw_T = [500e+3,500e+3,500e+3,500e+3,500e+3]
    sw_pdyn = [m_p*sw_rho[n]*(sw_v[n]**2) for n in range(len(runs))]
    sw_beta = [2*mu_0*sw_rho[n]*k*sw_T[n]/(sw_B[n]**2) for n in range(len(runs))]

    return [sw_rho[runs.index(runid)],sw_v[runs.index(runid)],sw_B[runs.index(runid)],sw_pdyn[runs.index(runid)],sw_beta[runs.index(runid)]]

class Transient:
    # Class for identifying and handling individual jets and their properties

    def __init__(self,ID,runid,birthday):

        self.ID = ID
        self.runid = runid
        self.birthday = birthday
        self.cellids = []
        self.times = [birthday]

        print("Created jet with ID "+self.ID)

    def return_cellid_string(self):

        return "\n".join([",".join(map(str,l)) for l in self.cellids])

    def return_time_string(self):

        return "\n".join(map(str,self.times))

def visual_slams_finder(runid,filenumber,boxre=[6,18,-8,6],vmax=1.5,plaschke=1.0,sw=1.0,outputdir="SLAMS/contours/"):

    if runid in ["AEA","AEC"]:
        B_sw = 10.0e-9
    else:
        B_sw = 5.0e-9

    B_sw = sw_par_dict(runid)[2]

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if runid == "BFD":
        pass_vars=["proton/rho","proton/V","CellID","B"]
    else:
        pass_vars=["rho","v","CellID","B"]

    global g_plaschke,g_sw,g_runid
    g_plaschke = plaschke
    g_sw = sw
    g_runid = runid

    pt.plot.plot_colormap(filename=bulkpath+bulkname,draw=1,outputdir=outputdir,run=runid,step=filenumber,usesci=0,lin=1,cbtitle="",boxre=boxre,colormap="parula",vmin=0,vmax=vmax,var="Pdyn",external=ext_slams,pass_vars=pass_vars)

def ext_slams(ax1, XmeshCentres,YmeshCentres, pass_maps):

    # Using list comprehension here due to the varying nature of the pass_vars variable names
    B,cellids,rho,v = pass_maps.values()

    vx = v[:,:,0]
    vmag = np.linalg.norm(v,axis=-1)

    Bmag = np.linalg.norm(B,axis=-1)

    shp = rho.shape

    sw_rho,sw_v,sw_B,sw_pdyn,sw_beta = sw_par_dict(g_runid)

    pdyn = m_p*rho*(vmag**2)
    pdyn_x = m_p*rho*(vx**2)

    slams = np.ma.masked_greater(Bmag,g_sw*sw_B)
    slams.mask[pdyn < g_plaschke*sw_pdyn] = False
    slams.mask[rho > 3*sw_rho] = False
    slams.fill_value = 0
    slams[slams.mask == False] = 1

    contour = ax1.contour(XmeshCentres,YmeshCentres,slams.filled(),[0.5],linewidths=1.0, colors="black")

    return None

def make_slams_mask(filenumber,runid,boxre=[6,18,-8,6]):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # if file has separate populations, read proton population
    if vlsvreader.check_population("proton"):
        try:
            rho = vlsvreader.read_variable("proton/rho")[np.argsort(origid)]
            v = vlsvreader.read_variable("proton/V")[np.argsort(origid)]
        except:
            rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
            v = vlsvreader.read_variable("v")[np.argsort(origid)]
    else:
        rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("v")[np.argsort(origid)]

    B = vlsvreader.read_variable("B")[np.argsort(origid)]
    Bmag = np.linalg.norm(B,axis=-1)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]
    pdyn_sw = m_p*rho_sw*(v_sw**2)
    if runid in ["AEA","AEC"]:
        B_sw = 10.0e-9
    else:
        B_sw = 5.0e-9

    # make custom SLAMS mask
    slams = np.ma.masked_greater(Bmag,1.25*B_sw)
    slams.mask[pdyn < 1.25*pdyn_sw] = False
    slams.mask[rho > 3*rho_sw] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~slams.mask).compressed()

    if not os.path.exists("SLAMS/masks/"+runid+"/"):
        try:
            os.makedirs("SLAMS/masks/"+runid+"/")
        except OSError:
            pass

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre))
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci
    else:
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci

def sort_slams(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1]):
    # sort masked cells into events based on proximity in X,Y-space

    # initialise list of events and current event
    events = []
    curr_event = np.array([],dtype=int)

    for cell in cells:

        # check if cell already in list of events
        bl_a = False
        for event in events:
            if cell in event:
                bl_a = True
        if bl_a:
            continue

        # number of times to search for more neighbors, larger is better but possibly slower.
        it_range = range(200)

        # initialise current event
        curr_event = np.array([cell])

        for n in it_range:

            curr_event_size = curr_event.size

            # find neighbors within the confines of the mask
            curr_event = np.unique(np.append(curr_event,np.intersect1d(cells,get_neighbors(vlsvobj,curr_event,neighborhood_reach))))

            # exit loop if all valid neighbors found
            if curr_event_size == curr_event.size:
                break

        # cast cellids of current event to int and append to list of events
        curr_event = curr_event.astype(int)
        events.append(curr_event)

    # remove events smaller than the minimum size and larger than maximum size
    events_culled = [slams for slams in events if slams.size >= min_size and slams.size <= max_size]

    return events_culled

def slams_maker(runid,start,stop,boxre=[6,18,-8,6],maskfile=False):

    outputdir = "SLAMS/events/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    for file_nr in range(start,stop+1):

        # find correct file based on file number and run id
        if runid in ["AEC","AEF","BEA","BEB"]:
            bulkpath = "/proj/vlasov/2D/"+runid+"/"
        elif runid == "AEA":
            bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
        else:
            bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

        if runid == "AED":
            bulkname = "bulk.old."+str(file_nr).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(file_nr)+" not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # create mask
        if maskfile:
            msk = np.loadtxt("SLAMS/masks/"+runid+"/"+str(file_nr)+".mask").astype(int)
        else:
            msk = make_slams_mask(file_nr,runid,boxre)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        slams = sort_slams(vlsvobj,msk,10,4500,[2,2])

        # erase contents of output file
        open(outputdir+str(file_nr)+".events","w").close()

        # open output file
        fileobj = open(outputdir+str(file_nr)+".events","a")

        # write jets to outputfile
        for slam in slams:

            fileobj.write(",".join(map(str,slam))+"\n")

        fileobj.close()

    return None

def eventfile_read(runid,filenr):
    # Read array of arrays of cellids from file

    outputlist = []

    ef = open("SLAMS/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read().strip("\n")
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(list(map(int,line.split(","))))

    return outputlist

def track_slams(runid,start,stop,threshold=0.5):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(start).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(start)+" not found, exiting")
        return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists("SLAMS/slams/"+runid):
        try:
            os.makedirs("SLAMS/slams/"+runid)
        except OSError:
            pass

    # Get solar wind parameters
    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]

    # Open file, get Cell IDs and sort them
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
    sorigid = vlsvobj.read_variable("CellID")
    sorigid = sorigid[sorigid.argsort()]
    
    # Find bow shock cells and area of one cell
    bs_cells = bow_shock_finder(vlsvobj,rho_sw,v_sw)
    dA = get_cell_area(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start)
    events = eventfile_read(runid,start+1)

    # do nothing
    bs_events = []
    for old_event in events_old:
        bs_events.append(old_event)

    # Initialise list of jet objects
    jetobj_list = []
    dead_jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    # Print current time
    print("t = "+str(float(start+1)/2)+"s")

    # Look for slams
    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > threshold*len(event):

                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object
                jetobj_list.append(Transient(curr_id,runid,float(start)/2))

                # Append current events to jet object properties
                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in range(start+2,stop+1):

        for jetobj in jetobj_list:
            if float(n)/2 - jetobj.times[-1] > 5:
                dead_jetobj_list.append(jetobj)
                jetobj_list.remove(jetobj)

        # Print  current time
        print("t = "+str(float(n)/2)+"s")

        # Find correct bulkname
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            events = []
            continue

        # Open bulkfile and get bow shock cells
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        bs_cells = bow_shock_finder(vlsvobj,rho_sw,v_sw)

        # List of old events
        bs_events = []
        for old_event in events:
            bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events = eventfile_read(runid,n)
        events.sort(key=len)
        events = events[::-1]

        # Iniatilise list of cells currently being tracked
        curr_jet_temp_list = []

        # Update existing jets
        for event in events:

            for jetobj in jetobj_list:

                if jetobj.ID in flags:
                    
                    if np.intersect1d(jetobj.cellids[-2],event).size > threshold*len(event):

                        curr_id = str(counter).zfill(5)

                        # Create new jet
                        jetobj_new = Transient(curr_id,runid,float(n)/2)
                        jetobj_new.cellids.append(event)
                        jetobj_list.append(jetobj_new)
                        curr_jet_temp_list.append(event)

                        # Iterate counter
                        counter += 1

                        break

                else:

                    if np.intersect1d(jetobj.cellids[-1],event).size > threshold*len(event):

                        # Append event to jet object properties
                        jetobj.cellids.append(event)
                        jetobj.times.append(float(n)/2)
                        print("Updated jet with ID "+jetobj.ID)

                        # Flag jet object
                        flags.append(jetobj.ID)
                        curr_jet_temp_list.append(event)

                        break

        # Look for new jets at bow shock
        for event in events:

            if event not in curr_jet_temp_list:

                for bs_event in bs_events:

                    if np.intersect1d(bs_event,event).size > threshold*len(event):

                        # Create unique ID
                        curr_id = str(counter).zfill(5)

                        # Create new jet object
                        jetobj_list.append(Transient(curr_id,runid,float(n-1)/2))

                        # Append current events to jet object properties
                        jetobj_list[-1].cellids.append(bs_event)
                        jetobj_list[-1].cellids.append(event)
                        jetobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        counter += 1

                        break

    jetobj_list = jetobj_list + dead_jetobj_list

    for jetobj in jetobj_list:

        # Write jet object cellids and times to files
        jetfile = open("SLAMS/slams/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".slams","w")
        timefile = open("SLAMS/slams/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())

        jetfile.close()
        timefile.close()

    return None

def timefile_read(runid,filenr,key):
    # Read array of times from file

    tf = open("SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".times","r")
    contents = tf.read().split("\n")
    tf.close()

    return list(map(float,contents))

def jetfile_read(runid,filenr,key):
    # Read array of cellids from file

    outputlist = []

    jf = open("SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".slams","r")
    contents = jf.read()
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(list(map(int,line.split(","))))

    return outputlist

def propfile_write(runid,filenr,key,props):
    # Write jet properties to file

    open("SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props","w").close()
    pf = open("SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props","a")
    pf.write("time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax,pd_avg [nPa],pd_med [nPa],pd_max [nPa]"+"\n")
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
    pf.close()
    print("Wrote to SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props")

def plotmake_script_BFD(start,stop,runid="BFD",vmax=1.5,boxre=[4,20,-10,4]):

    if not os.path.exists("SLAMS/contours/"+runid):
        try:
            os.makedirs("SLAMS/contours/"+runid)
        except OSError:
            pass

    # Find names of property files
    filenames = os.listdir("SLAMS/slams/"+runid)
    prop_fns = []
    for filename in filenames:
        if ".props" in filename:
            prop_fns.append(filename)
    prop_fns.sort()

    xmean_dict = dict()
    ymean_dict = dict()
    xmax_dict = dict()
    ymax_dict = dict()

    for fname in prop_fns:
        jet_id = fname[4:-6]
        props = PropReader(ID=jet_id,runid=runid)
        time = props.read("time")
        x_mean = props.read("x_mean")
        y_mean = props.read("y_mean")
        z_mean = props.read("z_mean")
        x_vmax = props.read("x_vmax")
        y_vmax = props.read("y_vmax")
        z_vmax = props.read("z_vmax")
        if runid in ["BFD"]:
            y_mean = z_mean
            y_vmax = z_vmax
        for itr in range(time.size):
            if time[itr] not in xmean_dict:
                xmean_dict[time[itr]] = [x_mean[itr]]
                ymean_dict[time[itr]] = [y_mean[itr]]
                xmax_dict[time[itr]] = [x_vmax[itr]]
                ymax_dict[time[itr]] = [y_vmax[itr]]
            else:
                xmean_dict[time[itr]].append(x_mean[itr])
                ymean_dict[time[itr]].append(y_mean[itr])
                xmax_dict[time[itr]].append(x_vmax[itr])
                ymax_dict[time[itr]].append(y_vmax[itr])

    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for itr2 in range(start,stop+1):

        t = float(itr2)/2

        bulkname = "bulk."+str(itr2).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(itr2)+" not found, continuing")
            continue

        if runid == "BFD" and itr2 == 961:
            print("Broken file!")
            continue

        if runid in ["BFD"]:
            pass_vars = ["proton/rho","proton/V","CellID"]
        else:
            pass_vars = ["rho","v","CellID"]

        try:
            fullmask = np.loadtxt("SLAMS/masks/"+runid+"/"+str(itr2)+".mask").astype(int)
        except IOError:
            fullmask = np.array([])

        try:
            fileobj = open("SLAMS/events/"+runid+"/"+str(itr2)+".events","r")
            contents = fileobj.read()
            cells = list(map(int,contents.replace("\n",",").split(",")[:-1]))
        except IOError:
            cells = []

        # Create plot
        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir="SLAMS/contours/"+runid+"/",step=itr2,run=runid,usesci=0,lin=1,boxre=boxre,vmin=0,vmax=vmax,colormap="parula",cbtitle="",external=pms_ext,var="Pdyn",pass_vars=pass_vars,ext_pars=[xmean_dict[t],ymean_dict[t],cells,fullmask,xmax_dict[t],ymax_dict[t]])

    return None

def pms_ext(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    rho,v,cellids = extmaps

    x_list,y_list,cells,fullmask,xmax_list,ymax_list = ext_pars

    # Create mask
    msk = np.in1d(cellids,cells).astype(int)
    msk = np.reshape(msk,rho.shape)

    fullmsk = np.in1d(cellids,fullmask).astype(int)
    fullmsk = np.reshape(fullmsk,rho.shape)

    # Draw contours
    fullcont = ax.contour(XmeshXY,YmeshXY,fullmsk,[0.5],linewidths=1.0,colors="magenta")
    cont = ax.contour(XmeshXY,YmeshXY,msk,[0.5],linewidths=1.0,colors="black")

    # Plot jet positions
    ax.plot(x_list,y_list,"o",color="red",markersize=4)
    ax.plot(xmax_list,ymax_list,"o",color="white",markersize=4)

def calc_slams_properties(runid,start,jetid,tp_files=False):
    # Calculates SLAMS properties and writes them to a props file

    if str(start)+"."+jetid+".slams" not in os.listdir("SLAMS/slams/"+runid):
        print("SLAMS with ID "+jetid+" does not exist, exiting.")
        return 1

    # Read jet cellids and times
    jet_list = jetfile_read(runid,start,jetid)
    time_list = timefile_read(runid,start,jetid)

    # Discard jet if it's very short-lived
    if len(time_list) < 5:
        print("Jet not sufficiently long-lived, exiting.")
        return 1

    # Discard jet if it has large gaps in the times
    dt = (np.pad(np.array(time_list),(0,1),"constant")-np.pad(np.array(time_list),(1,0),"constant"))[1:-1]
    dt = np.ediff1d(time_list)
    if max(dt) > 5:
        print("Jet not sufficiently continuous, exiting.")
        return 1

    # Find correct bulk path
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    # Convert times to file numbers
    nr_list = [int(t*2) for t in time_list]

    # Initialise property array
    prop_arr = np.array([])

    for n in range(len(nr_list)):

        curr_list = jet_list[n]
        curr_list.sort()

        # Find correct file name
        if runid == "AED":
            bulkname = "bulk.old."+str(nr_list[n]).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(nr_list[n]).zfill(7)+".vlsv"

        # Open VLSV file
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        origid = vlsvobj.read_variable("CellID")
        sorigid = origid[origid.argsort()]

        # read variables
        if vlsvobj.check_variable("X"):
            X = vlsvobj.read_variable("X",cellids=curr_list)
            Y = vlsvobj.read_variable("Y",cellids=curr_list)
            Z = vlsvobj.read_variable("Z",cellids=curr_list)
        else:
            X,Y,Z = xyz_reconstruct(vlsvobj,cellids=curr_list)

        # Calculate area of one cell
        if n == 0 and vlsvobj.check_variable("DX"):
            dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
        elif n == 0 and not vlsvobj.check_variable("DX"):
            dA = get_cell_area(vlsvobj)

        # If file has more than one population, choose proton population
        var_list = ["rho","v","B","Temperature","CellID","beta","TParallel","TPerpendicular"]
        var_list_alt = ["proton/rho","proton/V","B","proton/Temperature","CellID","proton/beta","proton/TParallel","proton/TPerpendicular"]
        if vlsvobj.check_population("proton"):
            try:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [vlsvobj.read_variable(s,cellids=curr_list) for s in var_list_alt]
            except:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [vlsvobj.read_variable(s,cellids=curr_list) for s in var_list]
        else:
            rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [vlsvobj.read_variable(s,cellids=curr_list) for s in var_list]

        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        # Scale variables
        rho /= 1.0e+6
        v /= 1.0e+3
        B /= 1.0e-9
        pdyn /= 1.0e-9
        T /= 1.0e+6
        TParallel /= 1.0e+6
        TPerpendicular /= 1.0e+6

        # Calculate magnitudes of v and B
        vmag = np.linalg.norm(v,axis=-1)
        Bmag = np.linalg.norm(B,axis=-1)

        # Calculate means, medians and maximums for rho,vmag,Bmag,T,TParallel,TPerpendicular,beta
        n_avg = np.nanmean(rho)
        n_med = np.median(rho)
        n_max = np.max(rho)

        v_avg = np.nanmean(vmag)
        v_med = np.median(vmag)
        v_max = np.max(vmag)

        B_avg = np.nanmean(Bmag)
        B_med = np.median(Bmag)
        B_max = np.max(Bmag)

        pd_avg = np.nanmean(pdyn)
        pd_med = np.median(pdyn)
        pd_max = np.max(pdyn)

        T_avg = np.nanmean(T)
        T_med = np.median(T)
        T_max = np.max(T)

        TPar_avg = np.nanmean(TParallel)
        TPar_med = np.median(TParallel)
        TPar_max = np.max(TParallel)

        TPerp_avg = np.nanmean(TPerpendicular)
        TPerp_med = np.median(TPerpendicular)
        TPerp_max = np.max(TPerpendicular)

        beta_avg = np.nanmean(beta)
        beta_med = np.median(beta)
        beta_max = np.max(beta)

        # Convert X,Y,Z to spherical coordinates
        r = np.linalg.norm(np.array([X,Y,Z]),axis=0)
        theta = np.rad2deg(np.arccos(Z/r))
        phi = np.rad2deg(np.arctan(Y/X))

        # calculate geometric center of jet
        r_mean = np.mean(r)/r_e
        theta_mean = np.mean(theta)
        phi_mean = np.mean(phi)

        # Geometric center of jet in cartesian coordinates
        x_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.cos(np.deg2rad(phi_mean))
        y_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.sin(np.deg2rad(phi_mean))
        z_mean = r_mean*np.cos(np.deg2rad(theta_mean))

        # Position of maximum velocity in cartesian coordinates
        x_max = X[vmag==max(vmag)][0]/r_e
        y_max = Y[vmag==max(vmag)][0]/r_e
        z_max = Z[vmag==max(vmag)][0]/r_e

        # Minimum x and density at maximum velocity
        x_min = min(X)/r_e
        rho_vmax = rho[vmag==max(vmag)]
        b_vmax = beta[vmag==max(vmag)]

        #r_max = np.linalg.norm(np.array([x_mean,y_mean,z_mean]))
        #theta_max = np.rad2deg(np.arccos(z_mean/r_mean))
        #phi_max = np.rad2deg(np.arctan(y_mean/x_mean))

        # calculate jet size
        A = dA*len(curr_list)/(r_e**2)
        Nr_cells = len(curr_list)

        # calculate linear sizes of jet
        size_rad = (max(r)-min(r))/r_e
        size_tan = A/size_rad

        # current time
        time = time_list[n]

        ''' 
        0: time [s],
        1: x_mean [R_e],        2: y_mean [R_e],        3: z_mean [R_e],
        4: A [R_e^2],           5: Nr_cells,
        6: r_mean [R_e],        7: theta_mean [deg],    8: phi_mean [deg],
        9: size_rad [R_e],      10: size_tan [R_e],
        11: x_max [R_e],        12: y_max [R_e],        13: z_max [R_e],
        14: n_avg [1/cm^3],     15: n_med [1/cm^3],     16: n_max [1/cm^3],
        17: v_avg [km/s],       18: v_med [km/s],       19: v_max [km/s],
        20: B_avg [nT],         21: B_med [nT],         22: B_max [nT],
        23: T_avg [MK],         24: T_med [MK],         25: T_max [MK],
        26: TPar_avg [MK],      27: TPar_med [MK],      28: TPar_max [MK],
        29: TPerp_avg [MK],     30: TPerp_med [MK],     31: TPerp_max [MK],
        32: beta_avg,           33: beta_med,           34: beta_max
        35: x_min [R_e],        36: rho_vmax [1/cm^3],  37: b_vmax
        38: pd_avg [nPa],       39: pd_med [nPa],       40: pd_max [nPa]
        '''

        # Create temporary property array
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax,pd_avg,pd_med,pd_max]

        # append properties to property array
        prop_arr = np.append(prop_arr,np.array(temp_arr))

    # reshape property array
    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    # write property array to file
    propfile_write(runid,start,jetid,prop_arr)

    return prop_arr

def slams_2d_hist(runids,var1,var2,time_thresh=10):
    # Create 2D histogram of var1 and var2

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("SLAMS/slams/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff dictionary for eliminating false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(range(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise input variable list and variable list
    inp_var_list = [var1,var2]
    var_list = [[],[]]

    # Append variable values to var lists
    for ind in range(len(inp_var_list)):
        for n in range(len(runids)):
            for fname in file_list_list[n]:
                props = PropReader("",runids[n],fname=fname)
                if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    if inp_var_list[ind] == "duration":
                        var_list[ind].append(props.read("time")[-1]-props.read("time")[0])
                    elif inp_var_list[ind] == "size_ratio":
                        var_list[ind].append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                    elif inp_var_list[ind] in ["n_max","n_avg","n_med","rho_vmax"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[0])
                    elif inp_var_list[ind] in ["v_max","v_avg","v_med"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[1])
                    elif inp_var_list[ind] in ["B_max","B_avg","B_med"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[2])
                    elif inp_var_list[ind] in ["beta_max","beta_avg","beta_med","b_vmax"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[4])
                    elif inp_var_list[ind] in ["pdyn_vmax"]:
                        var_list[ind].append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                    elif inp_var_list[ind] in ["pd_avg","pd_med","pd_max"]:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind])/props.sw_pars[3])
                    elif inp_var_list[ind] == "death_distance":
                        var_list[ind].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]]))
                    else:
                        var_list[ind].append(props.read_at_amax(inp_var_list[ind]))

    v1_label,v1_xmin,v1_xmax,v1_step,v1_tickstep = var_pars_list(var1)
    v2_label,v2_xmin,v2_xmax,v2_step,v2_tickstep = var_pars_list(var2)

    # Create figure
    plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+v1_label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{"+v2_label[1:-1]+"}$",fontsize=24)
    ax.tick_params(labelsize=20)
    weights = [1/float(len(var_list[0]))]*len(var_list[0]) # Normalise by total number of jets
    bins = [np.linspace(xlims[0],xlims[1],21).tolist() for xlims in [[v1_xmin,v1_xmax],[v2_xmin,v2_xmax]]]

    hist = ax.hist2d(var_list[0],var_list[1],bins=bins,weights=weights)

    if v1_xmax == v2_xmax:
        ax.plot([0,v1_xmax],[0,v1_xmax],"r--")

    ax.set_xticks(np.arange(v1_xmin+v1_tickstep,v1_xmax+v1_tickstep,v1_tickstep))
    ax.set_yticks(np.arange(v2_xmin+v2_tickstep,v2_xmax+v2_tickstep,v2_tickstep))

    ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(v1_xmin+v1_tickstep,v1_xmax+v1_tickstep,v1_tickstep).astype(str)])
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(v2_xmin+v2_tickstep,v2_xmax+v2_tickstep,v2_tickstep).astype(str)])

    plt.title(",".join(runids),fontsize=24)
    plt.colorbar(hist[3], ax=ax)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("SLAMS/figures/histograms/"+"_".join(runids)+"/"):
        try:
            os.makedirs("SLAMS/figures/histograms/"+"_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("SLAMS/figures/histograms/"+"_".join(runids)+"/"+var1+"_"+var2+"_"+str(time_thresh)+"_2d.png")
    print("SLAMS/figures/histograms/"+"_".join(runids)+"/"+var1+"_"+var2+"_"+str(time_thresh)+"_2d.png")

    plt.close(fig)

    return None

def slams_vs_hist(runids,var,time_thresh=10):

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("SLAMS/slams/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff dictionary for eliminating false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Different colors for different runs
    run_colors_dict = dict(zip([runids[0],runids[1]],["red","blue"]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(range(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise var list
    var_list = [[],[]]

    val_dict = dict(zip(runids,var_list))

    # Append variable values to var lists
    for n in range(len(runids)):
        for fname in file_list_list[n]:
            props = PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                if var == "duration":
                    val_dict[runids[n]].append(props.read("time")[-1]-props.read("time")[0])
                elif var == "size_ratio":
                    val_dict[runids[n]].append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                elif var in ["n_max","n_avg","n_med","rho_vmax"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[0])
                elif var in ["v_max","v_avg","v_med"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[1])
                elif var in ["B_max","B_avg","B_med"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[2])
                elif var in ["beta_max","beta_avg","beta_med","b_vmax"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[4])
                elif var in ["pdyn_vmax"]:
                    val_dict[runids[n]].append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                elif var in ["pd_avg","pd_med","pd_max"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[3])
                elif var == "death_distance":
                    val_dict[runids[n]].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-bow_shock_r(runids[n],props.read("time")[-1]))
                else:
                    val_dict[runids[n]].append(props.read_at_amax(var))


    label,xmin,xmax,step,tickstep = var_pars_list(var)

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{Fraction~of~SLAMS}$",fontsize=24)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,0.75)
    ax.tick_params(labelsize=20)
    weights = [[1/float(len(val_dict[runids[n]]))]*len(val_dict[runids[n]]) for n in range(len(runids))] # Normalise by total number of jets

    ax.set_yticks(np.arange(0.1,0.8,0.1))
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(0.1,0.8,0.1).astype(str)])

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax)
        
        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],weights=weights,bins=bins,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=[runids[0]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[0]]),np.std(val_dict[runids[0]],ddof=1)),runids[1]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[1]]),np.std(val_dict[runids[1]],ddof=1))])

        ax.set_xticks(np.array([10**0,10**1,10**2,10**3]))
        ax.set_xticklabels(np.array(["$\\mathtt{10^0}$","$\\mathtt{10^1}$","$\\mathtt{10^2}$","$\\mathtt{10^3}$"]))

    else:
        bins = np.arange(xmin,xmax+step,step)

        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],bins=bins,weights=weights,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=[runids[0]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[0]]),np.std(val_dict[runids[0]],ddof=1)),runids[1]+"\nmed: %.1f\nstd: %.1f"%(np.median(val_dict[runids[1]]),np.std(val_dict[runids[1]],ddof=1))])

        ax.set_xticks(np.arange(xmin,xmax+tickstep,tickstep))
        ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin,xmax+tickstep,tickstep).astype(str)])

    if xmin == -xmax and 0.5*(xmax-xmin)%tickstep != 0.0:
        ax.set_xticks(np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep))
        if tickstep%1 != 0:
            ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep).astype(str)])
        else:
            ax.set_xticklabels(["$\\mathtt{"+str(int(lab))+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep)])

    plt.title(" vs. ".join(runids),fontsize=24)
    plt.legend(fontsize=20)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/"):
        try:
            os.makedirs("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")
    print("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")

    plt.close(fig)

    return None

def slams_all_hist(runids,var,time_thresh=10):
    # Creates histogram specified var

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("SLAMS/slams/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff values for elimination of false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[10,8,10,8,10]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(range(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise var list
    var_list = []

    # Append variable values to var list
    for n in range(len(runids)):
        for fname in file_list_list[n]:
            props = PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                if var == "duration":
                    var_list.append(props.read("time")[-1]-props.read("time")[0])
                elif var == "size_ratio":
                    var_list.append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                elif var in ["n_max","n_avg","n_med","rho_vmax"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[0])
                elif var in ["v_max","v_avg","v_med"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[1])
                elif var in ["B_max","B_avg","B_med"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[2])
                elif var in ["beta_max","beta_avg","beta_med","b_vmax"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[4])
                elif var in ["pdyn_vmax"]:
                    var_list.append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                elif var in ["pd_avg","pd_med","pd_max"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[3])
                elif var == "death_distance":
                    var_list.append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]])-bow_shock_r(runids[n],props.read("time")[-1]))
                else:
                    var_list.append(props.read_at_amax(var))

    var_list = np.asarray(var_list)

    label,xmin,xmax,step,tickstep = var_pars_list(var)

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$\\mathrm{"+label[1:-1]+"}$",fontsize=24)
    ax.set_ylabel("$\\mathrm{Fraction~of~SLAMS}$",fontsize=24)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,0.6)
    ax.tick_params(labelsize=20)
    weights = np.ones(var_list.shape)/float(var_list.size) # Normalise by total number of jets

    ax.set_yticks(np.arange(0.1,0.7,0.1))
    ax.set_yticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(0.1,0.7,0.1).astype(str)])

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax)
        hist = ax.hist(var_list,weights=weights,bins=bins)
        ax.set_xticks(np.array([10**0,10**1,10**2,10**3]))
        ax.set_xticklabels(np.array(["$\\mathtt{10^0}$","$\\mathtt{10^1}$","$\\mathtt{10^2}$","$\\mathtt{10^3}$"]))

    else:
        bins = np.arange(xmin,xmax+step,step)
        hist = ax.hist(var_list,bins=bins,weights=weights)
        ax.set_xticks(np.arange(xmin,xmax+tickstep,tickstep))
        ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin,xmax+tickstep,tickstep).astype(str)])

    if xmin == -xmax and 0.5*(xmax-xmin)%tickstep != 0.0:
        ax.set_xticks(np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep))
        if tickstep%1 != 0:
            ax.set_xticklabels(["$\\mathtt{"+lab+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep).astype(str)])
        else:
            ax.set_xticklabels(["$\\mathtt{"+str(int(lab))+"}$" for lab in np.arange(xmin+0.5*tickstep,xmax+0.5*tickstep,tickstep)])

    #ax.axvline(np.median(var_list), linestyle="dashed", color="black", linewidth=2)
    ax.annotate("med: %.1f\nstd: %.1f"%(np.median(var_list),np.std(var_list,ddof=1)), xy=(0.75,0.85), xycoords='axes fraction', fontsize=20, fontname="Computer Modern Typewriter")

    plt.title(",".join(runids),fontsize=24)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("SLAMS/figures/histograms/"+"_".join(runids)+"/"):
        try:
            os.makedirs("SLAMS/figures/histograms/"+"_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("SLAMS/figures/histograms/"+"_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")
    print("SLAMS/figures/histograms/"+"_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")

    plt.close(fig)

    return None

class PropReader:
    # Class for reading jet property files

    def __init__(self,ID,runid,start=580,fname=None):

        self.ID = ID
        self.runid = runid
        self.start = start
        self.sw_pars = sw_par_dict(runid)
        self.sw_pars[0] /= 1.0e+6
        self.sw_pars[1] /= 1.0e+3
        self.sw_pars[2] /= 1.0e-9
        self.sw_pars[3] /= 1.0e-9

        if type(fname) is not str:
            self.fname = str(start)+"."+ID+".props"
        else:
            self.fname = fname

        try:
            props_f = open("SLAMS/slams/"+runid+"/"+self.fname)
        except IOError:
            raise IOError("File not found!")

        props = props_f.read()
        props = props.split("\n")[1:]
        props = [line.split(",") for line in props]
        self.props = np.asarray(props,dtype="float")

        var_list = ["time","x_mean","y_mean","z_mean","A","Nr_cells","r_mean","theta_mean","phi_mean","size_rad","size_tan","x_vmax","y_vmax","z_vmax","n_avg","n_med","n_max","v_avg","v_med","v_max","B_avg","B_med","B_max","T_avg","T_med","T_max","TPar_avg","TPar_med","TPar_max","TPerp_avg","TPerp_med","TPerp_max","beta_avg","beta_med","beta_max","x_min","rho_vmax","b_vmax","pd_avg","pd_med","pd_max"]
        n_list = list(range(len(var_list)))
        self.var_dict = dict(zip(var_list,n_list))

    def read(self,name):
        if name not in self.var_dict:
            print("Variable not found!")
            return None
        else:
            return self.props[:,self.var_dict[name]]

    def amax_index(self):
        return self.read("A").argmax()

    def read_at_amax(self,name):
        return self.read(name)[self.amax_index()]

def slams_hist_script():

    runids = ["ABA","ABC","AEA","AEC"]
    #runids = ["BFD"]

    var_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A","death_distance"]

    for var in var_list:
        slams_all_hist(runids,var,time_thresh=10)

    return None

def slams_hist_script_vs(runids):

    var_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A","death_distance"]

    for var in var_list:
        slams_vs_hist(runids,var,time_thresh=10)

    return None

def hist_script_script():

    slams_hist_script()
    slams_hist_script_vs(["ABA","AEA"])
    slams_hist_script_vs(["ABC","AEC"])

    return None
