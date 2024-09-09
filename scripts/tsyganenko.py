import numpy as np
import geopack
import geopack.geopack as gp
import functools
import logging

def spherical_to_cartesian(r, theta, phi):
    '''
    r > 0
    0 < theta < pi
    -pi < phi < pi
    all are assumed to be numpy arrays of equal dimensions

    returns:  x, y, z   [tuple]
    '''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def tsyganenko_trace(x0, y0, z0, Txx = 't01', InternalB='dipole', Dst = -30, Kp = 4, Vx_sw = -750., N_sw = 1., Bx_imf = 0., By_imf = 0., Bz_imf = -5., R_inner = 0.99, R_outer=4.8, dir = None, maxloop=20000 ):

    '''
    Wrapper for Geopack's trace() routine, for tracing field lines along Tsyganenko model
    
    Inputs:
        x0: initial x-coordinate (GSE) of field tracing [R_E]  (float)
        y0: initial y-coordinate (GSE) of field tracing [R_E]  (float)
        z0: initial z-coordinate (GSE) of field tracing [R_E]  (float)

        --Keywords--
        Txx: (external) field model, by Tsyganenko publication year. options: 't89', 't96', 't01'
            Need to test: does 't04' work? Is geomagnetic activity estimated with Kp or Dst in that case?
        InternalB: internal model. options: 'dipole', 'igrf'
        Dst: Dst index in nT (geomagnetic activity), used for Txx= 't96', 't01' ('t04'?)
        Kp: Kp index 0-9 (geomagnetic activity), used for Txx='T89'
        Vx_sw: solar wind x-component of velocity [km/s] (GSE), other components assumed zero
        N_sw: solar wind density [cm-3]
        Bx_imf: driving solar wind imf Bx [nT] (GSE)
        By_imf: driving solar wind GSE imf By [nT] (GSE)
        Bz_imf: driving solar wind GSE imf Bz [nT] (GSE)
        R_inner: inner radius [R_E] of the tracing, if the tracing goes r<R_inner it stops
        R_outer: outer radius [R_E] of the tracing, if the tracing goes r>R_outer it stops
        dir: direction of the tracing relative to the magnetic field direction (parallel: dir= 1, anti-parallel: dir= -1)
             If unspecified (dir=None), dir traces against the field when z0>0 with the field when z0<0
             *** this is the opposite convention used by geopack's trace() function?
                 From geopack.py docs: "dir: Direction of tracing. dir = -1 for parallel; dir = 1 for anti-parallel."
    
    Returns:
        6-element tuple x,y,z,xx,yy,zz
        x: final x-coordinate(s) of field tracing [R_E]
        y: final y-coordinate(s) of field tracing [R_E]
        z: final z-coordinate(s) of field tracing [R_E]
        xx: numpy array containing the traced field line x-coordinates 
        yy: numpy array containing the traced field line y-coordinates 
        zz: numpy array containing the traced field line z-coordinates 

        ** All inputs and outputs are in GSE coordinate system **

    Example:
        x,y,z,xx,yy,zz = tsyganenko_trace(0,0, 1., Txx = 't01', Dst = -80, Kp = 4, Vx_sw = -750., N_sw = 4., Bx_imf = 0., By_imf = 0., Bz_imf = -10., R_inner = 1., R_outer=5., dir = None, maxloop = 20000 )

    Notes:

        Earth's dipole is assumed to point (~exactly) in the -z GSE direction

        Default solar wind and IMF parameters are for the EGI Vlasiator run, see Grandin et al. (2022?)
    
            Other runs:
            EGL (note: pulse arrives at magnetopause at roughly t=857 sec)
             (~before pulse arrival): Dst = -30 (Grandin et al. 2022), N_sw = 1, Bz_imf = -5
             (~after pulse arrival):  Dst = -80 (Horaites et al.), N_sw = 4, Bz_imf = -10
    
        The geopack package doesn't allow for easy specification of the dipole tilt angle.
        Rather, a UT was found when the tilt was nearly zero (see gp.recalc() line below)
        The specified UT results in a tilt angle of theta~1e-5 radians, or 6e-4 degrees
        Technically, the solutions should be rotated back by theta, to be compared with Vlasiator's zero-tilt runs
        But this small theta is within tolerance for most applications.

    '''

    # Constants

    mass_p = 1.660539e-27 # kg
    mass_e = 9.109384e-31 # kg
    qe = 1.602177e-19 # C
    erg = 6.2415e11 # eV
    deg2rad = np.pi/180.

    # =======================
    # Parameters used in the T01 (or T96) model

    #P_sw = (2e-6)*N_sw*Vx_sw**2 # in nPa, OMNI formula            (CHANGE: why 2 in coefficient?)
    P_sw = (1e21)*mass_p*N_sw*Vx_sw**2 # [nPa] dynamic pressure 

    Bs_sw = max(0.,-Bz_imf) # southward component (0 if Bz_imf > 0)
    Bperp_imf = np.sqrt(By_imf**2+Bz_imf**2)  # perp B-component   (V_sw is in -x direction)
    hBperp_sw = (Bperp_imf/40.)**2 / (1. + (Bperp_imf/40.))
    if Bz_imf > 0.:
        clockangle = np.arctan(By_imf/abs(Bz_imf))
    elif Bz_imf < 0.: # the +1e-3 is to get +180 deg if By_imf = 0 and Bz_imf < 0
        clockangle = np.sign(By_imf+1e-3) * (np.pi - np.arctan(abs(By_imf/Bz_imf)))
    else: # case with Bz_imf = 0; here if By_imf = 0 the clock angle is undefined, so 0 is as acceptable as +/-90 deg
        clockangle = np.sign(By_imf) * np.pi/2.
    
    G1 = abs(Vx_sw) * hBperp_sw * np.sin(clockangle/2.)**3
    G2 = 0.005 * abs(Vx_sw) * Bs_sw
    
    if Txx == 't89':
        paramTxx = Kp+1
    else:
        paramTxx = np.array([P_sw,Dst,By_imf,Bz_imf,G1,G2,0.,0.,0.,0.]) 

    # --- Mapping using geopack (dipole + T89) ---

    t_zerotilt = 1583962800 # 11 March 2020, 21:40 UT
    gp.recalc(t_zerotilt,vxgse=Vx_sw) # field initialisation at time with ~zero dipole tilt
   
    ## convert initial position from GSE to GSM coordinate system
    #x0_gsm = x0; y0_gsm = -y0; z0_gsm = -z0                  # y-> -y, z-> -z
    # in zero-tilt case, GSE = GSM, see https://www.spenvis.oma.be/help/background/coortran/coortran.html
    x0_gsm = x0; y0_gsm = y0; z0_gsm = z0

    if dir is None:
        dir = 1 if z0_gsm>0 else -1         # trace outwards by default (when R_inner = 1)
                                            # (B-field is radial at northern magnetic hemisphere)

#    gp_dir = -dir                           # Geopack's trace() has a weird convention for the tracing direction (dir)

#    x_gsm, y_gsm, z_gsm, xx_gsm, yy_gsm, zz_gsm = gp.trace(xi=x0_gsm,yi=y0_gsm,zi=z0_gsm,dir=gp_dir,r0=R_inner,
#                                                           rlim=R_outer,inname=InternalB,exname=Txx,parmod=paramTxx)

    x_gsm, y_gsm, z_gsm, xx_gsm, yy_gsm, zz_gsm = gp.trace(xi=x0_gsm,yi=y0_gsm,zi=z0_gsm,dir=dir,r0=R_inner,
                                                           rlim=R_outer,inname=InternalB,exname=Txx,parmod=paramTxx, maxloop=maxloop)

    # convert final position from GSM to GSE coordinate system
    #x_gse = x_gsm;   y_gse = -y_gsm;   z_gse = -z_gsm         # y-> -y, z-> -z
    #xx_gse = xx_gsm; yy_gse = -yy_gsm; zz_gse = -zz_gsm       # ""
    # in zero-tilt case, GSE = GSM, see https://www.spenvis.oma.be/help/background/coortran/coortran.html
    x_gse = x_gsm;   y_gse = y_gsm;   z_gse = z_gsm
    xx_gse = xx_gsm; yy_gse = yy_gsm; zz_gse = zz_gsm

    return x_gse, y_gse, z_gse, xx_gse, yy_gse, zz_gse
    

def tsyganenko_open(phi, lat, R_inner = 1, R_outer = 15, **kwargs):
    '''
    Checks whether the field line starting at radius ~R_inner
    will trace out to beyond R_outer. R_outer is chosen to be large, so if 
    a magnetic field line makes it out this far, it may be assumed to be open
    If the tracing crosses R_inner, the field line is closed

    Inputs: 
       phi: ~longitude, in degrees                    -180<=phi<=180
       lat: geographic latitude, in degrees           -90<=lat<=90

           e.g.,
              (phi=0, lat = 0) corresponds with subsolar point
              (phi=anything, lat = 90) corresponds with geographic north pole 

       R_inner: inner radius [R_E] of the tracing, if the tracing goes r<R_inner it stops
       R_outer: outer radius [R_E] of the tracing, if the tracing goes r>R_outer it stops

       Keywords (and kwargs) all get passed to tsyganenko_trace()

    Returns:
        Boolean---
            True (if open)
            False (if closed)

    Example: 
        tsyganenko_open(0, 45)   # returns False
        tsyganenko_open(0, 80)   # returns True
    '''
    #convert phi, lat to cartesian
    theta = 90 - lat   # "co-latitude"
    x0, y0, z0 = spherical_to_cartesian(R_inner*1.01, theta * np.pi / 180, phi * np.pi / 180 )
    logging.info("Cartesian coords : " + str((x0, y0, z0)))
    x,y,z,xx,yy,zz = tsyganenko_trace(x0, y0, z0, R_inner = R_inner, R_outer = R_outer, **kwargs)
    logging.info("Trace: " + str((x, y, z)))
    r = (x**2 + y**2 + z**2)**0.5
    if r <= R_inner*1.01:
        # otherwise, the results will be spurious
        return False
    elif (r>R_inner*1.01) and (r<R_outer * 0.99):
        # note, maxloop keyword needs to be large enough so that one of the boundaries R_inner or R_outer is reached
        # in this case, the field line connectivity is undetermined, should try again with larger maxloop
        return None
    elif r >= (R_outer * 0.99):
        return True




def tsyganenko_ocb(phi, lat_range=[0,90], nsteps = 10, **kwargs):
    '''
    Find the open/closed field line boundary (OCB)
    algorithm uses a binary search within a specified latitude range

    Inputs:
        phi: ~longitude, in degrees                                                     -180<=phi<=180
        lat_range: 2-element list or numpy array, 
                   containing min. and max. latidues [degrees] to search within         -90<=lat<=90

        kwargs are passed to tsyanenko_open()
  
    Returns:

        Estimate of the OCB latitude at the given phi, in degrees

    Notes:
        Algorithm assumes OBC latitude is a single-valued function of phi

        Theoretical accuracy of the OCB determination is ~ (lat_range[1]-lat_range[0]) / 2^nsteps

    Example:   # dayside (noon) cusp

        tsyganenko_ocb(0, lat_range = [60,80], nsteps = 10, 
                       Txx = 't01', Dst = -80, Kp = 4, Vx_sw = -750., N_sw = 4., Bx_imf = 0., By_imf = 0., Bz_imf = -10., 
                       R_inner = 1., R_outer=15., dir = None, maxloop = 10000)

    '''
    if (lat_range[0] * lat_range[1]) < 0:
        logging.info('Both elements of lat_range must have the same sign! (search within a single hemisphere)')

    lat_guess = (lat_range[1]+lat_range[0])/2.
    lat_delta = np.abs((lat_range[1]-lat_range[0])/2.)
    for i in range(nsteps):
        logging.info(lat_guess)
        tf = tsyganenko_open(phi, lat_guess, **kwargs)
        logging.info(tf)
        if tf == True:
            # field line open, OCB is more equatorward
            lat_guess = np.sign(lat_guess) * (np.abs(lat_guess) - lat_delta)
        elif tf == False:
            # field line closed, OCB is more poleward
            lat_guess = np.sign(lat_guess) * (np.abs(lat_guess) + lat_delta)
        lat_delta /= 2.
    return lat_guess





def tsyganenko_b(x, y, z, Txx = 't01', InternalB='dipole', Dst = -30, Kp = 4, Vx_sw = -750., N_sw = 1., Bx_imf = 0., By_imf = 0., Bz_imf = -5. ):

    '''
    Wrapper for Geopack's trace() routine, for tracing field lines along Tsyganenko model
    
    Inputs:
        x0: x-coordinate(s) (GSE) to evaluate the field [R_E]  (float)
        y0: y-coordinate(s) (GSE) to evaluate the field [R_E]  (float)
        z0: z-coordinate(s) (GSE) to evaluate the field [R_E]  (float)

        --Keywords--
        Txx: (external) field model, by Tsyganenko publication year. options: 't89', 't96', 't01'
            Need to test: does 't04' work? Is geomagnetic activity estimated with Kp or Dst in that case?
        InternalB: internal model. options: 'dipole', 'igrf'
        Dst: Dst index in nT (geomagnetic activity), used for Txx= 't96', 't01' ('t04'?)
        Kp: Kp index 0-9 (geomagnetic activity), used for Txx='T89'
        Vx_sw: solar wind x-component of velocity [km/s] (GSE), other components assumed zero
        N_sw: solar wind density [cm-3]
        Bx_imf: driving solar wind imf Bx [nT] (GSE)
        By_imf: driving solar wind GSE imf By [nT] (GSE)
        Bz_imf: driving solar wind GSE imf Bz [nT] (GSE)
        R_inner: inner radius [R_E] of the tracing, if the tracing goes r<R_inner it stops
        R_outer: outer radius [R_E] of the tracing, if the tracing goes r>R_outer it stops
        dir: direction of the tracing relative to the magnetic field direction (parallel: dir= 1, anti-parallel: dir= -1)
             If unspecified (dir=None), dir traces against the field when z0>0 with the field when z0<0
             *** this is the opposite convention used by geopack's trace() function?
                 From geopack.py docs: "dir: Direction of tracing. dir = -1 for parallel; dir = 1 for anti-parallel."
    
    Returns:
        3-element tuple bx,by,bz
        bx: x-component(s) of field [nT]
        by: y-component(s) of field [nT]
        bz: z-component(s) of field [nT]

        ** All positional inputs can either be single values or numpy arrays (or lists) **
        ** Depending on the inputs, the outputs bx, by, bz will either be single numbers or numpy arrays **
        ** All inputs and outputs are in GSE coordinate system **

    Example:
        bx,by,bz = tsyganenko_b(0,0, 1., Txx = 't01', Dst = -80, Kp = 4, Vx_sw = -750., N_sw = 4., Bx_imf = 0., By_imf = 0., Bz_imf = -10. )

        bx,by,bz = tsyganenko_b([0,0,0],[0,0,0], [1.,2,3], Txx = 't01', Dst = -80, Kp = 4, Vx_sw = -750., N_sw = 4., Bx_imf = 0., By_imf = 0., Bz_imf = -10. )

    Notes:

        Earth's dipole is assumed to point (~exactly) in the -z GSE direction

        Default solar wind and IMF parameters are for the EGI Vlasiator run, see Grandin et al. (2022?)
    
            Other runs:
            EGL (note: pulse arrives at magnetopause at roughly t=857 sec)
             (~before pulse arrival): Dst = -30 (Grandin et al. 2022), N_sw = 1, Bz_imf = -5
             (~after pulse arrival):  Dst = -80 (Horaites et al.), N_sw = 4, Bz_imf = -10
    
        The geopack package doesn't allow for easy specification of the dipole tilt angle.
        Rather, a UT was found when the tilt was nearly zero (see gp.recalc() line below)
        The specified UT results in a tilt angle of theta~1e-5 radians, or 6e-4 degrees
        Technically, the solutions should be rotated back by theta, to be compared with Vlasiator's zero-tilt runs
        But this small theta is within tolerance for most applications.

    '''

    try:
        # if x, y, z arr arrays
        f = functools.partial(tsyganenko_b, Txx = Txx, InternalB=InternalB, Dst = Dst, Kp = Kp,
                              Vx_sw = Vx_sw, N_sw = N_sw, Bx_imf = Bx_imf, By_imf = By_imf, Bz_imf = Bz_imf)
        barray = np.array(list(map(f, x, y, z )))
        return barray[:,0], barray[:,1], barray[:,2]    # bx, by, bz (arrays)
    except:
        #if x,y, z are single values (not arrays), run the program below
    # Constants    
        mass_p = 1.660539e-27 # kg
        mass_e = 9.109384e-31 # kg
        qe = 1.602177e-19 # C
        erg = 6.2415e11 # eV
        deg2rad = np.pi/180.
    
        # =======================
        # Parameters used in the T01 (or T96) model
    
        #P_sw = (2e-6)*N_sw*Vx_sw**2 # in nPa, OMNI formula            (CHANGE: why 2 in coefficient?)
        P_sw = (1e21)*mass_p*N_sw*Vx_sw**2 # [nPa] dynamic pressure 
    
        Bs_sw = max(0.,-Bz_imf) # southward component (0 if Bz_imf > 0)
        Bperp_imf = np.sqrt(By_imf**2+Bz_imf**2)  # perp B-component   (V_sw is in -x direction)
        hBperp_sw = (Bperp_imf/40.)**2 / (1. + (Bperp_imf/40.))
        if Bz_imf > 0.:
            clockangle = np.arctan(By_imf/abs(Bz_imf))
        elif Bz_imf < 0.: # the +1e-3 is to get +180 deg if By_imf = 0 and Bz_imf < 0
            clockangle = np.sign(By_imf+1e-3) * (np.pi - np.arctan(abs(By_imf/Bz_imf)))
        else: # case with Bz_imf = 0; here if By_imf = 0 the clock angle is undefined, so 0 is as acceptable as +/-90 deg
            clockangle = np.sign(By_imf) * np.pi/2.
        
        G1 = abs(Vx_sw) * hBperp_sw * np.sin(clockangle/2.)**3
        G2 = 0.005 * abs(Vx_sw) * Bs_sw
        
        if Txx == 't89':
            paramTxx = Kp+1
        else:
            paramTxx = np.array([P_sw,Dst,By_imf,Bz_imf,G1,G2,0.,0.,0.,0.]) 
    
        # --- Mapping using geopack (dipole + T89) ---
    
        t_zerotilt = 1583962800 # 11 March 2020, 21:40 UT, is this step necessary?
                                # Maybe redundant because dipole tilt angle specified in call to t01()
        ps = gp.recalc(t_zerotilt,vxgse=Vx_sw) # field initialisation at time with ~zero dipole tilt
       
        logging.info('ps: ' + str(ps))

        # convert initial position from GSE to GSM coordinate system
        # x_in = [...];   y_in = [...];   z_in = [...]
        # in zero-tilt case, GSE = GSM, see https://www.spenvis.oma.be/help/background/coortran/coortran.html
        x_in = x; y_in = y; z_in = z
    
        if Txx == 't89':
            tfunc = geopack.t89.t89
        if Txx == 't96':
            tfunc = geopack.t96.t96
        if Txx == 't01':
            tfunc = geopack.t01.t01
        if Txx == 't04':
            tfunc = geopack.t04.t04
    
        b0xgsm,b0ygsm,b0zgsm = gp.dip(x_in,y_in,z_in)    		# calc dipole B in GSM.
        dbxgsm,dbygsm,dbzgsm = tfunc(paramTxx, ps, x_in,y_in,z_in)       # 0 tilt angle
        bxgsm,bygsm,bzgsm = [b0xgsm+dbxgsm,b0ygsm+dbygsm,b0zgsm+dbzgsm]
    
        # convert final position from GSM to GSE coordinate system
        #bxgse = [...];   bygse = [...];   bzgse = [...]
        # in zero-tilt case, GSE = GSM, see https://www.spenvis.oma.be/help/background/coortran/coortran.html
        bxgse = bxgsm;   bygse = bygsm;   bzgse = bzgsm         # GSE = GSM for zero tilt
    
        return bxgse, bygse, bzgse   # nT
