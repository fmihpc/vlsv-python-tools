import numpy as np
import math
import scipy.optimize
import logging

# Script for calculating shock crossing values from Rankine-Hugoniot relations
# Feed it upstream and shock values in given reference frame, outputs the dHT state
# intput example:
# obliqueshock.rankine(5e5,1.0e6,[-750e3,0,0],[3.5355e-9,0,-3.5355e-9],[1,0,0],0)
# where T_upstream = 500 kK
#       n_upstream = 1/cc
#       inflow plasma speed is 750 km/s in -X
#       upstream magnetic field is 5 nT at 45 degree angle
#       Shock front points in +X direction and is stationary in input frame

mu0 = 4*math.pi*1.e-7
mp = 1.67e-27
c = 299792458
k = 1.3806506e-23
Gamma = 5.0/3.0

def polynome(X, theta, V1sq, beta1, vA1sq, Gamma, vs1sq):
    return (V1sq-X*vA1sq)**2 * (X*vs1sq + 0.5*V1sq*np.cos(theta)**2*(X*(Gamma-1)-(Gamma+1))) + 0.5*vA1sq*V1sq*X*np.sin(theta)**2 * ((Gamma+X*(2-Gamma))*V1sq - X*vA1sq*((Gamma+1)-X*(Gamma-1)))

def newtonmethod(theta, V1sq, beta1, vA1sq, Gamma, vs1sq):
    calctemp1 = 1. + 0.5*Gamma*beta1
    cos12 = np.cos(theta)**2
    sin12 = np.sin(theta)**2
    MA2=V1sq/vA1sq
    logging.info("MA: " + str(np.sqrt(MA2)))
    Ztry = max( ((0.5/cos12)*(calctemp1 + np.sqrt(calctemp1**2 - 2.*Gamma*beta1*cos12)) -1.),
                ((0.5/cos12)*(calctemp1 - np.sqrt(calctemp1**2 - 2.*Gamma*beta1*cos12)) -1.), 0.)
    # First (root for M**2) -1
    # Second and third (root for M**2) -1
    fform = (1. +Ztry)*((Ztry**2)*8.*cos12 +(3. -5.*Ztry)*sin12) -(Ztry**2)*5.*beta1
    gform = (1. +Ztry)*((Ztry**2)*2.*cos12 +(3. +Ztry)*sin12)
    Rtry = fform/gform
    M2try = (1. +Ztry)*Rtry
    rstep = 1.0
    compr = 0.
    while ((rstep >= 0.0001) and (compr < 0.001)):
        Ztry = max( ((0.5/cos12)*(calctemp1 + np.sqrt(calctemp1**2 - 2.*Gamma*beta1*cos12)) -1.),
                    ((0.5/cos12)*(calctemp1 - np.sqrt(calctemp1**2 - 2.*Gamma*beta1*cos12)) -1.), 0.)
        fform = (1. +Ztry)*((Ztry**2)*8.*cos12 +(3. -5.*Ztry)*sin12) -(Ztry**2)*5.*beta1
        gform = (1. +Ztry)*((Ztry**2)*2.*cos12 +(3. +Ztry)*sin12)
        Rtry = fform/gform
        M2try = (1. +Ztry)*Rtry
    
        while (abs(M2try - MA2) > 0.0001):
            fderi = (Ztry**2)*8.*cos12 +(3. -8.*Ztry)*sin12 -10.*Ztry*beta1 +(1. +Ztry)*(16.*Ztry*cos12 -5.*sin12)
            gderi = (Ztry**2)*2.*cos12 +(3. +Ztry)*sin12 +(1. +Ztry)*(4.*Ztry*cos12 +sin12)
            rderi = (gform*fderi-fform*gderi)/(gform**2)
            m2deri = (1. +Ztry)*rderi + Rtry
            
            # Newton step forward
            Ztry = Ztry + (MA2 - M2try)/m2deri * 0.5*rstep
            
            # Calculate new Rtry and M2try
            fform = (1. +Ztry)*((Ztry**2)*8.*cos12 +(3. -5.*Ztry)*sin12) -(Ztry**2)*5.*beta1
            gform = (1. +Ztry)*((Ztry**2)*2.*cos12 +(3. +Ztry)*sin12)
            Rtry = fform/gform
            M2try = (1. +Ztry)*Rtry

        if (Rtry <= MA2):
            compr = Rtry
        rstep = rstep * 0.1
    return compr

def rankine(Tu, rhou, V, B, n, Vsh):
    # V, B, n are vectors
    V = np.array(V)
    B = np.array(B)
    n = np.array(n)
    
    # normalise n
    n = n/np.linalg.norm(n)
    Vshvect = n*Vsh
    
    Vu = V - Vshvect
    Bu = B
    
    pu = rhou * k * Tu
    vHT = np.cross(n, np.cross(Vu, Bu)) / np.dot(n,Bu)
    #logging.info("The de Hoffmann Teller transformation velocity is ", vHT)

    VuHT = Vu - vHT
    BuHT = Bu #+ 1/c**2 * np.cross(vHT, np.cross(-Vu,Bu)) # Eu = -Vu X Bu
    #BuHT2 = Bu + 1/c**2 * np.cross(vHT, np.cross(-Vu,Bu)) # Eu = -Vu X Bu
    #logging.info("BuHT "+str(BuHT)+" alt "+str(BuHT2))
    
    Emotional = -np.cross(VuHT,BuHT)
    #logging.info("Verify: Motional E-field in HT frame: ", Emotional)

    theta = np.arccos(np.dot(BuHT,n)/np.linalg.norm(BuHT))

    VnuHT = np.dot(VuHT, n)
    VtuHT = np.linalg.norm(np.cross(VuHT,n))
    # VtuHT = np.sqrt(sum((VuHT - VnuHT .* n).**2))
    BnuHT = np.dot(BuHT, n)
    BtuHT = np.linalg.norm(np.cross(BuHT,n))
    # BtuHT = np.sqrt(sum((BuHT - BnuHT .* n).**2))

    beta1 = 2.0*mu0*pu / np.dot(Bu,Bu)
    vAusq = np.dot(BuHT,BuHT)/(mu0*mp*rhou)
    vsusq = Gamma*pu/(mp*rhou)

    # compression ratio
    sol = scipy.optimize.root(polynome, 2., args=(theta, np.dot(Vu,Vu), beta1, vAusq, Gamma, vsusq))
    X = sol.x
    # Alternative own Newton method
    X2 = newtonmethod(theta, np.dot(Vu,Vu), beta1, vAusq, Gamma, vsusq)
    logging.info("X " +str(X) + " X2 " + str(X2))
    
    VuHTsq = np.dot(VuHT,VuHT)
    VndHT = VnuHT / X
    VtdHT = VtuHT * (VuHTsq - vAusq)/(VuHTsq-X*vAusq)
    VdHTsq = VndHT**2 + VtdHT**2
    pd = pu * (X+((Gamma-1.)*X*VuHTsq)/(2.0*vsusq)*(1-VdHTsq/VuHTsq))
    rhod = rhou * X
    BndHT = BnuHT
    BtdHT = BtuHT * (VuHTsq-vAusq)*X/(VuHTsq-X*vAusq)
    # BdHT = np.sqrt(BndHT**2+BtdHT**2)

    nPlane = np.cross(Vu, Bu) / np.linalg.norm(np.cross(Vu, Bu)) # normal to the plane containing B and V
    t = np.cross(n, nPlane) / np.linalg.norm(np.cross(n, nPlane))
    # transversal direction, i.e. normal to n and within the (B,V) plane (i.e. in turn normal to nPlane)
    
    BdHT = (BndHT * n) + (BtdHT * t)
    VdHT = (VndHT * n) + (VtdHT * t)
    
    Bd = BdHT #- 1/c**2 * cross(vHT, cross(-Vu,Bu))
    #Bd2 = BdHT - 1/c**2 * np.cross(vHT, np.cross(-Vu,Bu))
    #logging.info("Bd "+str(Bd)+" alt "+str(Bd2))

    Vd = VdHT + vHT + Vshvect
    XB = np.linalg.norm(Bd)/np.linalg.norm(Bu)
    
    #//Td = pd / (Gamma * rhod * k)
    Td = pd / (rhod * k)
                      
    logging.info("Gas compression ratio " + str(X[0]))
    logging.info("Magnetic compression ratio " + str(XB))

    logging.info("")
    logging.info("This determines the dHT upstream state")
    logging.info("Density " + str(rhou))
    logging.info("Temperature " + str(Tu))
    #logging.info("Thermal pressure ",pu[0])
    logging.info(" V " + str(VuHT))
    logging.info(" B " + str(BuHT))

    logging.info("")
    logging.info("This determines the dHT downstream state")
    logging.info("Density " + str(rhod[0]))
    logging.info("Temperature " + str(Td[0]))
    #logging.info("Thermal pressure ",pd[0])
    logging.info(" V " + str(VdHT))
    logging.info(" B " + str(BdHT))

    return X[0], XB
