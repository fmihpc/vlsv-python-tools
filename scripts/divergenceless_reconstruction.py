# Reconstruction and interpolation of magntic field from face-average values
# Based on Balsara 2011 (https://doi.org/10.48550/arXiv.0811.2192)

import pytools as pt
import numpy as np
from scipy.special import legendre

P = tuple(map(lambda n: legendre(n), (0, 1, 2, 3, 4)))
order = 2
# y and z are simply cyclic rotations of this
def interpolate_x(a, x, y, z):
    B_x = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_x += a[i][j][k] * P[i](x) * P[j](y) * P[k](z)
    return B_x

def interpolate_y(b, x, y, z):
    return interpolate_x(b, y, z, x)

def interpolate_z(c, x, y, z):
    return interpolate_x(c, z, x, y)

# y and z are simply cyclic rotations of this
def interpolate_dbxdx(a, x, y, z):
    B_x = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_x += a[i][j][k] * np.polyder(P[i])(x) * P[j](y) * P[k](z)
    return B_x

def interpolate_dbydx(b, x, y, z):
    B_y = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_y += b[i][j][k] * P[i](y) * P[j](z) * np.polyder(P[k])(x)
    return B_y
    #return interpolate_dbxdx(b, y, z, x)

def interpolate_dbzdx(c, x, y, z):
    B_z = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_z += c[i][j][k] * P[i](z) * np.polyder(P[j])(x) * P[k](y)
    return B_z
    #return interpolate_dbxdx(c, z, x, y)

def interpolate_dbxdy(a, x, y, z):
    B_x = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_x += a[i][j][k] * P[i](x) * np.polyder(P[j])(y) * P[k](z)
    return B_x

def interpolate_dbydy(b, x, y, z):
    B_y = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_y += b[i][j][k] * np.polyder(P[i])(y) * P[j](z) * P[k](x)
    return B_y
    #return interpolate_dbxdy(b, y, z, x)

def interpolate_dbzdy(c, x, y, z):
    B_z = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_z += c[i][j][k] * P[i](z) * P[j](x) * np.polyder(P[k])(y)
    return B_z
    # return interpolate_dbxdy(c, z, x, y)

def interpolate_dbxdz(a, x, y, z):
    B_x = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_x += a[i][j][k] * P[i](x) * P[j](y) * np.polyder(P[k])(z)
    return B_x

def interpolate_dbydz(b, x, y, z):
    B_y = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_y += b[i][j][k] * P[i](y) * np.polyder(P[j])(z) * P[k](x)
    return B_y
    # return interpolate_dbxdz(b, y, z, x)

def interpolate_dbzdz(c, x, y, z):
    B_z = 0
    for i in range(order+1):
        for j in range(order):
            for k in range(order):
                B_z += c[i][j][k] * np.polyder(P[i])(z) * P[j](x) * P[k](y)
    return B_z
    # return interpolate_dbxdz(c, z, x, y)


# Returns moments as [ijk, order, x, y, z]
def solve_moments_from_B(fg_b):
    x_moments = solve_x_moments(fg_b[:, :, :, 0])
    y_moments = solve_y_moments(fg_b[:, :, :, 1])
    z_moments = solve_z_moments(fg_b[:, :, :, 2])
    return (x_moments, y_moments, z_moments)

# Returns moments for x as [order, x, y, z]
# With order (0, y, z, yy, yz, zz, yyy, yyz, yzz, zzz)
# y and z as cyclic rotations
def solve_x_moments(B_x):
    x_moments = np.zeros((10,) + np.shape(B_x))
    x_moments[0] = B_x
    x_moments[1:3] = np.gradient(B_x)[1:3]
    start = 3
    for i in range(2):
        j = i + 1
        x_moments[start:start+3-j] = np.gradient(x_moments[1+i])[j:3]
        start += (3-j)
    for i in range(3):
        j = i + 1 if i < 2 else 2
        x_moments[start:start+3-j] = np.gradient(x_moments[3+i])[j:3]
        start += (3-j)
    return x_moments

def solve_y_moments(B_y):
    return np.transpose(solve_x_moments(np.transpose(B_y, (1, 2, 0))), (0, 3, 1, 2))

def solve_z_moments(B_z):
    return np.transpose(solve_x_moments(np.transpose(B_z, (2, 0, 1))), (0, 2, 3, 1))

# Solves a, b, c components with a[x, y, z], b[y, z, x] and c[z, x, y] up to given order
# Input B should be the output of solve_moments_from_B
# Might be deprecated. A lot faster than calculating all coefficients but that's only a few seconds anyway for a 100^3 array
def solve_coefficients(B_moments, xyz):
    abc = np.zeros((3, order+1, order, order))
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # 4th order
    if (order > 3):
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            coords = (x if i else x + 1, y if k else y + 1, z if j else z + 1)
            a = abc[i]
            Bx = B_moments[i]

            a[0][3][0] = 1/2 * (Bx[6][xyz] + Bx[6][coords])
            a[0][2][1] = 1/2 * (Bx[7][xyz] + Bx[7][coords])
            a[0][1][2] = 1/2 * (Bx[8][xyz] + Bx[8][coords])
            a[0][0][3] = 1/2 * (Bx[9][xyz] + Bx[9][coords])

            a[1][3][0] = Bx[6][xyz] - Bx[6][coords]
            a[1][2][1] = Bx[7][xyz] - Bx[7][coords]
            a[1][1][2] = Bx[8][xyz] - Bx[8][coords]
            a[1][0][3] = Bx[9][xyz] - Bx[9][coords]

        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            a = abc[i]
            b = abc[j]
            c = abc[k]
            
            # Should be correct, check later
            a[4][0][0] = -1/4 * (b[1][0][3] + c[1][3][0])
            a[3][1][0] = -7/30 * c[1][2][1]
            a[3][0][1] = -7/30 * b[1][1][2]
            a[2][2][0] = -3/20 * c[1][1][2]
            a[2][0][2] = -3/20 * b[1][2][1]

    # 3rd order
    if (order > 2):
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            coords = (x if i else x + 1, y if k else y + 1, z if j else z + 1)
            a = abc[i]
            Bx = B_moments[i]

            a[0][2][0] = 1/2 * (Bx[3][xyz] + Bx[3][coords]) - 1/6 * a[2][2][0]
            a[0][1][1] = 1/2 * (Bx[4][xyz] + Bx[4][coords])
            a[0][0][2] = 1/2 * (Bx[5][xyz] + Bx[5][coords]) - 1/6 * a[2][0][2]

            a[1][2][0] = Bx[3][xyz] - Bx[3][coords]
            a[1][1][1] = Bx[4][xyz] - Bx[4][coords]
            a[1][0][2] = Bx[5][xyz] - Bx[5][coords]
        
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            a = abc[i]
            b = abc[j]
            c = abc[k]

            # Should be correct, check later
            a[3][0][0] = -1/3 * (b[1][0][2] + c[1][2][0])
            a[2][1][0] = -1/4 * c[1][1][1]
            a[2][0][1] = -1/4 * b[1][1][1]

    # 2nd order
    if (order > 1):
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            coords = (x if i else x + 1, y if k else y + 1, z if j else z + 1)
            a = abc[i]
            Bx = B_moments[i]

            a[0][1][0] = 1/2 * (Bx[1][xyz] + Bx[1][coords]) - 1/6 * a[2][1][0]
            a[0][0][1] = 1/2 * (Bx[2][xyz] + Bx[2][coords]) - 1/6 * a[2][0][1]

            a[1][1][0] = (Bx[1][xyz] - Bx[1][coords])
            if order > 2:
                a[1][1][0] -= 1/10 * a[3][1][0]
            a[1][0][1] = (Bx[2][xyz] - Bx[2][coords]) 
            if order > 2:
                a[1][1][0] -= 1/10 * a[3][0][1]
        
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            a = abc[i]
            b = abc[j]
            c = abc[k]

            # Should be correct, check later
            a[2][0][0] = -1/2 * (b[1][0][1] + c[1][1][0])
            if order > 3: 
                a[2][0][0] -= 3/35 * a[4][0][0] 
            if order > 2:
                a[2][0][0] -= 1/20 * (b[3][0][1] + c[3][1][0])

    # 1st order
    if (order > 0):
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            coords = (x if i else x + 1, y if k else y + 1, z if j else z + 1)
            a = abc[i]
            Bx = B_moments[i]

            a[1][0][0] = (Bx[0][xyz] - Bx[0][coords])
            if order > 2:
                 a[1][0][0] -= 1/10 * a[3][0][0]
    
    # 0th order
    for i in range(3):
        j = (i+1) % 3
        k = (i+2) % 3
        coords = (x if i else x + 1, y if k else y + 1, z if j else z + 1)
        a = abc[i]
        Bx = B_moments[i]

        a[0][0][0] = 1/2 * (Bx[0, x, y, z] + Bx[0][coords])
        if order > 1:
            a[0][0][0] -= 1/6 * a[2][0][0] 
        if order > 3:
            a[0][0][0] -= 1/70 * a[4][0][0]

    # Check constraint:
    #test = abc[0][1][0][0] + abc[1][1][0][0] + abc[2][1][0][0] + 1/10 * (abc[0][3][0][0] + abc[1][3][0][0] + abc[2][3][0][0])
    #if abs(test) > 0:
    #    print("Something went wrong, sum (17) is " + str(test))

    return abc

def neighborsum(a, idx):
    second = a[:-1, :-1, :-1]
    if idx == 0:
        first = a[1:, :-1, :-1]
    elif idx == 1:
        first = a[:-1, 1:, :-1]
    elif idx == 2:
        first = a[:-1, :-1, 1:]
    return first + second

def neighbordiff(a, idx):
    second = a[:-1, :-1, :-1]
    if idx == 0:
        first = a[1:, :-1, :-1]
    elif idx == 1:
        first = a[:-1, 1:, :-1]
    elif idx == 2:
        first = a[:-1, :-1, 1:]
    return first - second

# Solves a, b, c components with a[x, y, z], b[y, z, x] and c[z, x, y] up to given order
# Input B should be the output of solve_moments_from_B
def solve_all_coefficients(B_moments):
    shp = np.shape(B_moments[0][0])
    abc = np.zeros((3, order+1, order, order, shp[0] - 1, shp[1] - 1, shp[2] - 1))

    # 4th order
    if (order > 3):
        for i in range(3):
            a = abc[i]
            Bx = B_moments[i]

            a[0][3][0] = 1/2 * neighborsum(Bx[6], i)
            a[0][2][1] = 1/2 * neighborsum(Bx[7], i)
            a[0][1][2] = 1/2 * neighborsum(Bx[8], i)
            a[0][0][3] = 1/2 * neighborsum(Bx[9], i)

            a[1][3][0] = neighbordiff(Bx[6], i)
            a[1][2][1] = neighbordiff(Bx[7], i)
            a[1][1][2] = neighbordiff(Bx[8], i)
            a[1][0][3] = neighbordiff(Bx[9], i)

        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            a = abc[i]
            b = abc[j]
            c = abc[k]
            
            # Should be correct, check later
            a[4][0][0] = -1/4 * (b[1][0][3] + c[1][3][0])
            a[3][1][0] = -7/30 * c[1][2][1]
            a[3][0][1] = -7/30 * b[1][1][2]
            a[2][2][0] = -3/20 * c[1][1][2]
            a[2][0][2] = -3/20 * b[1][2][1]

    # 3rd order
    if (order > 2):
        for i in range(3):
            a = abc[i]
            Bx = B_moments[i]

            a[0][2][0] = 1/2 * neighborsum(Bx[3], i) - 1/6 * a[2][2][0]
            a[0][1][1] = 1/2 * neighborsum(Bx[4], i)
            a[0][0][2] = 1/2 * neighborsum(Bx[5], i) - 1/6 * a[2][0][2]

            a[1][2][0] = neighbordiff(Bx[3], i)
            a[1][1][1] = neighbordiff(Bx[4], i)
            a[1][0][2] = neighbordiff(Bx[5], i)
        
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            a = abc[i]
            b = abc[j]
            c = abc[k]

            # Should be correct, check later
            a[3][0][0] = -1/3 * (b[1][0][2] + c[1][2][0])
            a[2][1][0] = -1/4 * c[1][1][1]
            a[2][0][1] = -1/4 * b[1][1][1]

    # 2nd order

    if (order > 1):
        for i in range(3):
            a = abc[i]
            Bx = B_moments[i]

            a[0][1][0] = 1/2 * neighborsum(Bx[1], i) - 1/6 * a[2][1][0]
            a[0][0][1] = 1/2 * neighborsum(Bx[2], i) - 1/6 * a[2][0][1]

            a[1][1][0] = neighbordiff(Bx[1], i)
            if order > 2:
                a[1][1][0] -= 1/10 * a[3][1][0]
            a[1][0][1] = neighbordiff(Bx[2], i) 
            if order > 2:
                a[1][0][1] -= 1/10 * a[3][0][1]
        
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            a = abc[i]
            b = abc[j]
            c = abc[k]

            # Should be correct, check later
            a[2][0][0] = -1/2 * (b[1][0][1] + c[1][1][0])
            if order > 3:
                a[2][0][0] -= 3/35 * a[4][0][0]
            if order > 2:
                a[2][0][0] -= 1/20 * (b[3][0][1] + c[3][1][0])

    # 1st order
    if (order > 0):
        for i in range(3):
            a = abc[i]
            Bx = B_moments[i]

            a[1][0][0] = neighbordiff(Bx[0], i)
            if order > 2:
                a[1][0][0] -= 1/10 * a[3][0][0]
    
    # 0th order
    for i in range(3):
        a = abc[i]
        Bx = B_moments[i]

        a[0][0][0] = 1/2 * neighborsum(Bx[0], i)
        if order > 1:
            a[0][0][0] -= 1/6 * a[2][0][0]
        if order > 3:
            a[0][0][0] -= 1/70 * a[4][0][0]

    # Check constraint:
    #test = abc[0][1][0][0] + abc[1][1][0][0] + abc[2][1][0][0] + 1/10 * (abc[0][3][0][0] + abc[1][3][0][0] + abc[2][3][0][0])
    #print(np.amax(test))
    #if abs(test) > 0:
    #    print("Something went wrong, sum (17) is " + str(test))

    # return np.pad(np.transpose(abc, (4, 5, 6, 0, 1, 2, 3)), [(0, 1), (0, 1), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0)])

    return np.pad(abc, [(0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 1), (0, 1)])

def center_value(B_moments, xyz):
    abc = solve_coefficients(B_moments, xyz, order)
    return [interpolate_x(abc[0], 0, 0, 0), interpolate_y(abc[1], 0, 0, 0), interpolate_z(abc[2], 0, 0, 0)]

def ave_B(B_moments, xyz):
    abc = solve_coefficients(B_moments, xyz)
    a = abc[0]
    b = abc[1]
    c = abc[2]
    if order > 3:
        return (
            a[0][0][0] - 3/8 * (a[2][0][0] + a[0][2][0] + a[0][0][2]) + 9/64 * (a[2][2][0] + a[2][0][2]) + 15/128 * a[4][0][0],
            b[0][0][0] - 3/8 * (b[2][0][0] + b[0][2][0] + b[0][0][2]) + 9/64 * (b[2][2][0] + b[2][0][2]) + 15/128 * b[4][0][0],
            c[0][0][0] - 3/8 * (c[2][0][0] + c[0][2][0] + c[0][0][2]) + 9/64 * (c[2][2][0] + c[2][0][2]) + 15/128 * c[4][0][0]
        )
    elif order > 1:
        return (
            a[0][0][0] - 3/8 * (a[2][0][0] + a[0][2][0] + a[0][0][2]) + 9/64 * (a[2][2][0] + a[2][0][2]),
            b[0][0][0] - 3/8 * (b[2][0][0] + b[0][2][0] + b[0][0][2]) + 9/64 * (b[2][2][0] + b[2][0][2]),
            c[0][0][0] - 3/8 * (c[2][0][0] + c[0][2][0] + c[0][0][2]) + 9/64 * (c[2][2][0] + c[2][0][2])
        )
    else:
        return (
            a[0][0][0],
            b[0][0][0],
            c[0][0][0]
        )

def all_ave_Bs(B_moments):
    abc = solve_all_coefficients(B_moments)
    a = abc[0]
    b = abc[1]
    c = abc[2]
    if order > 3:
        return np.transpose(np.array(
            (
                a[0][0][0] - 3/8 * (a[2][0][0] + a[0][2][0] + a[0][0][2]) + 9/64 * (a[2][2][0] + a[2][0][2]) + 15/128 * a[4][0][0],
                b[0][0][0] - 3/8 * (b[2][0][0] + b[0][2][0] + b[0][0][2]) + 9/64 * (b[2][2][0] + b[2][0][2]) + 15/128 * b[4][0][0],
                c[0][0][0] - 3/8 * (c[2][0][0] + c[0][2][0] + c[0][0][2]) + 9/64 * (c[2][2][0] + c[2][0][2]) + 15/128 * c[4][0][0]
            )
        ), [1, 2, 3, 0])
    elif order > 1:
        return np.transpose(np.array(
            (
                a[0][0][0] - 3/8 * (a[2][0][0] + a[0][2][0] + a[0][0][2]) + 9/64 * (a[2][2][0] + a[2][0][2]),
                b[0][0][0] - 3/8 * (b[2][0][0] + b[0][2][0] + b[0][0][2]) + 9/64 * (b[2][2][0] + b[2][0][2]),
                c[0][0][0] - 3/8 * (c[2][0][0] + c[0][2][0] + c[0][0][2]) + 9/64 * (c[2][2][0] + c[2][0][2])
            )
        ), [1, 2, 3, 0])
    else:
        return np.transpose(np.array(
                (
                    a[0][0][0],
                    b[0][0][0],
                    c[0][0][0]
                )
            ), [1, 2, 3, 0])

def center_values(B_moments, coords):
    # Looks scuffed but is faster
    abc = np.transpose(np.transpose(solve_all_coefficients(B_moments), (4, 5, 6, 0, 1, 2, 3))[coords[:, 0], coords[:, 1], coords[:, 2]], (1, 2, 3, 4, 0))
    return np.transpose(np.array([interpolate_x(abc[0], 0, 0, 0), interpolate_y(abc[1], 0, 0, 0), interpolate_z(abc[2], 0, 0, 0)]))
    #return all_center_values(B_moments, order)[coords[:, 0], coords[:, 1], coords[:, 2], :]

def all_center_values(B_moments):
    abc = solve_all_coefficients(B_moments)
    return np.transpose(np.array([interpolate_x(abc[0], 0, 0, 0), interpolate_y(abc[1], 0, 0, 0), interpolate_z(abc[2], 0, 0, 0)]), (1, 2, 3, 0)) 

def all_center_values_dbxdi(B_moments):
    abc = solve_all_coefficients(B_moments)
    return np.transpose(np.array([interpolate_dbxdx(abc[0], 0, 0, 0), interpolate_dbxdy(abc[0], 0, 0, 0), interpolate_dbxdz(abc[0], 0, 0, 0)]), (1, 2, 3, 0))

def all_center_values_dbydi(B_moments):
    abc = solve_all_coefficients(B_moments)
    return np.transpose(np.array([interpolate_dbydx(abc[1], 0, 0, 0), interpolate_dbydy(abc[1], 0, 0, 0), interpolate_dbydz(abc[1], 0, 0, 0)]), (1, 2, 3, 0))

def all_center_values_dbzdi(B_moments):
    abc = solve_all_coefficients(B_moments)
    return np.transpose(np.array([interpolate_dbzdx(abc[2], 0, 0, 0), interpolate_dbzdy(abc[2], 0, 0, 0), interpolate_dbzdz(abc[2], 0, 0, 0)]), (1, 2, 3, 0))

def all_center_values_dbidi(B_moments):
    abc = solve_all_coefficients(B_moments)
    return np.concatenate(
            (np.transpose(np.array([interpolate_dbxdx(abc[0], 0, 0, 0), interpolate_dbxdy(abc[0], 0, 0, 0), interpolate_dbxdz(abc[0], 0, 0, 0)]), (1, 2, 3, 0)),
            np.transpose(np.array([interpolate_dbydx(abc[1], 0, 0, 0), interpolate_dbydy(abc[1], 0, 0, 0), interpolate_dbydz(abc[1], 0, 0, 0)]), (1, 2, 3, 0)),
            np.transpose(np.array([interpolate_dbzdx(abc[2], 0, 0, 0), interpolate_dbzdy(abc[2], 0, 0, 0), interpolate_dbzdz(abc[2], 0, 0, 0)]), (1, 2, 3, 0))),
            axis=3)

