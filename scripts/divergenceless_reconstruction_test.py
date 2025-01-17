#!/usr/bin/env python
from divergenceless_reconstruction import solve_moments_from_B, center_value, center_values, all_center_values, all_ave_Bs
import pytools as pt
import numpy as np
from time import perf_counter
from sys import argv
# Code for testing divergenceless_resconstruction.py

def main():
    if len(argv) < 2:
        print("Usage: ./divergenceless_reconstruction_test.py FILENAME")
        return(1)

    filename = argv[1]
    f = pt.vlsvfile.VlsvReader(filename)
    fg_b = f.read_fsgrid_variable('fg_b')
    print(np.shape(fg_b))
    fg_b_vol = f.read_fsgrid_variable('fg_b_vol')
    print('File read!')
    t = perf_counter()
    B_moments = solve_moments_from_B(fg_b[0:100, 0:100, 0:100])
    print(f'B_moments solved in {perf_counter() - t} seconds!')
    print()
    print(f'Volumetric b in (50, 50, 50): {fg_b_vol[50, 50, 50]}')
    print()
    for i in range(5):
        t = perf_counter()
        print(center_value(B_moments, (50, 50, 50), i))
        print(f'Single coefficients up to order {i} solved in {perf_counter() - t} seconds!')

        t = perf_counter()
        print(np.shape(center_values(B_moments, np.array([(50, 50, 50), (51, 51, 51), (54, 54, 64), (43, 65, 42), (43, 65, 32)]), i)))
        print(f'Multiple coefficients up to order {i} solved in {perf_counter() - t} seconds!')

        t = perf_counter()
        print(all_center_values(B_moments, i)[50, 50, 50])
        print(f'All coefficients up to order {i} solved in {perf_counter() - t} seconds!')

        print(f'Average scaled difference between Analysator and Vlasiator in order {i}:')
        print(np.average(((all_ave_Bs(B_moments, i))[:-1, :-1, :-1] - fg_b_vol[:99, :99, :99])/fg_b_vol[:99,:99,:99]))
        print(f'Max scaled difference between Analysator and Vlasiator in order {i}:')
        print(np.max(abs(((all_ave_Bs(B_moments, i))[:-1, :-1, :-1] - fg_b_vol[:99, :99, :99])/fg_b_vol[:99,:99,:99])))
        print()

    return 0

if __name__ == "__main__":
    ret = main()
    if(ret):
        exit(ret)
