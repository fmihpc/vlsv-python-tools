import sys, os
import numpy as np
from scipy.ndimage import uniform_filter1d
import pytools as pt
import matplotlib.pyplot as plt
import argparse

r_e = 6.371e6

"""
    Script for plotting a cut-through timeseries, aka keogram, aka time-elongation map, etc. for a given variable, a given set of coordinates, and a range of times.

    This script takes 12 parameters, example usage:
    
    python cutthrough_timeseries.py -var <var> -fnr <fnr0> <fnr1> -pointfile <pointfile> -bulkpath <bulkpath> -bulkprefix <bulkprefix> -outputname <outputname> -outputdir <outputdir> -intpol <intpol> -filt <filt> -op <op> -cmap <cmap>

    
    Parameter descriptions:
        var: Variable to plot (e.g. "proton/vg_rho", "vg_b_vol")
        fnr0: First file number to plot
        fnr1: Last file number to plot
        pointfile: Path to text file containing coordinates. Each row should correspond to one point, having 3 columns containing coordinates (X,Y,Z), in units of RE
        bulkpath: Path to directory containing bulk files
        bulkprefix: Starting string of bulk file name (e.g. bulk, bulk1, bulk5)
        outputname: Name of output file
        outputdir: Path to output directory
        intpol: Interpolate to cut-through sample points? (True/False), default=False
        filt: Filter out slowly changing signal? (<=0: no filtering, >0: filter with specified window size), default=-1
        op: Variable operator, default="pass"
        cmap: Colormap, default="viridis"
"""


def jplots(
    var,
    fnr0,
    fnr1,
    bulkpath,
    bulkprefix,
    outputname,
    outputdir,
    intpol=False,
    filt=-1,
    op="pass",
    cmap="viridis",
    pointfile=None,
):

    fnr_arr = np.arange(fnr0, fnr1 + 0.1, 1, dtype=int)
    t_arr = np.zeros_like(fnr_arr).astype(float)

    if bulkpath[-1] != "/":
        bulkpath += "/"

    coords = np.loadtxt(pointfile) * r_e
    point_list = np.arange(len(coords))
    if not intpol:
        fobj = pt.vlsvfile.VlsvReader(
            bulkpath + bulkprefix + ".{}.vlsv".format(str(fnr0).zfill(7))
        )
        cellids = [int(fobj.get_cellid(coord)) for coord in coords]

    data_arr = np.zeros((fnr_arr.size, point_list.size), dtype=float)

    for idx in range(fnr_arr.size):
        fnr = fnr_arr[idx]
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + bulkprefix + ".{}.vlsv".format(str(fnr).zfill(7))
        )
        t_arr[idx] = vlsvobj.read_parameter("time")
        if intpol:
            data_arr[idx, :] = [
                vlsvobj.read_interpolated_variable(var, coord, operator=op)
                for coord in coords
            ]
        else:
            data_arr[idx, :] = vlsvobj.read_variable(var, operator=op, cellids=cellids)

    if filt > 0:
        data_arr = data_arr - uniform_filter1d(data_arr, size=filt, axis=0)

    XmeshXY, YmeshXY = np.meshgrid(point_list, t_arr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 12), constrained_layout=True)

    im = ax.pcolormesh(
        XmeshXY,
        YmeshXY,
        data_arr,
        shading="gouraud",
        cmap=cmap,
        rasterized=True,
    )

    ax.set_xlim(point_list[0], point_list[-1])
    ax.set_ylim(t_arr[0], t_arr[-1])
    ax.set_xlabel("Point along cut", labelpad=10, fontsize=16)
    ax.set_ylabel("Time [s]", labelpad=10, fontsize=16)
    ax.set_title(var, pad=10, fontsize=16)

    cb = fig.colorbar(im, ax=ax)

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    if outputdir[-1] != "/":
        outputdir += "/"

    fig.savefig(outputdir + outputname, dpi=300)
    plt.close(fig)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-var", help="Variable to plot (e.g. proton/vg_rho)", type=str, required=True
)
parser.add_argument("-op", help="Operator for variable", type=str, default="pass")
parser.add_argument(
    "-fnr", nargs=2, help="First and last file number to plot", type=int, required=True
)
parser.add_argument(
    "-pointfile",
    help="A file with coordinates defining the curve to plot along",
    type=str,
)
parser.add_argument(
    "-bulkpath", help="Path to directory with bulk files", type=str, required=True
)
parser.add_argument(
    "-bulkprefix",
    help="Starting string of bulk file name (e.g. bulk, bulk1,bulk5)",
    type=str,
    required=True,
)
parser.add_argument("-outputdir", help="Output file directory", type=str)
parser.add_argument("-outputname", help="Output file name", type=str)
parser.add_argument(
    "-intpol",
    help="Interpolate to cut-through sample points? (True/False)",
    type=bool,
    default=False,
)
parser.add_argument(
    "-filt",
    help="Filter out slowly changing signal? (<=0: no filtering, >0: filter with specified window size)",
    type=int,
    default=-1,
)
parser.add_argument(
    "-cmap", help="Colormap to use for plot", type=str, default="viridis"
)
args = parser.parse_args()

if __name__ == "__main__":

    jplots(
        var=args.var,
        fnr0=args.fnr[0],
        fnr1=args.fnr[1],
        bulkpath=args.bulkpath,
        bulkprefix=args.bulkprefix,
        outputdir=args.outputdir,
        outputname=args.outputname,
        intpol=args.intpol,
        filt=args.filt,
        op=args.op,
        cmap=args.cmap,
        pointfile=args.pointfile,
    )
