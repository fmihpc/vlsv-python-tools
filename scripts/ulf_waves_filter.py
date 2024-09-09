import os  # nopep8
os.environ['PTNOLATEX'] = '1'  # nopep8
import pytools as pt  # nopep8
import glob
from scipy.signal import butter, filtfilt
import numpy as np
import logging


# fileNames={i : "/wrk-vakka/group/spacephysics/vlasiator/3D/EGI/bulk/dense_cold_hall1e5_afterRestart374/bulk1.{:07d}.vlsv".format(i) for i in range(662,1506)}
# fileNames={i : "/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/bulk1.{:07d}.vlsv".format(i) for i in range(501,1612)}
# fileNames={i :'/wrk-vakka/group/spacephysics/vlasiator/3D/EGL/bulk/bulk1.egl.{:07d}.vlsv'.format(i) for i in range(621,1760)}
# logging.info(fileNames)


def ulf_filter(
    filesDirectory="/wrk-vakka/group/spacephysics/vlasiator/3D/EGL/bulk/",
    var_to_filter="vg_b_vol",
    window_pad=50,
    timestate=1000,
    filter_order=5,
    target_wave="Pc2",
    fs=1,
    outputDirectory="temp/",
    run="EGL",
):
    """Function to compute the moving average of electromagentic fields (or any quantities for that matter), and apply band pass filter

    Args:
        filesDirectory (str): vlsv files directory
        var_to_filter (str, optional): variables (vg_b_vol, vg_e_vol, fg_e (computationally costy), fg_b (computationally costy)). Defaults to "vg_b_vol".
        window_pad (int, optional): moving average window. Defaults to 50 for 100 seconds moving average.
        timestate (int, optional): (start_of_simulation_time + window_pad < timestate < end_of_simulation_time - window_pad). Defaults to 1000.
        filter_order (int, optional): band pass filter order. Defaults to 5.
        target_wave (str, optional): ULF wave target (Pc2, Pc3, Pc4), currently resolved waves in Vlasiator. Defaults to "Pc2".
        fs (int, optional): samapling frequency. Defaults to 1.
        outputDirectory (str, optional): directory where the sidecars will be saved. Defaults to "/proj/kebedefa/temp/EGL/".
    output:
        vlsv file saved in the output directory
    """
    # var_to_filter = "vg_b_vol"
    fileNames = glob.glob(filesDirectory + "*.vlsv")
    fileNames.sort()
    # start of the run (assumes continous simulation data)
    run_start = int(fileNames[0].split('.')[-2])
    f0 = pt.vlsvfile.VlsvReader(fileNames[0])
    cids0 = f0.read_variable("CellID")
    ncells = len(cids0)
    nelems = len(f0.read_variable(var_to_filter, [1]))
    windowlength = 2 * window_pad + 1
    B_all = np.zeros((windowlength, ncells, nelems))
    logging.info("B_all.shape "+ str(B_all.shape))
    # timestate = 1000 # this can take from 713 to 1450 ( for example for EGI)
    average_range = (timestate - window_pad, timestate + window_pad)
    # collecting data for moving_averaging based on the window_pad/average_range
    for i, t in enumerate(range(average_range[0], average_range[1] + 1)):
        logging.info("loading in " + str(((i, t), fileNames[t-run_start])))
        reader = pt.vlsvfile.VlsvReader(fileNames[t-run_start])
        cids = reader.read_variable("CellID")
        sorti = np.argsort(cids)
        var = reader.read_variable(var_to_filter)[sorti, :]
        B_all[i, :, :] = var
    # filter_order =5
    if target_wave == "Pc2":
        lowcut = 0.1
        highcut = 0.45
    elif target_wave == "Pc3":
        lowcut = 0.02
        highcut = 0.1
    elif target_wave == "Pc4":
        lowcut = 0.006
        highcut = 0.02
    else:
        logging.info("no defined band filter")
    b, a = butter(filter_order, [lowcut, highcut], fs=fs, btype="band")
    # b, a = butter(filter_order, 0.05, btype='high', fs=fs)
    filtered_dataPc2 = filtfilt(b, a, B_all, axis=0)
    averaged = np.mean(B_all, axis=0)
    reader = pt.vlsvfile.VlsvReader(fileNames[timestate])

    if not os.path.exists(outputDirectory):
        # If it doesn't exist, create it
        os.makedirs(outputDirectory)

    writer = pt.vlsvfile.VlsvWriter(
        reader,
        "{:s}{:s}_test_movingaverage_{:07d}.vlsv".format(
            outputDirectory, run, timestate
        ),
        copy_meshes="SpatialGrid",
    )
    # Uncomment this if you want to copy all the vlsv varibale, but not needed for now
    # writer.copy_variables_list(reader, reader.get_all_variables())
    sorti = np.argsort(reader.read_variable("CellID"))
    rev_sorti = np.argsort(sorti)
    data_arr = averaged[rev_sorti, :]
    delta_var = reader.read_variable(var_to_filter)
    delta_var = delta_var - data_arr
    # moving average of var
    varinfo = pt.calculations.VariableInfo(
        data_arr,
        "{:s}_move_ave_{:d}".format(var_to_filter, windowlength),
        units="T",
        latex=r"$B_{\mathrm{ave}}$",
        latexunits=r"$\mathrm{T}$",
    )
    writer.write_variable_info(varinfo, "SpatialGrid", unitConversion=1)
    varinfo = pt.calculations.VariableInfo(
        delta_var,
        "{:s}_move_ave_{:d}_delta".format(var_to_filter, windowlength),
        units="T",
        latex=r"$\delta{}B_{\mathrm{ave}}$",
        latexunits=r"$\mathrm{T}$",
    )
    writer.write_variable_info(varinfo, "SpatialGrid", unitConversion=1)
    # window_pad =50 see above this refers to the target file (centered file)
    varinfo = pt.calculations.VariableInfo(
        filtered_dataPc2[window_pad, rev_sorti, :],
        "vg_b_vol_{:s}_{:d}_xyz".format(target_wave, windowlength),
        units="T^2Hz^-1",
        latex=r"$\delta{}B_{\mathrm{ave}}$",
        latexunits=r"$\mathrm{T}^2\,\mathrm{Hz}^{-1}\mathrm{orsomething}$",
    )
    writer.write_variable_info(varinfo, "SpatialGrid", unitConversion=1)
    del reader


if __name__ == "__main__":
    ulf_filter()
