# 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2024 University of Helsinki
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

''' This is a module that includes most, if not all, calculations within the analysator. Please type "." and press tab to see which functions this module includes.

Example:
import analysator as pt

pt.calculations.
#press tab

#Output: list of functions

pt.calculations.fourier?
#Press Enter

'''

# List of functions and classes that should be imported into the interface
import logging
from intpol_file import vlsv_intpol_file
from intpol_points import vlsv_intpol_points
from cutthrough import cut_through, cut_through_step, cut_through_curve, cut_through_swath
from fourier import fourier
from spectra import get_spectrum_energy, get_spectrum_alongaxis_vel
from variable import VariableInfo
from timeevolution import cell_time_evolution
from pitchangle import pitch_angles
#from backstream import extract_velocity_cells_sphere, extract_velocity_cells_non_sphere
from gyrophaseangle import gyrophase_angles_from_file
from themis_observation import themis_observation_from_file
from themis_observation import themis_plot_detector
from themis_observation import themis_plot_phasespace_contour
from themis_observation import themis_plot_phasespace_helistyle
#from themis_observation import simulation_to_spacecraft_frame
from cut3d import cut3d
from lineout import lineout
import fit
from fieldtracer import static_field_tracer, static_field_tracer_3d
from fieldtracer import dynamic_field_tracer
from non_maxwellianity import epsilon_M
from null_lines import LMN_null_lines_FOTE
from interpolator_amr import AMRInterpolator, supported_amr_interpolators
