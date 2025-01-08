# 
# This file is part of Analysator.
# Copyright 2013-2016 Finnish Meteorological Institute
# Copyright 2017-2018 University of Helsinki
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

activepopulation='proton' # default

speciesdict ={
    'avgs': 'p',
    'proton': 'p',
    'helium': 'He',
    'oxygen': 'O',
    'electron': 'e',
}
speciesamu ={
    'avgs': 1,
    'proton': 1,
    'helium': 4,
    'oxygen': 16,
#    'electron': 5.4461702e-4, # true electron mass
    'electron': 5.4461702e-3, # electron mass x10 used in Tempo
}
speciescharge ={
    'avgs': 1,
    'proton': 1,
    'helium': 2,
    'oxygen': 1,
    'electron': -1,
}
speciesprecipitationenergybins ={ # values updated when opening the vlsvreader object
    'avgs': -1,
    'proton': -1,
    'helium': -1,
    'oxygen': -1,
    'electron': -1,
}

# Define some units for intrinsic values
unitsdict = {
    'rhom': 'kg/m3',
    'rhoq': 'C/m3',
    'rho': '1/m3',
    'rhobackstream': '1/m3',
    'rhononbackstream': '1/m3',
    'rho_v': '1/m2s',
    'rhovbackstream': '1/m2s',
    'rhovnonbackstream': '1/m2s',
    'v': 'm/s',
    'vbackstream': 'm/s',
    'nNonbackstream': 'm/s',
    'b': 'T',
    'b_vol': 'T',
    'background_b': 'T',
    'perturbed_b': 'T',
    'bgb': 'T',
    'perb': 'T',
    'perb_vol': 'T',
    'e': 'V/m',
    'e_vol': 'V/m',
    'exhall_000_100': 'V/m',
    'exhall_001_101': 'V/m',
    'exhall_010_110': 'V/m',
    'exhall_011_111': 'V/m',
    'eyhall_000_010': 'V/m',
    'eyhall_001_011': 'V/m',
    'eyhall_100_110': 'V/m',
    'eyhall_101_111': 'V/m',
    'ezhall_000_001': 'V/m',
    'ezhall_010_011': 'V/m',
    'ezhall_100_101': 'V/m',
    'ezhall_110_111': 'V/m',
    'pressure': 'Pa',
    'pressure_dt2': 'Pa',
    'pressure_r': 'Pa',
    'pressure_v': 'Pa',
    'ptensordiagonal': 'Pa',
    'ptensoroffdiagonal': 'Pa',
    'ptensorbackstreamdiagonal': 'Pa',
    'ptensorbackstreamoffdiagonal': 'Pa',
    'ptensornonbackstreamdiagonal': 'Pa',
    'ptensornonbackstreamoffdiagonal': 'Pa',
    'max_v_dt': 's',
    'max_r_dt': 's',
    'max_fields_dt': 's',
    'minvalue': 's3/m6',
    'effectivesparsitythreshold': 's3/m6',
    'rho_loss_adjust': '1/m3',
    'energydensity': 'eV/cm3',
    'precipitationdiffflux': '1/(cm2 sr s eV)',
    'precipitationintegralenergyflux': 'keV/(cm2 sr s)',
    'precipitationmeanenergy': 'keV'
    }

# Define some LaTeX markup names for intrinsic values
latexdict = {
    'rhom': r'$\rho_m$',
    'rhoq': r'$\rho_q$',
    'rho': r'$n_\mathrm{p}$',
    'rhobackstream': r'$n_\mathrm{p,st}$',
    'rhononbackstream': r'$n_\mathrm{p,th}$',
    'rho_v': r'$\Gamma_\mathrm{p}$',
    'rhovbackstream': r'$\Gamma_\mathrm{p,st}$',
    'rhovnonbackstream': r'$\Gamma_\mathrm{p,th}$',
    'v': r'$V$',
    'vbackstream': r'$V_\mathrm{p,st}$',
    'vnonbackstream': r'$V_\mathrm{p,th}$',
    'b': r'$B$',
    'b_vol': r'$B_\mathrm{vol}$',
    'background_b': r'$B_\mathrm{bg}$',
    'perturbed_b': r'$B_\mathrm{pert}$',
    'bgb': r'$B_\mathrm{bg}$',
    'perb': r'B_\mathrm{pert}$',
    'perb_vol': r'B_\mathrm{vol,pert}$',
    'e': r'$E$',
    'e_vol': r'$E_\mathrm{vol}$',
    'exhall_000_100': r'$E_\mathrm{Hall,000,100}$',
    'exhall_001_101': r'$E_\mathrm{Hall,001,101}$',
    'exhall_010_110': r'$E_\mathrm{Hall,010,110}$',
    'exhall_011_111': r'$E_\mathrm{Hall,011,111}$',
    'eyhall_000_010': r'$E_\mathrm{Hall,000,010}$',
    'eyhall_001_011': r'$E_\mathrm{Hall,001,011}$',
    'eyhall_100_110': r'$E_\mathrm{Hall,100,110}$',
    'eyhall_101_111': r'$E_\mathrm{Hall,101,111}$',
    'ezhall_000_001': r'$E_\mathrm{Hall,000,001}$',
    'ezhall_010_011': r'$E_\mathrm{Hall,010,011}$',
    'ezhall_100_101': r'$E_\mathrm{Hall,100,101}$',
    'ezhall_110_111': r'$E_\mathrm{Hall,110,111}$',
    'pressure': r'$P$',
    'pressure_dt2': r'$P_{\mathrm{d}t/2}}$',
    'pressure_r': r'$P_r$',
    'pressure_v': r'$P_v$',
    'ptensordiagonal': r'$\mathcal{P}_\mathrm{diag}$',
    'ptensoroffdiagonal': r'$\mathcal{P}_\mathrm{off-diag}$',
    'ptensorbackstreamdiagonal': r'$\mathcal{P}_\mathrm{st,diag}$',
    'ptensorbackstreamoffdiagonal': r'$\mathcal{P}_\mathrm{st,off-diag}$',
    'ptensornonbackstreamdiagonal': r'$\mathcal{P}_\mathrm{th,diag}$',
    'ptensornonbackstreamoffdiagonal': r'$\mathcal{P}_\mathrm{th,off-diag}$',
    'max_v_dt': r'$\Delta t_{\mathrm{max},v}$',
    'max_r_dt': r'$\Delta t_{\mathrm{max},r}$',
    'max_fields_dt': r'$\Delta t_\mathrm{max,FS}$',
    'minvalue': r'$f_\mathrm{Min}$',
    'effectivesparsitythreshold': r'$f_\mathrm{Min}$',
    'rho_loss_adjust': r'$\Delta_\mathrm{loss} n_\mathrm{p}$',
    }

# Define some LaTeX markup names for intrinsic values
latexdictmultipop = {
    'rho': r'$n_\mathrm{REPLACEPOP}$',
    'rhobackstream': r'$n_\mathrm{REPLACEPOP,st}$',
    'rhononbackstream': r'$n_\mathrm{REPLACEPOP,th}$',
    'rho_v': r'$\Gamma_\mathrm{REPLACEPOP}$',
    'rhovbackstream': r'$\Gamma_\mathrm{REPLACEPOP,st}$',
    'rhovnonbackstream': r'$\Gamma_\mathrm{REPLACEPOP,th}$',
    'v': r'$V_\mathrm{REPLACEPOP}$',
    'vbackstream': r'$V_\mathrm{REPLACEPOP,st}$',
    'vnonbackstream': r'$V_\mathrm{REPLACEPOP,th}$',
    'pressure': r'$P_\mathrm{REPLACEPOP}$',
    'pressure_dt2': r'$P_{\mathrm{REPLACEPOP},\mathrm{d}t/2}}$',
    'pressure_r': r'$P_{\mathrm{REPLACEPOP},r}$',
    'pressure_v': r'$P_{\mathrm{REPLACEPOP},v}$',
    'ptensordiagonal': r'$\mathcal{P}_\mathrm{REPLACEPOP,diag}$',
    'ptensoroffdiagonal': r'$\mathcal{P}_\mathrm{REPLACEPOP,off-diag}$',
    'ptensorbackstreamdiagonal': r'$\mathcal{P}_\mathrm{REPLACEPOP,st,diag}$',
    'ptensorbackstreamoffdiagonal': r'$\mathcal{P}_\mathrm{REPLACEPOP,st,off-diag}$',
    'ptensornonbackstreamdiagonal': r'$\mathcal{P}_\mathrm{REPLACEPOP,th,diag}$',
    'ptensornonbackstreamoffdiagonal': r'$\mathcal{P}_\mathrm{REPLACEPOP,th,off-diag}$',
    'minvalue': r'$f_\mathrm{REPLACEPOP,Min}}$',
    'effectivesparsitythreshold': r'$f_\mathrm{REPLACEPOP,Min}$',
    'energydensity': r'$U_\mathrm{REPLACEPOP}$',
    'precipitationdiffflux': r'$\mathcal{F}_{\mathrm{prec},\mathrm{REPLACEPOP}}$',
    'precipitationintegralenergyflux': r'$\int \mathcal{F}_{\mathrm{prec},\mathrm{REPLACEPOP}}$',
    'precipitationmeanenergy': r'$<E_{\mathrm{prec},\mathrm{REPLACEPOP}}>$'
    }

# Define some LaTeX markup units for intrinsic values
latexunitsdict = {
    'rhom': r'$\mathrm{kg}\,\mathrm{m}^{-3}$',
    'rhoq': r'$\mathrm{C}\,\mathrm{m}^{-3}$',
    'rho': r'$\mathrm{m}^{-3}$',
    'rhobackstream': r'$\mathrm{m}^{-3}$',
    'rhononbackstream': r'$\mathrm{m}^{-3}$',
    'rho_v': r'$\mathrm{m}^{-2}$s',
    'rhovbackstream': r'$\mathrm{m}^{-2}$s',
    'rhovnonbackstream': r'$\mathrm{m}^{-2}$s',
    'v': r'$\mathrm{m}\,\mathrm{s}^{-1}$',
    'vbackstream': r'$\mathrm{m}\,\mathrm{s}^{-1}$',
    'vnonbackstream': r'$\mathrm{m}\,\mathrm{s}^{-1}$',
    'b': r'T',
    'b_vol': r'T',
    'background_b': r'T',
    'perturbed_b': r'T',
    'bgb': r'T',
    'perb': r'T',
    'perb_vol': r'T',
    'e': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'e_vol': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'exhall_000_100': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'exhall_001_101': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'exhall_010_110': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'exhall_011_111': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'eyhall_000_010': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'eyhall_001_011': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'eyhall_100_110': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'eyhall_101_111': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'ezhall_000_001': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'ezhall_010_011': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'ezhall_100_101': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'ezhall_110_111': r'$\mathrm{V}\,\mathrm{m}^{-1}$',
    'pressure': r'Pa',
    'pressure_dt2': r'Pa',
    'pressure_r': r'Pa',
    'pressure_v': r'Pa',
    'ptensordiagonal': r'Pa',
    'ptensoroffdiagonal': r'Pa',
    'ptensorbackstreamdiagonal': r'Pa',
    'ptensorbackstreamoffdiagonal': r'Pa',
    'ptensornonbackstreamdiagonal': r'Pa',
    'ptensornonbackstreamoffdiagonal': r'Pa',
    'max_v_dt': r's',
    'max_r_dt': r's',
    'max_fields_dt': r's',
    'minvalue': r'$\mathrm{m}^{-6}\,\mathrm{s}^{3}$',
    'effectivesparsitythreshold': r'$\mathrm{m}^{-6}\,\mathrm{s}^{3}$',
    'rho_loss_adjust': r'$\mathrm{m}^{-3}$',
    'energydensity': r'$\mathrm{eV}\,\mathrm{cm}^{-3}$',
    'precipitationdiffflux': r'$\mathrm{cm}^{-2} \,\mathrm{sr}^{-1}\,\mathrm{s}^{-1}\,\mathrm{eV}^{-1}$',
    'precipitationintegralenergyflux': r'$\mathrm{keV} \, \mathrm{cm}^{-2} \, \mathrm{sr}^{-1} \, \mathrm{s}^{-1}$',
    'precipitationmeanenergy': r'keV'
    }

