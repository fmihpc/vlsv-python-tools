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

# This file has a class "Variable" that holds all the important data for variables e.g. variable's name, the units and the data on the variable
import numpy as np
from plot import cbfmtsci
from numbers import Number
import logging

class VariableInfo:
   ''' A class/struct for holding variable info. This includes the variable data in array form, the name and the units.
       Variable info is in: data, name, units.

       .. code-block:: python

          LIST VARIABLES IN VARIBLE INFO:
          data              Data of the variable (array list)
          name              Name of the variable
          units             Units of the variable
          latex             Name of the variable in LaTeX
          latexunits        Units of the variable in LaTeX
   '''
   def __init__(self, data_array, name="", units="", latex="", latexunits=""):
      ''' Initializes variable info.

          :param data_array:         Raw data for the variable (an array for example)
          :param name:               Name of the variable
          :param units:              Name of the variable's units
          :param latex:              Name of the variable in LaTeX
          :param latexunits:         Name of the variable's units in LaTeX
      '''
      self.data = np.ma.asarray(data_array)
      self.name = name
      self.units = units
      self.latex = latex
      self.latexunits = latexunits
      self.scaleDict = {}

   def __repr__(self):
      return "VariableInfo(Name: \'" + str(self.name) + "\' Units: \'" + str(self.units) + "\')"

   def __str__(self):
      return self.__repr__()

   def get_variable(self, index ):
      ''' Creates a new variable with identical data but only certain index is included

          :param index:         Vector index
          :returns: an edited version of the variable

          .. note::

             E.g. if the data is an array of 3d vectors, get_variable(0) would return the variable with data[:,0] as the data
      '''
      if len(self.data) <= 0:
         logging.info("BAD DATA LENGTH")
         return []
      if len(np.atleast_1d(self.data[0])) <= index:
         logging.info("BAD INDEX, THE INDEX IS LARGER THAN VECTOR SIZE!")
         return []
      return VariableInfo( self.data[:,index], self.name, self.units, self.latex, self.latexunits )


   def get_scaled_units(self, vscale=None, env='EarthSpace', manualDict=None):
      ''' Return scaling metadata

          :param env:          A string to choose the scaling dictionary [default: EarthSpace]
          :param manualDict:   a dictionary of {units : {scalingparams}}; used to update the included dictionary
          :param vscale:       float, factor to scale the variable with
          :returns: (norming factor, scaledUnits, scaledLatexUnits)

      '''

     #
      if env=='EarthSpace':
         self.scaleDict = {
                 's'      : {'defaultScale':1,
                             1e6: {'scaledUnits':'us', 'scaledLatexUnit':r'$\mu\mathrm{s}$'},
                             1e3: {'scaledUnits':'ms', 'scaledLatexUnit':r'$\mathrm{ms}$'}
                            },
                 'T'      : {'defaultScale':1e9,
                             1e9: {'scaledUnits':'nT', 'scaledLatexUnit':r'$\mathrm{nT}$'}
                            },
                 'K'      : {'defaultScale':1e-6,
                             1e-6:{'scaledUnits':'MK', 'scaledLatexUnit':r'$\mathrm{MK}$'}
                            },
                 'Pa'     : {'defaultScale':1e9,
                             1e9:{'scaledUnits':'nPa', 'scaledLatexUnit':r'$\mathrm{nPa}$'}
                            },
                 '1/m'  :   {'defaultScale':1e3,
                             1e3:{'scaledUnits':'1/km', 'scaledLatexUnit':r'$\mathrm{km}^{-1}$'}
                            },
                 '1/m^3'  : {'defaultScale':1e-6,
                             1e-6:{'scaledUnits':'1/cm^3', 'scaledLatexUnit':r'$\mathrm{cm}^{-3}$'}
                            },
                 '1/m3'   : {'defaultScale':1e-6,
                             1e-6:{'scaledUnits':'1/cm^3', 'scaledLatexUnit':r'$\mathrm{cm}^{-3}$'}
                            },
                 'm/s'    : {'defaultScale':1e-3,
                             1e-3:{'scaledUnits':'km/s', 'scaledLatexUnit':r'$\mathrm{km}\,\mathrm{s}^{-1}$'}
                            },
                 'V/m'    : {'defaultScale':1e3,
                             1e3:{'scaledUnits':'mV/m', 'scaledLatexUnit':r'$\mathrm{mV}\,\mathrm{m}^{-1}$'}
                            },
                 'eV/cm3' : {'defaultScale':1e-3,
                             1e-3:{'scaledUnits':'keV/cm^3', 'scaledLatexUnit':r'$\mathrm{keV}\,\mathrm{cm}^{-3}$'}
                            },
                 'eV/cm^3': {'defaultScale':1e-3,
                             1e-3:{'scaledUnits':'keV/cm^3', 'scaledLatexUnit':r'$\mathrm{keV}\,\mathrm{cm}^{-3}$'}
                            },
                 'T/m': {'defaultScale':1e-12,
                             1e-9:{'scaledUnits':'nT/m', 'scaledLatexUnit':r'$\mathrm{nT}\,\mathrm{m}^{-1}$'},
                             1e-12:{'scaledUnits':'nT/km', 'scaledLatexUnit':r'$\mathrm{nT}\,\mathrm{km}^{-1}$'}
                           },
                 'kg/m3'  : {'defaultScale':5.97863741e26,
                             5.97863741e26:{'scaledUnits':'amu/m^3', 'scaledLatexUnit':r'$\mathrm{amu}\,\mathrm{m}^{-3}$'},
                             5.97863741e20:{'scaledUnits':'amu/cm^3', 'scaledLatexUnit':r'$\mathrm{amu}\,\mathrm{cm}^{-3}$'}
                            },
                 'A/m^2' :   {'defaultScale':1,
                             1e-3:{'scaledUnits':'mA/m^2', 'scaledLatexUnit':r'$\mathrm{mA}\,\mathrm{m}^{-2}$'}
                            },
                 'W/m^2' :   {'defaultScale':1,
                             1e-3:{'scaledUnits':'mW/m^2', 'scaledLatexUnit':r'$\mathrm{mW}\,\mathrm{m}^{-2}$'}
                            },
                 'm^2' :   {'defaultScale':1,
                             1e-4:{'scaledUnits':'km^2', 'scaledLatexUnit':r'$\mathrm{km}^{2}$'}
                            },
                 'V' :   {'defaultScale':1,
                             1e-3:{'scaledUnits':'mV', 'scaledLatexUnit':r'$\mathrm{mV}$'}
                            },
                 'Degrees' :   {'defaultScale':1,
                            },
                 'mho' :   {'defaultScale':1,
                            },
         }

      else:
         self.scaleDict = {}
      if manualDict is not None:
         self.scaleDict.update(manualDict)
      unitScale = 1.0
      scaledUnits = self.units
      scaledLatexUnits = self.latexunits

      if self.units != '':
         dictKey = self.units
         try:
            udict = self.scaleDict[dictKey]
         except:
            if vscale is None:
               return 1.0, self.units, self.latexunits
            else:
               return vscale, self.units, self.latexunits
         if vscale is None:
            try:
               unitScale = udict['defaultScale']
            except:
               return 1.0, self.units, self.latexunits
         elif np.isclose(vscale, 1.0):
               return 1.0, self.units, self.latexunits
         else:
             unitScale = vscale

         if not any([np.isclose(unitScale, tryScale) for tryScale in udict.keys() if isinstance(tryScale, Number)]):
             #
             return vscale, self.units+" x{vscale:e}".format(vscale=vscale), self.latexunits+r"{\times}"+cbfmtsci(vscale,None)
         try:
            #above guarantees the list comprehension does not give an empty list
            unitScale = [scale for scale in udict.keys() if isinstance(scale, Number) and np.isclose(scale,unitScale)][0]
            scaledUnits = udict[unitScale]['scaledUnits']
         except KeyError:
            # logging.info('Missing scaledUnits in specialist dict for' + self.units + ' for unitScale='+str(unitScale))
            return 1.0, self.units, self.latexunits
         try:
            scaledLatexUnits = udict[unitScale]['scaledLatexUnit']
         except:
            # logging.info('Missing scaledLatexUnits in specialist dict for ' + self.units+ ' for unitScale='+str(unitScale))
            return 1.0, self.units, self.latexunits
      else:
          if vscale is None or np.isclose(vscale, 1.0):
            return 1.0, self.units, self.latexunits
          else:
            return vscale, self.units+"x{vscale:e}".format(vscale=vscale), self.latexunits+r"{\times}"+cbfmtsci(vscale,None)

      return unitScale, scaledUnits, scaledLatexUnits

   # A utility to get variableinfo with corresponding units for simple plotting. Add "canonical" scalings as
   # necessary, for default/other environments.
   def get_scaled_var(self, vscale=None, data=None, env='EarthSpace', manualDict=None):
      ''' Automatically scales the variableinfo data and adjusts the units correspondingly with the
          default dictionaries.

          :param data:         in case you wish to provide new data array (why, though?)
          :param env:          A string to choose the scaling dictionary [default: EarthSpace]
          :param manualDict:   a dictionary of {units : {scalingparams}}; used to update the included dictionary
          :returns: self, with scaled units with pre-formatted units included in the varinfo.


      '''

      if data is None:
         data = self.data
      else:
         self.data = data

      unitScale, scaledUnits, scaledLatexUnits = self.get_scaled_units(vscale=vscale, env=env, manualDict=manualDict)
      if unitScale == 1: # no change, optimize out the calculation
          return self

      self.data = data*unitScale
      self.units = scaledUnits
      self.latexunits = scaledLatexUnits
      return self



def get_data( variable ):
   ''' Function to use when not sure if variable is in raw form ( simply a list with data ), or a VariableInfo instance

      :param variable:           The variable as a VariableInfo instance or a list
      :returns: data of the variable
   '''
   if isinstance(variable, VariableInfo):
      return variable.data
   else:
      return variable

def get_name( variable ):
   ''' Function to use when not sure if variable is in raw form ( simply a list with data ), or a VariableInfo instance

      :param variable:           The variable as a VariableInfo instance or a list
      :returns: the name of the variable or \"\" if not a VariableInfo instance
   '''
   if isinstance(variable, VariableInfo):
      return variable.name
   else:
      return ""

def get_units( variable ):
   ''' Function to use when not sure if variable is in raw form ( simply a list with data ), or a VariableInfo instance

      :param variable:           The variable as a VariableInfo instance or a list
      :returns: the units of the variable or \"\" if not a VariableInfo instance
   '''
   if isinstance(variable, VariableInfo):
      return variable.units
   else:
      return ""

def get_latex( variable ):
   ''' Function to use when not sure if variable is in raw form ( simply a list with data ), or a VariableInfo instance

      :param variable:           The variable as a VariableInfo instance or a list
      :returns: the variable in LaTeX format or \"\" if not a VariableInfo instance
   '''
   if isinstance(variable, VariableInfo):
      return variable.latex
   else:
      return ""

def get_latexunits( variable ):
   ''' Function to use when not sure if variable is in raw form ( simply a list with data ), or a VariableInfo instance

      :param variable:           The variable as a VariableInfo instance or a list
      :returns: the units of the variable in LaTeX format or \"\" if not a VariableInfo instance
   '''
   if isinstance(variable, VariableInfo):
      return variable.latexunits
   else:
      return ""

