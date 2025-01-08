# Installation:
#################################################
#################################################

# Install dependencies:
sudo apt-get install python-matplotlib python-numpy python-scipy ipython

# Set pythonpath environment variable:
echo "export PYTHONPATH=$PYTHONPATH:$HOME/analysator" >> $HOME/.bashrc

# Using Analysator:
#################################################
ipython                  # Start ipython
import analysator as pt  # Import Analysator


# Navigating functions:
pt.calculations.pitch_angles? #press [Enter]
pt.vlsvfile.VlsvReader? #press [Enter]
pt.plot.plot_colormap? #press [Enter]
pt.plot.plot_vdf? #press [Enter]


# For non-interactive mode (also when no X is available):
# set the environment variable PTNONINTERACTIVE to any value before launching
# python/ipython. If in interactive mode, experimental non-blocking
# windows via matplotlib.pyplot.ion() are in use.
#################################################
export PTNONINTERACTIVE=1

# For selecting a backend manually (if Agg or TkAgg is not available)
#################################################
export PTBACKEND=Qt5Agg

# For disabling full LaTeX formatted output (if texlive is not installed)
#################################################
export PTNOLATEX=1

#Analysator is using logging. It is controlled by setting the enviroment variable ANALYSATOR_LOG_LEVEL
#################################################
Supported: DEBUG, INFO (default), WARNING, ERROR, CRITICAL

For example, disable INFO prints via:
`export ANALYSATOR_LOG_LEVEL='WARNING'`

# MayaVi2 support (deprecated)
#################################################
Some old plotting tools requiring the MayaVi2 visualization library can be activated
by setting the environment variable:
export PTMAYAVI2=1

This naturally depends on mayavi, so:
sudo apt-get install mayavi2

# For setting the default output directory (default: $HOME/Plots)
#################################################
export PTOUTPUTDIR=/proj/USERNAME/Plots/

# If using a jupyter notebook on the UH hub system, you can activate interactive plot windows with
# %matplotlib ipympl or %matplotlib notebook before importing pytools

# Examples and instructions for batch scripting (on CSC's system) are found in
# examples/generate_panel.py and examples/generate_movie.sh

# For more information visit the link:
#################################################
https://github.com/fmihpc/analysator/wiki

# Analysator documentation on the link:
#################################################
https://fmihpc.github.io/analysator/

# For citations, use the DOI or the ready-made button the the right in GitHub!
#################################################
https://doi.org/10.5281/zenodo.4462514

