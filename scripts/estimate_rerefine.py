import analysator as pt
import numpy as np
import scipy
import logging

def classify_alpha(exprmaps, requestvariables=False):
    if requestvariables==True:
        return ['3d', 'vg_amr_alpha', 'proton/vg_blocks', 'vg_amr_reflevel', 'vg_x', 'vg_y', 'vg_z', 'vg_dx']
    alpha = exprmaps['vg_amr_alpha']
    blocks = exprmaps['proton/vg_blocks']
    reflevel = exprmaps['vg_amr_reflevel']
    x = exprmaps['vg_x']
    y = exprmaps['vg_y']
    z = exprmaps['vg_z']
    dx = np.amax(exprmaps['vg_dx'])
    data_class = np.copy(reflevel)
    percs = refineTresh * refineMult ** np.arange(maxRefLevel)
    for i in np.arange(len(percs)):
        data_class[np.logical_and(np.logical_and(alpha > percs[i], data_class < maxRefLevel), x**2 + y**2 + z**2 > (3.81E7 + 2*dx)**2)] += 1
    # Ionosphere and borders
    #data_class[x**2 + y**2 + z**2 < (3.81E7 + 2*dx)**2] = 2
    for i in np.arange(maxRefLevel):
        margin = 2 * (2 + i)
        data_class[np.logical_and(x < xmin + margin*dx, data_class > i)] = i
        data_class[np.logical_and(x > xmax - margin*dx, data_class > i)] = i
        data_class[np.logical_and(y < ymin + margin*dy, data_class > i)] = i
        data_class[np.logical_and(y > ymax - margin*dy, data_class > i)] = i
        data_class[np.logical_and(z < zmin + margin*dz, data_class > i)] = i
        data_class[np.logical_and(z > zmax - margin*dz, data_class > i)] = i
    return data_class

# Filename here
filename = '/wrk/users/lkotipal/REFINEMENT-TEST/bulk.-2147483647.vlsv'
f = pt.vlsvfile.VlsvReader(filename)
# Change these to match your run config
# If you want to use classify_alpha by itself you'll need these as well
maxRefLevel = 3
refineTresh = 1.0
refineMult = 2.0

alpha = f.read_variable('vg_amr_alpha')
blocks = f.read_variable('proton/vg_blocks')
reflevel = f.read_variable('vg_amr_reflevel')
x = f.read_variable('vg_x')
y = f.read_variable('vg_y')
z = f.read_variable('vg_z')

xmin = f.read_parameter('xmin')
xmax = f.read_parameter('xmax')
ymin = f.read_parameter('ymin')
ymax = f.read_parameter('ymax')
zmin = f.read_parameter('zmin')
zmax = f.read_parameter('zmax')
dx = np.amax(f.read_variable('vg_dx'))
dy = np.amax(f.read_variable('vg_dy'))
dz = np.amax(f.read_variable('vg_dz'))

data_class = np.copy(reflevel)
percs = refineTresh * refineMult ** np.arange(maxRefLevel)
for i in range(len(percs)):
    data_class[np.logical_and(np.logical_and(alpha > percs[i], data_class < maxRefLevel), x**2 + y**2 + z**2 > (3.81E7 + 2*dx)**2)] += 1
for i in np.arange(maxRefLevel):
    margin = 2 * (2 + i)
    data_class[np.logical_and(x < xmin + margin*dx, data_class > i)] = i
    data_class[np.logical_and(x > xmax - margin*dx, data_class > i)] = i
    data_class[np.logical_and(y < ymin + margin*dy, data_class > i)] = i
    data_class[np.logical_and(y > ymax - margin*dy, data_class > i)] = i
    data_class[np.logical_and(z < zmin + margin*dz, data_class > i)] = i
    data_class[np.logical_and(z > zmax - margin*dz, data_class > i)] = i

# Not sure if magic number 22 is correct here
nBlocks = np.sum(blocks)
newBlocks = np.sum(blocks * 8.0**(data_class - reflevel))
logging.info(f'Original blocks: {nBlocks}')
logging.info(f'Original mem: {nBlocks * 64 * 22 / 1024**3}')
logging.info(f'Original nodes: {nBlocks * 64 * 22 / 1024**3 / 60}')
logging.info(f'New blocks: {newBlocks}')
logging.info(f'New mem: {newBlocks * 64 * 22 / 1024**3}')
logging.info(f'New nodes: {newBlocks * 64 * 22 / 1024**3 / 60}')
logging.info(f'Ratio: {newBlocks/nBlocks}')

# Change these as you wish
pt.plot.plot_colormap3dslice(filename=filename, expression=classify_alpha, colormap='viridis_r')
