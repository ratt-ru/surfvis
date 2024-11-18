"""
Per-baseline time freq plots

author - Ian Heywood
email  - ianh@astro.ox.ac.uk
date   - 02/08/2021
"""
__version__ = '0.0.1'


import os
import sys
import logging


def set_envs(nthreads, ncpu):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NPY_NUM_THREADS"] = str(nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    os.environ["JAX_PLATFORMS"] = 'cpu'
    os.environ["JAX_ENABLE_X64"] = 'True'
    # this may be required for numba parallelism
    # find python and set LD_LIBRARY_PATH
    paths = sys.path
    ppath = [paths[i] for i in range(len(paths)) if 'pfb/bin' in paths[i]]
    if len(ppath):
        ldpath = ppath[0].replace('bin', 'lib')
        ldcurrent = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ["LD_LIBRARY_PATH"] = f'{ldpath}:{ldcurrent}'
