# flake8: noqa
import os
import sys
from contextlib import ExitStack
from surfvis.workers.main import cli
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('INIT')
import time
import fsspec

from scabha.schema_utils import clickify_parameters
from surfvis.parser.schemas import schema


@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.qaplots)
def qaplots(**kw):
    '''
    Quality assurance plots
    '''
    opts = OmegaConf.create(kw)

    if '://' in opts.output_folder:
        protocol = output_folder.split('://')[0]
        prefix = f'{protocol}://'
    else:
        protocol = 'file'
        prefix = ''

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(output_folder.split('/')[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = basedir + f'/{opts.xcolumn}_{ycolumn}'

    import psutil
    nthreads = psutil.cpu_count(logical=True)
    ncpu = psutil.cpu_count(logical=False)
    if opts.nthreads is None:
        opts.nthreads = nthreads//2
        ncpu = ncpu//2

    # if opts.product.upper() not in ["I","Q", "U", "V"]:
    #     raise NotImplementedError(f"Product {opts.product} not yet supported")

    OmegaConf.set_struct(opts, True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f'{str(basedir)}/qaplots_{timestamp}.log'
    pyscilog.log_to_file(logname)
    print(f'Logs will be written to {logname}', file=log)

    # TODO - prettier config printing
    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    from pfb import set_envs
    from ducc0.misc import resize_thread_pool, thread_pool_size
    resize_thread_pool(opts.nthreads)
    set_envs(opts.nthreads, ncpu)

    # with ExitStack() as stack:
    import dask
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    from pfb import set_client
    from distributed import wait, get_client
    client = set_client(opts.nworkers, log, client_log_level=opts.log_level)

    ti = time.time()
    _init(**opts)

    print(f"All done after {time.time() - ti}s", file=log)

    try:
        client.close()
    except Exception as e:
        raise e

def _init(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)
