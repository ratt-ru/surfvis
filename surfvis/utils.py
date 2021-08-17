

import concurrent.futures as cf
import os
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def surf(p, q, data, weight, flag, basename, products):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return _surf(p, q, data, weight, flag, basename, products=products)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def _surf(p, q, data, weight, flag, basename, products='riap'):
    # per baseline output folder
    foldername = basename + f'/baseline_{str(p)}_{str(q)}/'
    # test if dir exists
    if not os.path.isdir(foldername):
        os.system('mkdir '+foldername)

    ntime, nchan, ncorr = data.shape

    ratio = ntime/nchan

    # whiten data
    dataw = data * np.sqrt(weight)
    dataw[flag] = np.nan

    # plot whitened data amplitudes and phases
    if 'a' in products.lower():
        fig, ax = plt.subplots(nrows=ncorr, ncols=1, figsize=(ncorr*5*ratio,5))

        if ncorr == 1:
            ax = [ax]

        for c in range(ncorr):
            im = ax[c].imshow(np.abs(dataw[:, :, c]), cmap='inferno', interpolation=None)
            ax[c].set_title(f"Corr: {c}")
            ax[c].axis('off')

            divider = make_axes_locatable(ax[c])
            cax = divider.append_axes("bottom", size="10%", pad=0.01)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

        fig.tight_layout(w_pad=0.98,h_pad=0.98,rect=[0.02,0.02,0.95,0.95])

        plt.savefig(foldername + f"data_corr{c}_abs.png",
                    dpi=500, bbox_inches='tight')

        plt.close(fig)

    if 'p' in products.lower():
        fig, ax = plt.subplots(nrows=ncorr, ncols=1, figsize=(ncorr*5*ratio,5))

        if ncorr == 1:
            ax = [ax]

        for c in range(ncorr):
            im = ax[c].imshow(np.angle(dataw[:, :, c]), cmap='inferno', interpolation=None)
            ax[c].set_title(f"Corr: {c}")
            ax[c].axis('off')

            divider = make_axes_locatable(ax[c])
            cax = divider.append_axes("bottom", size="10%", pad=0.01)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

        fig.tight_layout(w_pad=0.98,h_pad=0.98,rect=[0.02,0.02,0.95,0.95])

        plt.savefig(foldername + f"data_corr{c}_phase.png",
                    dpi=500, bbox_inches='tight')

        plt.close(fig)

    # histogram real and imaginary parts
    for c in range(ncorr):
        datac = dataw[:, :, c]
        flagc = flagc[:, :, c]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,5))
        ax[0].hist(datac[~flagc].real, bins=25)
        ax[0].set_title(f"real")
        # ax[0].axis('off')

        ax[1].hist(datac[~flagc].imag, bins=25)
        ax[1].set_title(f"imag")
        # ax[1].axis('off')

        fig.tight_layout(w_pad=0.98,h_pad=0.98,rect=[0.02,0.02,0.95,0.95])
        plt.savefig(foldername + f"hist_corr{c}.png", dpi=500, bbox_inches='tight')
        plt.close(fig)

    # if it's an autocorrelation we plot some extra goodies
    if p == q:
        fig, ax = plt.subplots(nrows=ncorr, ncols=1, figsize=(ncorr*10*ratio,10))
        chi2_dof = np.nan

        # weighted sum over time and plot as a function of freq
        datanu = np.sum(data * weight, axis=0)
        weightnu = np.sum(weight, axis=0)
        datanu = np.where(weightnu > 0.0, datanu/weightnu, np.nan)

        fig, ax = plt.subplots(nrows=ncorr, ncols=1, figsize=(ncorr*5*ratio,5))
        nu = np.arange(nchan)
        if ncorr == 1:
            ax = [ax]

        for c in range(ncorr):
            ax[c].errorbar(nu, datanu[:, c], np.sqrt(1.0/weightnu[:, c]), fmt='xr')
            ax[c].set_title(f'corr_{c}')

        fig.tight_layout(w_pad=0.98,h_pad=0.98,rect=[0.02,0.02,0.95,0.95])
        plt.savefig(foldername + f"lightcurve.png", dpi=500, bbox_inches='tight')
        plt.close(fig)


    else:
        chi2 = np.vdot(dataw[~flag], dataw[~flag]).real
        chi2_dof = chi2/np.sum(~flag)

    wsum = np.sum(weight[~flag])

    return p, q, chi2_dof, wsum



