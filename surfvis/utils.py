

import concurrent.futures as cf
import os
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def surf(p, q, gp, gq, data, resid, weight, flag, basename):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return _surf(p, q, gp, gq, data, resid, weight, flag, basename)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def _surf(p, q, gp, gq, data, resid, weight, flag, basename):
    # per baseline output folder
    foldername = basename + f'/baseline_{str(p)}_{str(q)}/'
    # test if dir exists
    if not os.path.isdir(foldername):
        os.system('mkdir '+foldername)


    ntime, nchan, ncorr = data.shape


    # we only do diagonals for now
    if ncorr > 1:
        plotcorrs = (0, -1)
        ncorr = len(plotcorrs)
    else:
        plotcorrs = (0,)

    # plot the raw data phases and amplitudes (unflagged)
    fig, ax = plt.subplots(nrows=ncorr, ncols=1)

    if ncorr == 1:
        ax = [ax]

    for c in plotcorrs:
        im = ax[c].imshow(np.abs(data[:, :, c]), cmap='inferno', interpolation=None)
        ax[c].set_title(f"Corr: {c}")
        ax[c].axis('off')

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="5%", pad=0.01)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

    plt.savefig(foldername + "data_abs.png",
                dpi=500, bbox_inches='tight')

    plt.close(fig)

    fig, ax = plt.subplots(nrows=ncorr, ncols=1)

    if ncorr == 1:
        ax = [ax]

    for c in plotcorrs:
        im = ax[c].imshow(np.angle(data[:, :, c]), cmap='inferno', interpolation=None)
        ax[c].set_title(f"Corr: {c}")
        ax[c].axis('off')

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="10%", pad=0.01)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

    plt.savefig(foldername + "data_phase.png",
                dpi=500, bbox_inches='tight')

    plt.close(fig)

    # if it's an autocorrelation we also plot the corrected data
    # amplitude ignoring flags
    if p == q:
        fig, ax = plt.subplots(nrows=ncorr, ncols=1)

        if ncorr == 1:
            ax = [ax]

        for c in plotcorrs:
            datac = data[:, :, c]/(gp[:, :, c] * gq[:, :, c].conj())
            im = ax[c].imshow(np.abs(datac), cmap='inferno', interpolation=None)
            ax[c].set_title(f"Corr: {c}")
            ax[c].axis('off')

            divider = make_axes_locatable(ax[c])
            cax = divider.append_axes("bottom", size="10%", pad=0.01)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

        plt.savefig(foldername + "corrected_data_abs.png",
                    dpi=500, bbox_inches='tight')
        plt.close(fig)

    # data with flags applied
    data[flag] = np.nan
    fig, ax = plt.subplots(nrows=ncorr, ncols=1)

    if ncorr == 1:
        ax = [ax]

    for c in plotcorrs:
        im = ax[c].imshow(np.abs(data[:, :, c]), cmap='inferno', interpolation=None)
        ax[c].set_title(f"Corr: {c}")
        ax[c].axis('off')

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="10%", pad=0.01)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

    plt.savefig(foldername + "flagged_data_abs.png",
                dpi=500, bbox_inches='tight')

    plt.close(fig)

    fig, ax = plt.subplots(nrows=ncorr, ncols=1)

    if ncorr == 1:
        ax = [ax]

    for c in plotcorrs:
        im = ax[c].imshow(np.angle(data[:, :, c]), cmap='inferno', interpolation=None)
        ax[c].set_title(f"Corr: {c}")
        ax[c].axis('off')

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="10%", pad=0.01)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

    plt.savefig(foldername + "flagged_data_phase.png",
                dpi=500, bbox_inches='tight')

    plt.close(fig)


    # residuals
    resid *= np.sqrt(weight)
    resid[flag] = np.nan

    # amplitude and phase
    fig, ax = plt.subplots(nrows=ncorr, ncols=1)

    if ncorr == 1:
        ax = [ax]

    for c in plotcorrs:
        im = ax[c].imshow(np.abs(resid[:, :, c]), cmap='inferno', interpolation=None)
        ax[c].set_title(f"Corr: {c}")
        ax[c].axis('off')

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="10%", pad=0.01)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

    plt.savefig(foldername + "resid_abs.png",
                dpi=500, bbox_inches='tight')

    plt.close(fig)

    fig, ax = plt.subplots(nrows=ncorr, ncols=1)

    if ncorr == 1:
        ax = [ax]

    for c in plotcorrs:
        im = ax[c].imshow(np.angle(resid[:, :, c]), cmap='inferno', interpolation=None)
        ax[c].set_title(f"Corr: {c}")
        ax[c].axis('off')

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="10%", pad=0.01)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

    plt.savefig(foldername + "resid_phase.png",
                dpi=500, bbox_inches='tight')

    plt.close(fig)

    # histogram real and imaginary parts
    for c in plotcorrs:
        residc = resid[:, :, c]
        flagc = flag[:, :, c]
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].hist(residc[~flagc].real, bins=25)
        ax[0].set_title(f"real")
        # ax[0].axis('off')

        ax[1].hist(residc[~flagc].imag, bins=25)
        ax[1].set_title(f"imag")
        # ax[1].axis('off')

        plt.savefig(foldername + f"hist_corr{c}.png", dpi=500, bbox_inches='tight')
        plt.close(fig)

    if p == q:
        chi2_dof = np.nan  # spoils the plot if we don't do this
    else:
        chi2 = np.vdot(resid[~flag], resid[~flag]).real
        N = np.sum(~flag)
        if N > 0:
            chi2_dof = chi2/N
        else:
            chi2_dof = np.nan
    wsum = np.sum(weight[~flag])

    # plot gain in autocorr folder
    if p == q:
        fig, ax = plt.subplots(nrows=ncorr, ncols=1)

        if ncorr == 1:
            ax = [ax]

        for c in plotcorrs:
            im = ax[c].imshow(np.abs(gp[:, :, c]), cmap='inferno', interpolation=None)
            ax[c].set_title(f"Corr: {c}")
            ax[c].axis('off')

            divider = make_axes_locatable(ax[c])
            cax = divider.append_axes("bottom", size="10%", pad=0.01)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

        plt.savefig(foldername + "gain_abs.png",
                    dpi=500, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(nrows=ncorr, ncols=1)

        if ncorr == 1:
            ax = [ax]

        for c in plotcorrs:
            im = ax[c].imshow(np.angle(gp[:, :, c]), cmap='inferno', interpolation=None)
            ax[c].set_title(f"Corr: {c}")
            ax[c].axis('off')

            divider = make_axes_locatable(ax[c])
            cax = divider.append_axes("bottom", size="10%", pad=0.01)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            # cb.ax.tick_params(length=0.1, width=0.1, labelsize=1.0, pad=0.1)

        plt.savefig(foldername + "gain_phase.png",
                    dpi=500, bbox_inches='tight')
        plt.close(fig)


    return p, q, chi2_dof, wsum



