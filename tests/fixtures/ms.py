"""Build a tiny Measurement Set for tests and demos.

Small enough to write in well under a second, complete enough that xarray-ms
will open it: FEED, STATE and the FIELD direction columns all have to be
populated or the MSv4 view refuses to construct.

The RESIDUAL column is deliberately non-standard. That is what makes it
visible to an MSv4 reader as a secondary column, rather than being renamed the
way DATA and WEIGHT_SPECTRUM are.

The columns are mutually consistent, so corrections can be checked exactly::

    DATA           = G_p MODEL_DATA G_q^H + RESIDUAL
    CORRECTED_DATA = inv(G_p) DATA inv(G_q)^H

with diagonal gains ``G`` that vary in time, frequency and antenna, returned
as ``info["gains"]`` and written by :func:`write_gain_table`.
"""

import numpy as np


def make_ms(
    path: str,
    nant: int = 8,
    nchan: int = 32,
    ncorr: int = 4,
    ntime: int = 12,
    nscan: int = 2,
    hot: tuple[int, int] | None = (1, 5),
    hot_scale: float = 25.0,
    seed: int = 42,
) -> dict:
    """Write a Measurement Set with a known bad baseline.

    Args:
        path: Where to write the MS.
        nant: Number of antennas.
        nchan: Channels per spectral window.
        ncorr: Correlations.
        ntime: Time slots per scan.
        nscan: Number of scans.
        hot: ``(p, q)`` antenna pair whose residuals are inflated, or None.
        hot_scale: Factor applied to the hot baseline's residuals.
        seed: RNG seed.

    Returns:
        A dict describing what was written, including the hot baseline.
    """
    from casacore.tables import complete_ms_desc, default_ms, makearrcoldesc, makedminfo, maketabdesc, table

    rng = np.random.default_rng(seed)
    baselines = [(p, q) for p in range(nant) for q in range(p + 1, nant)]
    nbl = len(baselines)
    nrow = nbl * ntime * nscan

    desc = complete_ms_desc("MAIN")
    default_ms(path, desc, makedminfo(desc))

    main = table(path, readonly=False, ack=False)
    main.addrows(nrow)
    ant1 = np.array([p for _ in range(ntime * nscan) for p, _ in baselines])
    ant2 = np.array([q for _ in range(ntime * nscan) for _, q in baselines])
    times = np.repeat(np.arange(ntime * nscan, dtype=float) * 8.0 + 4.9e9, nbl)
    scans = np.repeat(np.repeat(np.arange(1, nscan + 1), ntime), nbl)

    main.putcol("ANTENNA1", ant1)
    main.putcol("ANTENNA2", ant2)
    main.putcol("TIME", times)
    main.putcol("DATA_DESC_ID", np.zeros(nrow, int))
    main.putcol("FIELD_ID", np.zeros(nrow, int))
    main.putcol("SCAN_NUMBER", scans)
    main.putcol("STATE_ID", np.zeros(nrow, int))
    main.putcol("UVW", rng.normal(size=(nrow, 3)) * 1e3)

    shape = (nrow, nchan, ncorr)
    main.putcol("FLAG", np.zeros(shape, bool))
    main.putcol("WEIGHT_SPECTRUM", np.ones(shape, np.float32))

    main.addcols(maketabdesc(makearrcoldesc("RESIDUAL", 0.0 + 0j, ndim=2, valuetype="complex")))
    resid = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    if hot is not None:
        p, q = hot
        bad = (ant1 == p) & (ant2 == q)
        resid[bad] *= hot_scale

    # A slightly polarized point source, and diagonal gains on every axis.
    receptors = [(0, 0), (0, 1), (1, 0), (1, 1)] if ncorr == 4 else [(0, 0), (1, 1)][:ncorr]
    stokes_model = {(0, 0): 1.1, (0, 1): 0.05, (1, 0): 0.05, (1, 1): 0.9}
    model = np.empty(shape, np.complex128)
    for c, ij in enumerate(receptors):
        model[..., c] = stokes_model[ij]
    gains = rng.uniform(0.5, 2.0, size=(ntime * nscan, nchan, nant, 2)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, size=(ntime * nscan, nchan, nant, 2))
    )
    tslot = np.repeat(np.arange(ntime * nscan), nbl)
    corrupt = np.empty(shape, np.complex128)
    for c, (i, j) in enumerate(receptors):
        corrupt[..., c] = gains[tslot, :, ant1, i] * np.conj(gains[tslot, :, ant2, j])

    data = (corrupt * model + resid).astype(np.complex64)
    main.putcol("DATA", data)
    main.putcol("RESIDUAL", resid)
    main.putcol("MODEL_DATA", model.astype(np.complex64))
    main.putcol("CORRECTED_DATA", (data / corrupt).astype(np.complex64))
    main.close()

    names = [f"m{i:03d}" for i in range(nant)]
    sub = table(f"{path}/ANTENNA", readonly=False, ack=False)
    sub.addrows(nant)
    sub.putcol("NAME", np.array(names))
    sub.putcol("POSITION", rng.normal(size=(nant, 3)) * 1e3)
    sub.close()

    sub = table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False)
    sub.addrows(1)
    sub.putcell("CHAN_FREQ", 0, np.linspace(1.0e9, 1.1e9, nchan))
    sub.putcell("CHAN_WIDTH", 0, np.full(nchan, 1e8 / nchan))
    sub.putcell("NUM_CHAN", 0, nchan)
    sub.putcell("REF_FREQUENCY", 0, 1.0e9)
    sub.close()

    sub = table(f"{path}/POLARIZATION", readonly=False, ack=False)
    sub.addrows(1)
    sub.putcell("NUM_CORR", 0, ncorr)
    sub.putcell("CORR_TYPE", 0, np.array([9, 10, 11, 12] if ncorr == 4 else [9, 12][:ncorr]))
    sub.close()

    sub = table(f"{path}/FIELD", readonly=False, ack=False)
    sub.addrows(1)
    sub.putcell("NAME", 0, "testfield")
    sub.putcell("SOURCE_ID", 0, 0)
    for col in ("PHASE_DIR", "REFERENCE_DIR", "DELAY_DIR"):
        sub.putcell(col, 0, np.array([[0.5, -0.6]]))
    sub.close()

    sub = table(f"{path}/DATA_DESCRIPTION", readonly=False, ack=False)
    sub.addrows(1)
    sub.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    sub.putcell("POLARIZATION_ID", 0, 0)
    sub.close()

    # xarray-ms validates ANTENNA1/2 against FEED::ANTENNA_ID, so FEED must exist.
    sub = table(f"{path}/FEED", readonly=False, ack=False)
    sub.addrows(nant)
    sub.putcol("ANTENNA_ID", np.arange(nant))
    sub.putcol("FEED_ID", np.zeros(nant, int))
    sub.putcol("SPECTRAL_WINDOW_ID", np.full(nant, -1))
    sub.putcol("TIME", np.zeros(nant))
    sub.putcol("INTERVAL", np.zeros(nant))
    sub.putcol("NUM_RECEPTORS", np.full(nant, 2))
    sub.putcol("BEAM_ID", np.full(nant, -1))
    sub.putcol("BEAM_OFFSET", np.zeros((nant, 2, 2)))
    sub.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]] * nant))
    sub.putcol("POL_RESPONSE", np.zeros((nant, 2, 2), np.complex64))
    sub.putcol("POSITION", np.zeros((nant, 3)))
    sub.putcol("RECEPTOR_ANGLE", np.zeros((nant, 2)))
    sub.close()

    sub = table(f"{path}/STATE", readonly=False, ack=False)
    sub.addrows(1)
    sub.putcell("OBS_MODE", 0, "OBSERVE_TARGET#ON_SOURCE")
    sub.putcell("SIG", 0, True)
    sub.putcell("REF", 0, False)
    sub.putcell("CAL", 0, 0.0)
    sub.putcell("LOAD", 0, 0.0)
    sub.putcell("SUB_SCAN", 0, 0)
    sub.putcell("FLAG_ROW", 0, False)
    sub.close()

    return {
        "path": path,
        "nant": nant,
        "nchan": nchan,
        "ncorr": ncorr,
        "ntime": ntime,
        "nscan": nscan,
        "antenna_names": names,
        "hot": hot,
        "gains": gains,
        "times": np.unique(times),
        "freqs": np.linspace(1.0e9, 1.1e9, nchan),
    }


def write_gain_table(info: dict, path: str, term: str = "NET") -> str:
    """Write ``info["gains"]`` the way QuartiCal lays out a gain term.

    One zarr group per (field, spw, scan) -- ``<path>/<term>/<term>_<n>`` --
    holding ``gains`` ``(gain_time, gain_freq, antenna, direction,
    correlation)`` and ``gain_flags``, with the partition ids as attributes.

    Returns:
        The term directory, which is what ``--gtable`` takes.
    """
    import xarray as xr

    ntime, nscan = info["ntime"], info["nscan"]
    for n in range(nscan):
        sl = slice(n * ntime, (n + 1) * ntime)
        g = info["gains"][sl][:, :, :, None, :]
        ds = xr.Dataset(
            {
                "gains": (("gain_time", "gain_freq", "antenna", "direction", "correlation"), g),
                "gain_flags": (("gain_time", "gain_freq", "antenna", "direction"), np.zeros(g.shape[:-1], np.int8)),
            },
            coords={
                "gain_time": info["times"][sl],
                "gain_freq": info["freqs"],
                "antenna": info["antenna_names"],
                "direction": [0],
                "correlation": ["XX", "YY"],
            },
            attrs={"FIELD_ID": 0, "DATA_DESC_ID": 0, "SCAN_NUMBER": n + 1, "TYPE": "complex"},
        )
        ds.to_zarr(f"{path}/{term}/{term}_{n}", mode="w")
    return f"{path}/{term}"


def add_flag_version(ms: str, name: str, flags: np.ndarray, comment: str = "") -> None:
    """Save ``flags`` as a CASA flag version of ``ms``.

    CASA keeps versions next to the MS, in ``<ms>.flagversions/``: a
    ``FLAG_VERSION_LIST`` of ``name : comment`` lines, and one ``flags.<name>``
    table per version holding FLAG and FLAG_ROW, row-aligned with the MS.
    """
    import os

    from casacore.tables import makearrcoldesc, makescacoldesc, maketabdesc, table

    root = f"{ms.rstrip('/')}.flagversions"
    os.makedirs(root, exist_ok=True)
    desc = maketabdesc([makearrcoldesc("FLAG", False, ndim=2, valuetype="boolean"), makescacoldesc("FLAG_ROW", False)])
    t = table(f"{root}/flags.{name}", desc, nrow=flags.shape[0], ack=False)
    try:
        t.putcol("FLAG", flags)
        t.putcol("FLAG_ROW", np.zeros(flags.shape[0], bool))
    finally:
        t.close()
    with open(f"{root}/FLAG_VERSION_LIST", "a") as listing:
        listing.write(f"{name} : {comment or name}\n")


def copy_with_tiled_flags(src: str, dest: str) -> str:
    """Copy an MS, re-storing FLAG in a TiledShapeStMan.

    ``default_ms`` gives FLAG a StandardStMan, which re-reads from disk on
    every access. Real Measurement Sets -- anything written by the MeerKAT
    archive, CASA or dask-ms -- use a TiledShapeStMan instead, whose tile cache
    belongs to the open table object and therefore keeps serving the flags a
    long-running reader saw first. Tests about staleness need the storage
    manager the field actually uses.

    Args:
        src: An existing Measurement Set.
        dest: Where to write the copy.

    Returns:
        ``dest``.
    """
    import shutil

    from casacore.tables import makearrcoldesc, maketabdesc, table

    shutil.copytree(src, dest)
    t = table(dest, readonly=False, ack=False)
    try:
        flags = t.getcol("FLAG")
        _, nchan, ncorr = flags.shape
        t.removecols("FLAG")
        t.addcols(
            maketabdesc(makearrcoldesc("FLAG", False, ndim=2, shape=[nchan, ncorr], valuetype="boolean")),
            {
                "TYPE": "TiledShapeStMan",
                "NAME": "TiledFlag",
                "SPEC": {"DEFAULTTILESHAPE": [ncorr, nchan, 4]},
            },
        )
        t.putcol("FLAG", flags)
    finally:
        t.close()
    return dest
