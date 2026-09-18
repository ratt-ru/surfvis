"""FastAPI application wiring the chi-squared store to waterfall plots."""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from surfvis.web import msdata, render
from surfvis.web.store import Chi2Store, Selection

HERE = Path(__file__).parent


def create_app(data: str | Path, ms: str | Path | None = None) -> FastAPI:
    """Build the app around one chi-squared dataset.

    Args:
        data: Path to the zarr written by ``surfvis chi2 --dataout``.
        ms: Measurement Set to read waterfalls from. Defaults to the path
            recorded in the dataset, which is right unless it has moved.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    store = Chi2Store(data)
    app = FastAPI(title="surfvis")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    first = store.scans[0]
    ms_path = str(ms) if ms is not None else store.chunk(Selection(*first, 0, 0, store.bins(*first)[2][0])).ms

    def _selection(field, spw, scan, t, f, c) -> Selection:
        return Selection(field=field, spw=spw, scan=scan, time_bin=t, freq_bin=f, corr=c)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        field, spw, scan = store.scans[0]
        times, freqs, corrs = store.bins(field, spw, scan)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "store": store,
                "ms": ms_path,
                "field": field,
                "spw": spw,
                "scan": scan,
                "times": times,
                "freqs": freqs,
                "corrs": corrs,
            },
        )

    @app.get("/scans", response_class=HTMLResponse)
    def scans(request: Request, field: int = Query(...), spw: int = Query(...)):
        """Re-render the scan/bin selectors when field or spw changes."""
        spws = store.spws(field)
        if spw not in spws:
            spw = spws[0]
        scan = store.scan_numbers(field, spw)[0]
        times, freqs, corrs = store.bins(field, spw, scan)
        return templates.TemplateResponse(
            request,
            "_selectors.html",
            {
                "store": store,
                "field": field,
                "spw": spw,
                "scan": scan,
                "times": times,
                "freqs": freqs,
                "corrs": corrs,
            },
        )

    @app.get("/heatmap", response_class=HTMLResponse)
    def heatmap(
        request: Request,
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(0),
        f: int = Query(0),
        c: int = Query(0),
        scale: str = Query("log"),
    ):
        try:
            chunk = store.chunk(_selection(field, spw, scan, t, f, c))
        except (KeyError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=f"No such chunk: {exc}") from exc

        try:
            norm, vmin, vmax = render.make_norm(chunk.chi2_dof, scale)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        colours = render.cell_colours(chunk.chi2_dof, norm)
        finite = chunk.finite
        return templates.TemplateResponse(
            request,
            "_heatmap.html",
            {
                "chunk": chunk,
                "colours": colours,
                "names": chunk.antenna_names,
                "nant": len(chunk.antenna_names),
                "vmin": vmin,
                "vmax": vmax,
                "scale": scale,
                "nvalid": int(finite.size),
                "worst": float(np.nanmax(chunk.chi2_dof)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "query": f"field={field}&spw={spw}&scan={scan}&t={t}&f={f}&c={c}",
                "scales": render.SCALES,
                "columns": _columns(ms_path, field, spw, scan),
                "default_column": chunk.rcol,
            },
        )

    @app.get("/histogram.png")
    def histogram_png(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(0),
        f: int = Query(0),
        c: int = Query(0),
        scale: str = Query("log"),
    ):
        chunk = store.chunk(_selection(field, spw, scan, t, f, c))
        norm, _, _ = render.make_norm(chunk.chi2_dof, scale)
        png = render.histogram_png(chunk.chi2_dof, norm, title=f"scan {scan}  t{t} f{f} pol{c}")
        return Response(png, media_type="image/png")

    @app.get("/waterfall", response_class=HTMLResponse)
    def waterfall(
        request: Request,
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(...),
        f: int = Query(...),
        c: int = Query(...),
        p: int = Query(...),
        q: int = Query(...),
        column: str = Query(""),
        quantity: str = Query("amp"),
    ):
        """Return the <img> wrapper; the browser then fetches the PNG."""
        chunk = store.chunk(_selection(field, spw, scan, t, f, c))
        names = chunk.antenna_names
        column = column or chunk.rcol
        query = f"field={field}&spw={spw}&scan={scan}&t={t}&f={f}&c={c}&p={p}&q={q}&column={column}&quantity={quantity}"
        return templates.TemplateResponse(
            request,
            "_waterfall.html",
            {
                "query": query,
                "name1": names[p],
                "name2": names[q],
                "value": float(chunk.chi2_dof[p, q]),
                "columns": _columns(ms_path, field, spw, scan),
                "column": column,
                "quantity": quantity,
                "quantities": msdata.QUANTITIES,
                "t0": chunk.t0,
                "tf": chunk.tf,
                "chan0": chunk.chan0,
                "chanf": chunk.chanf,
            },
        )

    @app.get("/waterfall.png")
    def waterfall_png(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(...),
        f: int = Query(...),
        c: int = Query(...),
        p: int = Query(...),
        q: int = Query(...),
        column: str = Query(""),
        quantity: str = Query("amp"),
    ):
        chunk = store.chunk(_selection(field, spw, scan, t, f, c))
        names = chunk.antenna_names
        try:
            wf = msdata.waterfall(
                ms_path,
                field=field,
                spw=spw,
                scan=scan,
                name1=names[p],
                name2=names[q],
                polarization=c,
                column=column or chunk.rcol,
                quantity=quantity,
                rect=(chunk.t0, chunk.tf, chunk.chan0, chunk.chanf),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(render.waterfall_png(wf), media_type="image/png")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "scans": len(store.scans), "ms": ms_path}

    return app


def _columns(ms: str, field: int, spw: int, scan: int) -> list[str]:
    """Plottable columns, or an empty list if the MS cannot be opened."""
    try:
        return msdata.available_columns(ms, field, spw, scan)
    except Exception:
        return []
