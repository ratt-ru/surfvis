"""FastAPI application wiring the chi-squared store to waterfall plots."""

import hashlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from surfvis.utils import products as products_mod
from surfvis.web import msdata, render
from surfvis.web.flagging import Flagger
from surfvis.web.store import Chi2Store, Selection

HERE = Path(__file__).parent


class FlagRequest(BaseModel):
    """A polygon on one baseline's waterfall, in its (time slot, channel) coordinates."""

    field: int
    spw: int
    scan: int
    p: int
    q: int
    product: str
    vertices: list[tuple[float, float]] = Field(min_length=3)
    # "product": the correlations the plotted product is formed from; "all": every correlation.
    correlations: Literal["product", "all"] = "product"
    dry_run: bool = False


def create_app(data: str | Path, ms: str | Path | None = None, allow_flagging: bool = False) -> FastAPI:
    """Build the app around one chi-squared dataset.

    Args:
        data: Path to the zarr written by ``surfvis chi2 --dataout``.
        ms: Measurement Set to read waterfalls from. Defaults to the path
            recorded in the dataset, which is right unless it has moved.
        allow_flagging: Enable the routes that write FLAG. Off by default:
            the browser is read-only unless asked otherwise.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    store = Chi2Store(data)
    app = FastAPI(title="surfvis")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    # StaticFiles sends no Cache-Control, so browsers cache by heuristic and
    # keep serving an old stylesheet after an upgrade. A version in the URL
    # makes every change a new resource.
    templates.env.globals["static_version"] = _static_version(HERE / "static")

    first = store.scans[0]
    ms_path = str(ms) if ms is not None else store.chunk(Selection(*first, 0, 0, store.bins(*first)[2][0])).ms

    def _selection(field, spw, scan, t, f, c) -> Selection:
        return Selection(field=field, spw=spw, scan=scan, time_bin=t, freq_bin=f, product=c)

    def _chunk(field, spw, scan, t, f, c):
        try:
            return store.chunk(_selection(field, spw, scan, t, f, c))
        except (KeyError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=f"No such chunk: {exc}") from exc

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
                "field_names": msdata.field_names(ms_path),
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
                "field_names": msdata.field_names(ms_path),
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
        c: str = Query(...),
        scale: str = Query("log"),
    ):
        chunk = _chunk(field, spw, scan, t, f, c)

        try:
            norm, vmin, vmax = render.make_norm(chunk.chi2_dof, scale)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        colours = render.cell_colours(chunk.chi2_dof, norm)
        finite = chunk.finite
        # What the hover readout and keyboard navigation need, sent once
        # rather than as attributes on every cell.
        grid = {
            "names": chunk.antenna_names,
            "values": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in chunk.chi2_dof],
            "positions": msdata.antenna_positions(ms_path, chunk.antenna_names),
        }
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
                "nvalid": chunk.nbaselines,
                "worst": float(np.nanmax(chunk.chi2_dof)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "query": urlencode({"field": field, "spw": spw, "scan": scan, "t": t, "f": f, "c": c}),
                "scales": render.SCALES,
                "product": c,
                "grid": grid,
            },
        )

    @app.get("/histogram.png")
    def histogram_png(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(0),
        f: int = Query(0),
        c: str = Query(...),
        scale: str = Query("log"),
    ):
        chunk = _chunk(field, spw, scan, t, f, c)
        norm, _, _ = render.make_norm(chunk.chi2_dof, scale)
        png = render.histogram_png(chunk.chi2_dof, norm)
        return Response(png, media_type="image/png")

    def _resolve(column: str) -> str:
        """MSv4 name for a column, so ``DATA`` from chi2 matches ``VISIBILITY`` in the menus."""
        if not column:
            return column
        try:
            return msdata.data_group(ms_path).resolve(column)
        except Exception:
            return column

    def _waterfall_settings(chunk, product, column, model, gtable, flagversion, quantity):
        """Fill unset waterfall controls from how chi2 formed the heatmap.

        ``None`` means "not given", so the heatmap's setting applies; an empty
        string is a deliberate "none" (no model, no gains).
        """
        return {
            "product": product or chunk.selection.product,
            "column": _resolve(column or chunk.rcol),
            "model": _resolve(chunk.mcol if model is None else model),
            "gtable": chunk.gtable if gtable is None else gtable.strip(),
            "flagversion": flagversion or "",
            "quantity": quantity,
        }

    @app.get("/waterfall", response_class=HTMLResponse)
    def waterfall(
        request: Request,
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        t: int = Query(...),
        f: int = Query(...),
        c: str = Query(...),
        p: int = Query(...),
        q: int = Query(...),
        product: str = Query(""),
        column: str = Query(""),
        model: str | None = Query(None),
        gtable: str | None = Query(None),
        flagversion: str = Query(""),
        quantity: str = Query("amp"),
    ):
        """Return the controls and <img> wrapper; the browser then fetches the PNG.

        The controls drive the waterfall only. The heatmap is what ``chi2``
        computed, so the panel says when the two no longer describe the same
        quantity.
        """
        chunk = _chunk(field, spw, scan, t, f, c)
        names = chunk.antenna_names
        settings = _waterfall_settings(chunk, product, column, model, gtable, flagversion, quantity)
        base = {"field": field, "spw": spw, "scan": scan, "t": t, "f": f, "c": c, "p": p, "q": q}
        problems = msdata.check(
            ms_path,
            field,
            spw,
            scan,
            settings["product"],
            settings["column"],
            model=settings["model"],
            gtable=settings["gtable"],
            flagversion=settings["flagversion"],
        )
        defaults = _waterfall_settings(chunk, "", "", None, None, "", "amp")
        # Controls the user moved off the heatmap's setting. Clicking another
        # cell sends these along, so a choice survives a change of baseline.
        keep = {key for key, value in settings.items() if value != defaults[key]}
        return templates.TemplateResponse(
            request,
            "_waterfall.html",
            {
                "base": base,
                "query": urlencode({**base, **settings}),
                "name1": names[p],
                "name2": names[q],
                "value": float(chunk.chi2_dof[p, q]),
                # The column is corrected and has a model subtracted, so only
                # visibility-like columns make sense; WEIGHT does not.
                "columns": _columns(ms_path, field, spw, scan, complex_only=True),
                "models": _columns(ms_path, field, spw, scan, complex_only=True),
                "products": _products(ms_path, field, spw, scan),
                "flagversions": msdata.flag_versions(ms_path),
                "quantities": msdata.QUANTITIES,
                "s": settings,
                "chunk": chunk,
                "keep": keep,
                "differs": bool(keep - {"quantity"}),
                "problems": problems,
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
        c: str = Query(...),
        p: int = Query(...),
        q: int = Query(...),
        product: str = Query(""),
        column: str = Query(""),
        model: str | None = Query(None),
        gtable: str | None = Query(None),
        flagversion: str = Query(""),
        quantity: str = Query("amp"),
    ):
        chunk = _chunk(field, spw, scan, t, f, c)
        names = chunk.antenna_names
        settings = _waterfall_settings(chunk, product, column, model, gtable, flagversion, quantity)
        try:
            wf = msdata.waterfall(
                ms_path,
                field=field,
                spw=spw,
                scan=scan,
                name1=names[p],
                name2=names[q],
                rect=(chunk.t0, chunk.tf, chunk.chan0, chunk.chanf),
                **settings,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(render.waterfall_png(wf), media_type="image/png")

    flagger = Flagger(ms_path) if allow_flagging else None

    def _flagger() -> Flagger:
        if flagger is None:
            raise HTTPException(status_code=403, detail="Flagging is disabled; start serve with --allow-flagging")
        return flagger

    @app.post("/flag")
    def flag(req: FlagRequest):
        """Flag the cells inside a polygon on one baseline, or count them with ``dry_run``."""
        flagger_ = _flagger()
        try:
            names = list(store.group(req.field, req.spw, req.scan).attrs["antenna_names"])
            labels = msdata.labels(ms_path, req.field, req.spw, req.scan)
            if req.correlations == "all":
                corr_idx = list(range(len(labels)))
            else:
                corr_idx = [labels.index(c) for c in products_mod.needed([req.product], labels)]
            return flagger_.apply(
                req.field,
                req.spw,
                req.scan,
                names[req.p],
                names[req.q],
                np.asarray(req.vertices, dtype=float),
                corr_idx,
                dry_run=req.dry_run,
            )
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/flag/undo")
    def flag_undo():
        """Restore the cells the most recent flag edit changed."""
        edit = _flagger().undo()
        if edit is None:
            raise HTTPException(status_code=409, detail="Nothing to undo")
        return {"undone": edit.id, "description": edit.description, "cells": edit.cells}

    @app.get("/flag/state")
    def flag_state():
        if flagger is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "backup": flagger.backup,
            "edits": [
                {"id": e.id, "description": e.description, "cells": e.cells, "newly_flagged": e.newly_flagged}
                for e in flagger.edits
            ],
        }

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "scans": len(store.scans), "ms": ms_path, "flagging": flagger is not None}

    return app


def _static_version(root: Path) -> str:
    """Short digest of the static files' names, sizes and mtimes."""
    stats = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file())
    return hashlib.sha1(repr(stats).encode()).hexdigest()[:10]


def _columns(ms: str, field: int, spw: int, scan: int, complex_only: bool = False) -> list[str]:
    """Plottable columns, or an empty list if the MS cannot be opened."""
    try:
        return msdata.available_columns(ms, field, spw, scan, complex_only=complex_only)
    except Exception:
        return []


def _products(ms: str, field: int, spw: int, scan: int) -> list[str]:
    """Correlations and formable Stokes parameters, or an empty list if the MS cannot be opened."""
    try:
        return msdata.available_products(ms, field, spw, scan)
    except Exception:
        return []
