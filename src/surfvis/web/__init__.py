"""On-demand waterfall browser for surfchi2 output.

A small FastAPI + htmx app. The chi-squared zarr written by ``surfvis chi2``
drives an antenna-by-antenna heatmap; clicking a cell reads that baseline out
of the Measurement Set and renders its waterfall.

Never a Stimela cab -- a long-running GUI is not a batch task.
"""
