"""CLI for surfvis."""

import typer

app = typer.Typer(
    name="surfvis",
    help="Per-baseline time/frequency and chi-squared plots from Measurement Sets",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Per-baseline time/frequency and chi-squared plots from Measurement Sets"""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from surfvis.cli.chi2 import chi2  # noqa: E402
from surfvis.cli.flag_chi2 import flag_chi2  # noqa: E402
from surfvis.cli.summary import summary  # noqa: E402
from surfvis.cli.surf import surf  # noqa: E402

app.command(name="summary")(summary)
app.command(name="surf")(surf)
app.command(name="chi2")(chi2)
app.command(name="flag-chi2")(flag_chi2)

__all__ = ["app"]
