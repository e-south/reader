from __future__ import annotations

from rich import box
from rich.panel import Panel

from . import shared
from .shared import app, table


@app.command(help="Show a quick guided walkthrough.")
def demo():
    steps = [
        ("1", "Find experiments", "reader ls"),
        ("2", "Show experiment details", "reader inspect 1"),
        ("3", "List protocols", "reader protocols"),
        (
            "4",
            "Scaffold a new experiment",
            "reader init ./experiments/20260317_new_assay --protocol plate_reader/dual_reporter_screen",
        ),
        ("5", "Starter YAML for a protocol", "reader protocols plate_reader/dual_reporter_screen --example-config"),
        ("6", "Show pipeline chain", "reader steps 1"),
        ("7", "Explain plan", "reader explain 1"),
        ("8", "Validate config + inputs", "reader validate 1"),
        ("9", "Run pipeline (records)", "reader run 1"),
        ("10", "See records", "reader records 1"),
        ("11", "List plot specs", "reader plot 1 --list"),
        ("12", "Save plots", "reader plot 1"),
        ("13", "Run exports", "reader export 1"),
        ("14", "Notebook (marimo)", "reader notebook 1"),
    ]
    listing = table("Reader Demo")
    listing.add_column("#", justify="right", style="muted")
    listing.add_column("Goal", style="accent")
    listing.add_column("Command", style="path")
    for row in steps:
        listing.add_row(*row)
    shared.console.print(
        Panel(
            listing,
            border_style="accent",
            box=box.ROUNDED,
            subtitle="[muted]Tip: replace the index with a path or experiment directory[/muted]",
        )
    )
