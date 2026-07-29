from __future__ import annotations

from rich import box
from rich.panel import Panel

from . import shared
from .shared import app, table


@app.command(help="Show a quick guided walkthrough.")
def demo():
    steps = [
        ("1", "Find experiments", "reader ls --details --readiness"),
        ("2", "Show experiment details", "reader inspect ./experiments/<experiment>/config.yaml"),
        ("3", "List protocols", "reader protocols"),
        (
            "4",
            "Scaffold a new experiment",
            "reader init ./experiments/my_experiment --protocol <protocol-id>",
        ),
        ("5", "Starter YAML for a protocol", "reader protocols <protocol-id> --example-config"),
        ("6", "Show pipeline chain", "reader steps ./experiments/<experiment>/config.yaml"),
        ("7", "Explain plan", "reader explain ./experiments/<experiment>/config.yaml"),
        ("8", "Validate config + inputs", "reader validate ./experiments/<experiment>/config.yaml"),
        ("9", "Run pipeline (records)", "reader run ./experiments/<experiment>/config.yaml"),
        ("10", "See records", "reader records ./experiments/<experiment>/config.yaml"),
        ("11", "List plot specs", "reader plot ./experiments/<experiment>/config.yaml --list"),
        ("12", "Save plots", "reader plot ./experiments/<experiment>/config.yaml"),
        ("13", "Run exports", "reader export ./experiments/<experiment>/config.yaml"),
        ("14", "Notebook (Marimo)", "reader notebook ./experiments/<experiment>/config.yaml"),
        ("15", "Verify records and provenance", "reader verify ./experiments/<experiment>/config.yaml"),
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
