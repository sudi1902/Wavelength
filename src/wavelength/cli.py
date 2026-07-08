"""Wavelength CLI.

    wavelength extract video.mp4 [video2.mov ...]   # the core workflow
    wavelength setup [bandit|demucs]                # first-run model install
    wavelength library                              # list stored effects
    wavelength info                                 # config + engine status
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from wavelength import __version__
from wavelength.config import SUPPORTED_EXTENSIONS, load_settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline.ingest import IngestError
from wavelength.pipeline.runner import PipelineError, extract_video
from wavelength.pipeline.separate import SeparationError, get_separator

app = typer.Typer(
    name="wavelength",
    help="Extract individual sound effects from short-form videos.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@app.command()
def extract(
    videos: list[Path] = typer.Argument(..., help="Video file(s) to process"),
    engine: str = typer.Option(
        None, help="Separation engine: bandit (default), demucs, or none"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-process videos already in the library"
    ),
    force_long: bool = typer.Option(
        False, "--force-long", help="Process videos over the duration cap"
    ),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Extract sound effects from VIDEOS into the library."""
    settings = load_settings()
    if library_dir is not None:
        settings.library_dir = library_dir.expanduser()

    failures = 0
    with LibraryDB(settings.db_path) as db:
        separator = None
        for video in videos:
            if video.suffix.lower() not in SUPPORTED_EXTENSIONS:
                err_console.print(
                    f"[yellow]skipping {video.name}: unsupported extension[/]"
                )
                continue
            console.print(f"[bold]{video.name}[/]")
            try:
                if separator is None:
                    separator = get_separator(settings, engine)
                result = extract_video(
                    video, settings,
                    force=force, max_duration_override=force_long,
                    progress=lambda msg: console.print(f"  [dim]{msg}[/]"),
                    separator=separator, db=db,
                )
            except (IngestError, PipelineError, SeparationError) as exc:
                failures += 1
                err_console.print(f"  [red]error:[/] {exc}")
                continue

            if result.already_processed:
                continue
            for path in result.effect_paths:
                console.print(f"  [green]+[/] {path}")
            summary = f"  [bold green]{result.effect_count} effect(s) saved[/]"
            details = [
                f"{n} rejected ({reason})" for reason, n in result.rejected.items()
            ]
            if result.duplicates:
                details.append(f"{result.duplicates} duplicate(s) skipped")
            if details:
                summary += f" [dim]— {', '.join(details)}[/]"
            console.print(summary)

    raise typer.Exit(code=1 if failures else 0)


@app.command()
def setup(
    engine: str = typer.Argument("bandit", help="Engine to set up: bandit or demucs"),
):
    """Download and install a separation engine (one-time)."""
    settings = load_settings()
    try:
        separator = get_separator(settings, engine)
        if separator.is_ready():
            console.print(f"[green]'{engine}' is already set up.[/]")
            return
        separator.ensure_ready()
        console.print(f"[green]'{engine}' is ready.[/]")
    except SeparationError as exc:
        err_console.print(f"[red]setup failed:[/] {exc}")
        raise typer.Exit(code=1)


@app.command()
def library(
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """List effects in the library."""
    settings = load_settings()
    if library_dir is not None:
        settings.library_dir = library_dir.expanduser()

    with LibraryDB(settings.db_path) as db:
        rows = db.list_effects()
    if not rows:
        console.print("Library is empty. Run: wavelength extract <video>")
        return

    table = Table(title=f"Sound Effects Library ({len(rows)})")
    table.add_column("file")
    table.add_column("duration", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("source")
    table.add_column("added")
    for row in rows:
        table.add_row(
            row["path"],
            f"{row['duration_s']:.2f}s",
            f"{row['sample_rate'] // 1000}k",
            row["source_filename"],
            row["created_at"][:10],
        )
    console.print(table)


@app.command()
def info():
    """Show configuration and engine readiness."""
    settings = load_settings()
    console.print(f"wavelength [bold]{__version__}[/]")
    console.print(f"  library:  {settings.library_dir}")
    console.print(f"  cache:    {settings.cache_dir}")
    console.print(f"  engine:   {settings.engine} (device: {settings.device})")
    console.print(f"  archive sources: {settings.archive_sources}")
    for name in ("bandit", "demucs", "none"):
        try:
            ready = get_separator(settings, name).is_ready()
        except SeparationError:
            ready = False
        state = "[green]ready[/]" if ready else "[yellow]not set up[/]"
        console.print(f"  engine '{name}': {state}")


if __name__ == "__main__":
    app()
