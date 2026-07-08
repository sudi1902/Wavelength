"""Wavelength CLI.

    wavelength extract <file-or-url> ...            # the core workflow
    wavelength setup bandit|clap|demucs             # one-time model installs
    wavelength library [list|search|review|...]     # manage the collection
    wavelength info                                 # config + engine status
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from wavelength import __version__
from wavelength.config import SUPPORTED_EXTENSIONS, Settings, load_settings
from wavelength.library.db import LibraryDB
from wavelength.pipeline.download import DownloadError, is_url
from wavelength.pipeline.ingest import IngestError
from wavelength.pipeline.label import ClapLabeler, LabelingError
from wavelength.pipeline.runner import PipelineError, extract_url, extract_video
from wavelength.pipeline.separate import SeparationError, get_separator

app = typer.Typer(
    name="wavelength",
    help="Extract individual sound effects from short-form videos.",
    no_args_is_help=True,
)
library_app = typer.Typer(help="Browse and manage the sound effects library.")
app.add_typer(library_app, name="library")

console = Console()
err_console = Console(stderr=True)


def _settings(library_dir: Path | None = None) -> Settings:
    settings = load_settings()
    if library_dir is not None:
        settings.library_dir = library_dir.expanduser()
    return settings


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------

@app.command()
def extract(
    videos: list[str] = typer.Argument(
        ..., help="Video file(s) and/or URLs (TikTok, Instagram, ...) to process"
    ),
    engine: str = typer.Option(
        None, help="Separation engine: bandit (default), demucs, or none"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-process videos already in the library"
    ),
    force_long: bool = typer.Option(
        False, "--force-long", help="Process videos over the duration cap"
    ),
    browser: str = typer.Option(
        None,
        help="Borrow login cookies from this browser for URL downloads "
        "(chrome, safari, firefox, edge) — needed for most Instagram links",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Show detection internals: gate levels and why each candidate "
        "segment was kept or rejected",
    ),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Extract sound effects from VIDEOS (files or URLs) into the library."""
    settings = _settings(library_dir)

    failures = 0
    with LibraryDB(settings.db_path) as db:
        separator = None
        for target in videos:
            target_is_url = is_url(target)
            video = Path(target)
            if not target_is_url and video.suffix.lower() not in SUPPORTED_EXTENSIONS:
                err_console.print(
                    f"[yellow]skipping {video.name}: unsupported extension[/]"
                )
                continue
            console.print(f"[bold]{target if target_is_url else video.name}[/]")
            try:
                if separator is None:
                    separator = get_separator(settings, engine)
                progress = lambda msg: console.print(f"  [dim]{msg}[/]")  # noqa: E731
                kwargs = dict(
                    force=force, max_duration_override=force_long,
                    progress=progress, separator=separator, db=db, debug=debug,
                )
                if target_is_url:
                    result = extract_url(target, settings, browser=browser, **kwargs)
                else:
                    result = extract_video(video, settings, **kwargs)
            except (IngestError, PipelineError, SeparationError, DownloadError) as exc:
                failures += 1
                err_console.print(f"  [red]error:[/] {exc}")
                continue

            if result.already_processed:
                continue
            for path in result.effect_paths:
                console.print(f"  [green]+[/] {path.name}")
            summary = f"  [bold green]{result.effect_count} effect(s) saved[/]"
            details = [
                f"{n} rejected ({reason})" for reason, n in result.rejected.items()
            ]
            if result.quarantined_paths:
                details.append(
                    f"{len(result.quarantined_paths)} quarantined "
                    "(review: wavelength library review)"
                )
            if result.duplicates:
                details.append(f"{result.duplicates} duplicate(s) skipped")
            if details:
                summary += f" [dim]— {', '.join(details)}[/]"
            console.print(summary)

    raise typer.Exit(code=1 if failures else 0)


# --------------------------------------------------------------------------
# setup / info
# --------------------------------------------------------------------------

@app.command()
def setup(
    engine: str = typer.Argument(
        "bandit", help="What to set up: bandit, clap, or demucs"
    ),
):
    """Download and install a model (one-time per engine)."""
    settings = load_settings()
    try:
        if engine == "clap":
            labeler = ClapLabeler(settings)
            if labeler.is_ready():
                console.print("[green]'clap' is already set up.[/]")
                return
            labeler.ensure_ready()
        else:
            separator = get_separator(settings, engine)
            if separator.is_ready():
                console.print(f"[green]'{engine}' is already set up.[/]")
                return
            separator.ensure_ready()
        console.print(f"[green]'{engine}' is ready.[/]")
    except (SeparationError, LabelingError) as exc:
        err_console.print(f"[red]setup failed:[/] {exc}")
        raise typer.Exit(code=1)


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
    clap_state = (
        "[green]ready[/]" if ClapLabeler(settings).is_ready()
        else "[yellow]not set up[/] (wavelength setup clap)"
    )
    console.print(f"  labeler 'clap': {clap_state}")


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------

def _label_of(row) -> str:
    return row["user_label"] or row["auto_label"] or "-"


def _source_of(row) -> str:
    source = row["source_title"] or row["source_filename"]
    if row["source_uploader"]:
        source = f"{source} (@{row['source_uploader']})"
    return source


def _effects_table(rows, title: str, extra_col: str | None = None) -> Table:
    table = Table(title=title)
    table.add_column("id", justify="right")
    table.add_column("label")
    table.add_column("file")
    table.add_column("dur", justify="right")
    table.add_column("tags")
    table.add_column("source")
    if extra_col:
        table.add_column(extra_col)
    for row in rows:
        cells = [
            str(row["id"]),
            _label_of(row),
            Path(row["path"]).name,
            f"{row['duration_s']:.2f}s",
            row["tags"] or "-",
            _source_of(row),
        ]
        if extra_col:
            cells.append(row["quarantine_reason"] or "-")
        table.add_row(*cells)
    return table


@library_app.callback(invoke_without_command=True)
def library_default(ctx: typer.Context):
    """With no subcommand, list the library."""
    if ctx.invoked_subcommand is None:
        # Explicit None: a bare call would leak typer's OptionInfo default.
        list_effects(library_dir=None)


@library_app.command("list")
def list_effects(
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """List all effects in the library."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        rows = db.list_effects()
    if not rows:
        console.print("Library is empty. Run: wavelength extract <video-or-url>")
        return
    console.print(_effects_table(rows, f"Sound Effects Library ({len(rows)})"))


@library_app.command()
def search(
    query: str = typer.Argument(..., help="Text to match against labels, "
                                "tags, filenames, and source titles"),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Search the library."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        rows = db.search_effects(query)
    if not rows:
        console.print(f"No effects match '{query}'.")
        return
    console.print(_effects_table(rows, f"Matches for '{query}' ({len(rows)})"))


@library_app.command()
def review(
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """List quarantined effects (suspected speech/music bleed)."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        rows = db.list_effects(status="quarantine")
    if not rows:
        console.print("Quarantine is empty.")
        return
    console.print(
        _effects_table(rows, f"Quarantine ({len(rows)})", extra_col="reason")
    )
    console.print(
        "[dim]Promote keepers: wavelength library promote <id> — "
        "delete junk: wavelength library delete <id>[/]"
    )


def _get_effect_or_exit(db: LibraryDB, effect_id: int):
    row = db.get_effect(effect_id)
    if row is None:
        err_console.print(f"[red]No effect with id {effect_id}[/]")
        raise typer.Exit(code=1)
    return row


@library_app.command()
def promote(
    effect_id: int = typer.Argument(..., help="Effect id (see: library review)"),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Move a quarantined effect into the library."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        row = _get_effect_or_exit(db, effect_id)
        if row["status"] != "quarantine":
            console.print("Effect is already in the library.")
            return
        src = settings.library_dir / row["path"]
        rel = Path(row["path"])
        # quarantine/YYYY/MM/x.wav -> effects/YYYY/MM/x.wav
        dest_rel = Path("effects").joinpath(*rel.parts[1:])
        dest = settings.library_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        db.set_status(effect_id, "library", path=str(dest_rel))
    console.print(f"[green]Promoted[/] {dest.name}")


@library_app.command()
def rename(
    effect_id: int = typer.Argument(..., help="Effect id (see: library list)"),
    label: str = typer.Argument(..., help="New label"),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Set your own label on an effect (file path stays stable)."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        _get_effect_or_exit(db, effect_id)
        db.set_user_label(effect_id, label)
    console.print(f"[green]Renamed[/] effect {effect_id} → '{label}'")


@library_app.command()
def tag(
    effect_id: int = typer.Argument(..., help="Effect id (see: library list)"),
    tags: str = typer.Argument(..., help="Comma-separated tags"),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Set tags on an effect (comma-separated, replaces existing tags)."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        _get_effect_or_exit(db, effect_id)
        db.set_tags(effect_id, tags)
    console.print(f"[green]Tagged[/] effect {effect_id}: {tags}")


@library_app.command()
def delete(
    effect_id: int = typer.Argument(..., help="Effect id (see: library list)"),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Delete an effect (file and record)."""
    settings = _settings(library_dir)
    with LibraryDB(settings.db_path) as db:
        row = _get_effect_or_exit(db, effect_id)
        (settings.library_dir / row["path"]).unlink(missing_ok=True)
        db.delete_effect(effect_id)
    console.print(f"[green]Deleted[/] effect {effect_id} ({Path(row['path']).name})")


@library_app.command()
def export(
    effect_ids: list[int] = typer.Argument(..., help="Effect id(s) to export"),
    dest: Path = typer.Option(
        Path("."), help="Destination directory (default: current dir)"
    ),
    mp3: bool = typer.Option(False, "--mp3", help="Export as 320 kbps MP3"),
    library_dir: Path = typer.Option(None, help="Override library location"),
):
    """Copy effects out of the library as WAV (default) or MP3."""
    settings = _settings(library_dir)
    dest = dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    with LibraryDB(settings.db_path) as db:
        for effect_id in effect_ids:
            row = _get_effect_or_exit(db, effect_id)
            src = settings.library_dir / row["path"]
            name = row["user_label"] or Path(row["path"]).stem
            if mp3:
                out = dest / f"{name}.mp3"
                result = subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-i", str(src),
                     "-c:a", "libmp3lame", "-b:a", "320k", str(out)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    err_console.print(
                        f"[red]MP3 encode failed for {src.name}:[/] "
                        f"{result.stderr.strip()[-200:]}"
                    )
                    raise typer.Exit(code=1)
            else:
                out = dest / f"{name}.wav"
                shutil.copy2(src, out)
            console.print(f"[green]Exported[/] {out}")


if __name__ == "__main__":
    app()
