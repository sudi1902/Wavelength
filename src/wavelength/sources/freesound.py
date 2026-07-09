"""Freesound.org API client: search Creative Commons sounds and fetch
previews.

Uses token auth (free key from https://freesound.org/apiv2/apply). Token
auth grants search + HQ preview downloads (~128 kbps mp3); full-quality
originals need OAuth2, which is deliberately out of scope for now — an HQ
studio preview still beats TikTok-compressed, separation-processed audio.

Every result carries its license; the caller surfaces attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

API_BASE = "https://freesound.org/apiv2"

# Freesound filter names for the licenses we understand.
LICENSE_FILTERS = {
    "cc0": '"Creative Commons 0"',
    "by": '"Attribution"',
    "by-nc": '"Attribution Noncommercial"',
}

# Ordered: more specific names first — "attribution noncommercial" contains
# "attribution", so NC must match before plain BY.
LICENSE_SHORT = {
    "creative commons 0": "CC0",
    "attribution noncommercial": "CC-BY-NC",
    "attribution": "CC-BY",
}


class FreesoundError(Exception):
    """Freesound request failed; message is user-facing."""


@dataclass
class FreesoundSound:
    id: int
    name: str
    username: str
    license: str  # short form: CC0 / CC-BY / CC-BY-NC
    duration_s: float
    preview_url: str
    page_url: str

    @property
    def attribution(self) -> str:
        return f'"{self.name}" by {self.username} on Freesound.org ({self.license})'


def _short_license(value: str) -> str:
    lowered = value.lower()
    for key, short in LICENSE_SHORT.items():
        if key in lowered:
            return short
    # License may come as a CC URL, e.g. .../licenses/by/4.0/
    if "/zero/" in lowered or "publicdomain" in lowered:
        return "CC0"
    if "/by-nc" in lowered:
        return "CC-BY-NC"
    if "/by/" in lowered:
        return "CC-BY"
    return value


def search(
    api_key: str,
    query: str,
    *,
    min_duration_s: float = 0.05,
    max_duration_s: float = 10.0,
    licenses: list[str] | None = None,
    limit: int = 24,
) -> list[FreesoundSound]:
    if not api_key:
        raise FreesoundError(
            "No Freesound API key configured. Get a free key at "
            "https://freesound.org/apiv2/apply and add it to "
            "~/.config/wavelength/config.toml under [similar]:\n"
            'freesound_api_key = "YOUR_KEY"'
        )
    filters = [f"duration:[{min_duration_s:.2f} TO {max_duration_s:.2f}]"]
    license_names = [
        LICENSE_FILTERS[lic] for lic in (licenses or []) if lic in LICENSE_FILTERS
    ]
    if license_names:
        filters.append(f"license:({' OR '.join(license_names)})")

    try:
        response = requests.get(
            f"{API_BASE}/search/text/",
            params={
                "query": query,
                "filter": " ".join(filters),
                "fields": "id,name,username,license,duration,previews,url",
                "page_size": min(limit, 150),
                "token": api_key,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise FreesoundError(f"Could not reach Freesound: {exc}") from exc
    if response.status_code == 401:
        raise FreesoundError("Freesound rejected the API key (401). Check it.")
    if response.status_code != 200:
        raise FreesoundError(
            f"Freesound search failed (HTTP {response.status_code}): "
            + response.text[:200]
        )

    sounds = []
    for item in response.json().get("results", []):
        previews = item.get("previews") or {}
        preview = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        if not preview:
            continue
        sounds.append(
            FreesoundSound(
                id=item["id"],
                name=item["name"],
                username=item["username"],
                license=_short_license(item.get("license", "")),
                duration_s=float(item.get("duration", 0.0)),
                preview_url=preview,
                page_url=item.get("url", f"https://freesound.org/s/{item['id']}/"),
            )
        )
    return sounds[:limit]


def download_preview(sound: FreesoundSound, dest: Path) -> Path:
    """Download the sound's HQ preview mp3 to ``dest`` (cached: skipped if
    present)."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(sound.preview_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        tmp.rename(dest)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise FreesoundError(
            f"Preview download failed for '{sound.name}': {exc}"
        ) from exc
    return dest
