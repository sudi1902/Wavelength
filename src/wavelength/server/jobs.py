"""Background extraction jobs for the server.

A single worker thread processes jobs sequentially — separation is
resource-heavy, and one model run at a time is the right behavior on a
laptop. Progress lines stream into the job record so the UI can poll and
render live status. The separator instance is reused across jobs.
"""

from __future__ import annotations

import itertools
import queue
import threading
from dataclasses import dataclass, field

from wavelength.config import Settings
from wavelength.pipeline.download import is_url
from wavelength.pipeline.runner import extract_url, extract_video
from wavelength.pipeline.separate import get_separator


@dataclass
class Job:
    id: int
    target: str  # URL or file path
    status: str = "queued"  # queued | running | done | error
    log: list[str] = field(default_factory=list)
    error: str | None = None
    effect_count: int = 0
    quarantined: int = 0
    already_processed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target,
            "status": self.status,
            "log": self.log[-30:],
            "error": self.error,
            "effect_count": self.effect_count,
            "quarantined": self.quarantined,
            "already_processed": self.already_processed,
        }


class JobQueue:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: dict[int, Job] = {}
        self._queue: queue.Queue[int] = queue.Queue()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._separator = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, target: str, *, force: bool = False) -> Job:
        job = Job(id=next(self._ids), target=target)
        job.force = force  # type: ignore[attr-defined]
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)
        return jobs[:limit]

    def _run(self) -> None:
        from pathlib import Path

        while True:
            job_id = self._queue.get()
            job = self._jobs[job_id]
            job.status = "running"
            progress = job.log.append
            try:
                if self._separator is None:
                    self._separator = get_separator(self.settings)
                force = getattr(job, "force", False)
                if is_url(job.target):
                    result = extract_url(
                        job.target, self.settings, force=force,
                        progress=progress, separator=self._separator,
                    )
                else:
                    result = extract_video(
                        Path(job.target), self.settings, force=force,
                        progress=progress, separator=self._separator,
                    )
                job.effect_count = result.effect_count
                job.quarantined = len(result.quarantined_paths)
                job.already_processed = result.already_processed
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - job errors go to the UI
                job.error = str(exc)
                job.status = "error"
