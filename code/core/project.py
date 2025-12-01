from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Optional, List
import uuid

STUDIO_DIR = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = STUDIO_DIR / "projects"
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

@dataclass
class ProjectManifest:
    project_id: str
    name: str
    use_case: str
    tz: str = "UTC"
    created_by: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())
    last_opened: Optional[str] = None
    paths: dict | None = None
    status: dict | None = None
    provenance: dict | None = None
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

def _slug(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name.strip())
    return s[:64] or "project"

def create_project(name: str, use_case: str, created_by: Optional[str]) -> Path:
    folder = PROJECTS_ROOT / _slug(name)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest(
        project_id=str(uuid.uuid4()),
        name=name,
        use_case=use_case,
        created_by=created_by or "user",
        paths={
            "root": str(folder),
            "data_raw": "data_raw/",
            "data_normalised": "data_normalised/",
            "metadata": "metadata/",
            "exports": "exports/",
            "logs": "logs/",
        },
        status={
            "import_results": "empty",
            "audio_resolver": "empty",
            "metadata_joins": "empty",
            "analysis": "blocked",
            "export": "blocked",
        },
        provenance={"app": "pamalytics_studio", "version": "0.1.0"},
    )
    for p in manifest.paths.values():
        (folder / p).mkdir(parents=True, exist_ok=True)
    (folder / "project.json").write_text(manifest.to_json(), encoding="utf-8")
    return folder

def list_projects() -> List[Path]:
    if not PROJECTS_ROOT.exists():
        return []
    return sorted(
        [p for p in PROJECTS_ROOT.iterdir() if (p / "project.json").exists()],
        key=lambda p: p.stat().st_mtime, reverse=True
    )

def load_project(folder: Path) -> dict:
    return json.loads((folder / "project.json").read_text(encoding="utf-8"))

def touch_last_opened(folder: Path) -> None:
    data = load_project(folder)
    data["last_opened"] = datetime.now(dt_timezone.utc).isoformat()
    (folder / "project.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
