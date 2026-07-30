from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ResearchSettings:
    """Runtime settings for the local Phase 1 research pipeline."""

    start_date: str = "2019-01-01"
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    database_path: Path = PROJECT_ROOT / "research/data/institutional_phase1.sqlite"
    universe_path: Path = RESEARCH_ROOT / "config/validation_universe_v1.csv"
    market_overrides_path: Path = RESEARCH_ROOT / "config/market_overrides_v1.csv"
    output_dir: Path = PROJECT_ROOT / "research/output"
    finmind_token: str | None = field(default_factory=lambda: os.getenv("FINMIND_TOKEN"))
    request_interval_seconds: float = 0.25
    horizons: tuple[int, ...] = (5, 10, 20)
    primary_horizon: int = 10
    label_threshold: float = 0.05

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
