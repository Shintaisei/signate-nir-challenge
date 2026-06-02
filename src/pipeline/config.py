"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompetitionConfig:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    submissions_dir: Path
    outputs_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    competition_key: str
    task_key: str
    encoding: str
    id_col: str
    target_col: str
    meta_cols: tuple[str, ...]


def load_config() -> CompetitionConfig:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    raw_dir = data_dir / "raw"

    return CompetitionConfig(
        project_root=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=data_dir / "processed",
        submissions_dir=data_dir / "submissions",
        outputs_dir=root / "outputs",
        train_path=raw_dir / "train.csv",
        test_path=raw_dir / "test.csv",
        sample_submission_path=raw_dir / "sample_submit.csv",
        competition_key="37308d147238487c96551300b8e4cb76",
        task_key="8940dcfa70434a6aaaa28d661652d536",
        encoding="cp932",
        id_col="sample number",
        target_col="含水率",
        meta_cols=("sample number", "樹種", "species number"),
    )
