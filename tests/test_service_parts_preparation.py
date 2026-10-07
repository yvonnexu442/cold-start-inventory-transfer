from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts/generate_service_parts_checkpoints.py"
    spec = importlib.util.spec_from_file_location("service_parts_preparation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _archive(path: Path, *, include_man: bool = True) -> None:
    prefix = "Spare-Part-Demand-Forecasting-main/All Data sets"
    with zipfile.ZipFile(path, "w") as handle:
        if include_man:
            handle.writestr(f"{prefix}/MAN.xlsx", b"man-workbook")
        handle.writestr(f"{prefix}/BRAF.xls", b"braf-workbook")


def test_prepare_extracts_and_is_idempotent(tmp_path: Path) -> None:
    module = _load_script()
    archive = tmp_path / "source.zip"
    destination = tmp_path / "interim"
    _archive(archive)
    first = module.prepare_source_workbooks(archive, destination)
    second = module.prepare_source_workbooks(archive, destination)
    assert first == second
    assert Path(first["MAN.xlsx"]).read_bytes() == b"man-workbook"
    assert Path(first["BRAF.xls"]).read_bytes() == b"braf-workbook"


def test_prepare_refuses_different_existing_content(tmp_path: Path) -> None:
    module = _load_script()
    archive = tmp_path / "source.zip"
    destination = tmp_path / "interim"
    _archive(archive)
    paths = module.prepare_source_workbooks(archive, destination)
    Path(paths["MAN.xlsx"]).write_bytes(b"different")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.prepare_source_workbooks(archive, destination)


def test_prepare_reports_missing_archive_and_workbook(tmp_path: Path) -> None:
    module = _load_script()
    with pytest.raises(FileNotFoundError, match="source archive not found"):
        module.prepare_source_workbooks(tmp_path / "missing.zip", tmp_path / "out")
    archive = tmp_path / "source.zip"
    _archive(archive, include_man=False)
    with pytest.raises(FileNotFoundError, match="MAN.xlsx is absent"):
        module.prepare_source_workbooks(archive, tmp_path / "out")
