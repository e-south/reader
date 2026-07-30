from __future__ import annotations

import re
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parents[5] / "domains" / "plate_reader"
ANALYSIS_PACKAGE = DOMAIN_ROOT / "analysis" / "response_window"
PLOT_PACKAGE = DOMAIN_ROOT / "plots" / "response_window"


def test_response_window_modules_stay_bounded() -> None:
    limits = {
        ANALYSIS_PACKAGE / "aggregation.py": 220,
        ANALYSIS_PACKAGE / "contracts.py": 360,
        ANALYSIS_PACKAGE / "event_sensitivity.py": 60,
        ANALYSIS_PACKAGE / "materialize.py": 280,
        ANALYSIS_PACKAGE / "reduction.py": 260,
        ANALYSIS_PACKAGE / "seeds.py": 40,
        ANALYSIS_PACKAGE / "sources.py": 340,
        ANALYSIS_PACKAGE / "observation_resampling.py": 130,
        PLOT_PACKAGE / "diagnostic.py": 320,
        PLOT_PACKAGE / "diagnostic_render.py": 320,
        PLOT_PACKAGE / "schema.py": 40,
        PLOT_PACKAGE / "summary.py": 150,
    }
    observed = {path: len(path.read_text(encoding="utf-8").splitlines()) for path in limits}
    violations = {
        path.relative_to(DOMAIN_ROOT).as_posix(): lines for path, lines in observed.items() if lines > limits[path]
    }
    assert violations == {}


def test_response_window_roles_have_distinct_packages() -> None:
    analysis_modules = {path.name for path in ANALYSIS_PACKAGE.glob("*.py")}

    assert analysis_modules == {
        "__init__.py",
        "aggregation.py",
        "contracts.py",
        "event_sensitivity.py",
        "materialize.py",
        "reduction.py",
        "seeds.py",
        "sources.py",
        "observation_resampling.py",
    }
    assert not (DOMAIN_ROOT / "evidence").exists()
    assert {path.name for path in PLOT_PACKAGE.glob("*.py")} == {
        "__init__.py",
        "diagnostic.py",
        "diagnostic_render.py",
        "schema.py",
        "summary.py",
    }
    assert not (DOMAIN_ROOT.parent / "review.py").exists()


def test_response_window_dependencies_point_outward_from_core() -> None:
    core_source = "\n".join(path.read_text(encoding="utf-8") for path in ANALYSIS_PACKAGE.glob("*.py"))
    plot_source = "\n".join(path.read_text(encoding="utf-8") for path in PLOT_PACKAGE.rglob("*.py"))

    assert "reader_workbench.domains.plate_reader.evidence" not in core_source
    assert "reader_workbench.domains.plate_reader.plots" not in core_source
    assert "reader_workbench.domains.plate_reader.evidence" not in plot_source


def test_response_window_domain_has_no_sfxi_or_workbench_imports() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for package in (ANALYSIS_PACKAGE, PLOT_PACKAGE)
        for path in package.rglob("*.py")
    )
    assert "reader_workbench.domains.logic.sfxi" not in source
    assert "reader_workbench.workbench" not in source


def test_response_window_never_calls_reference_relative_fluorescence_brightness() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for package in (ANALYSIS_PACKAGE, PLOT_PACKAGE)
        for path in package.rglob("*.py")
    )

    assert "BRIGHTNESS" not in source
    assert "brightness" not in source.lower()
    assert "anchored_fluorescence" not in source


def test_response_window_public_analysis_uses_signal_value_vocabulary() -> None:
    source = "\n".join(
        (ANALYSIS_PACKAGE / name).read_text(encoding="utf-8") for name in ("contracts.py", "reduction.py", "sources.py")
    ).lower()

    assert re.search(r"\bfluorescence\b", source) is None
    assert re.search(r"\bratio\b", source) is None
