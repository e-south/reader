from __future__ import annotations

from pathlib import Path

from reader.tests.support import REPO_ROOT

EXPERIMENT_CONFIGS = sorted(REPO_ROOT.glob("experiments/**/config.yaml"))

EXPECTED_FILE_PREFLIGHT_BLOCKERS = {
    "experiments/2025/20250702_sensor_panel_M9_glu/config.yaml": "inputs/metadata.xlsx",
    "experiments/2026/20260313_mono_functional_sponges/config.yaml": "No raw .xlsx files discovered",
    "experiments/2026/20260314_bi_functional_lexA_cpxR_baeR_family_sponges/config.yaml": "No raw .xlsx files discovered",
    "experiments/2026/20260315_bi_functional_sox_family_sponges/config.yaml": "No raw .xlsx files discovered",
    "experiments/2026/202603XX_tetra_functional_sponges/config.yaml": "No raw .xlsx files discovered",
    "experiments/2026/202603XX_tri_functional_sponges/config.yaml": "No raw .xlsx files discovered",
    "experiments/template/config.yaml": "No raw .xlsx files discovered",
}

OPTIONAL_DEPENDENCY_BLOCKERS = {
    "experiments/2026/20260101_cytometer_retron/config.yaml": "flowio is required",
}

END_TO_END_RUNNABLE_CONFIGS = [
    config_path
    for config_path in EXPERIMENT_CONFIGS
    if str(config_path.relative_to(REPO_ROOT)) not in EXPECTED_FILE_PREFLIGHT_BLOCKERS
    and str(config_path.relative_to(REPO_ROOT)) not in OPTIONAL_DEPENDENCY_BLOCKERS
]


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))
