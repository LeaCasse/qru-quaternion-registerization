from __future__ import annotations

import importlib


def test_active_modules_import_without_side_effects():
    modules = [
        "qru_registerization.amplitude_interface",
        "qru_registerization.bloch",
        "qru_registerization.fixed_point",
        "qru_registerization.gates",
        "qru_registerization.io",
        "qru_registerization.pipeline",
        "qru_registerization.quaternion_diagnostics",
        "qru_registerization.quaternion_geometry",
        "qru_registerization.transpile_config",
    ]
    for module in modules:
        importlib.import_module(module)
