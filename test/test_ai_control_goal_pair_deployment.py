import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
AI_CONTROL = REPO / "ai-control-260729"


def _load_plugin(monkeypatch):
    node_sdk = types.ModuleType("node_sdk")

    class CommandStep:
        def __init__(self, *args):
            self.args = args

    node_sdk.CommandStep = CommandStep
    csv_debug = types.ModuleType("ai_control_node.csv_debug")

    class CsvLogger:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, *args, **kwargs):
            pass

    csv_debug.CsvLogger = CsvLogger
    monkeypatch.syspath_prepend(str(AI_CONTROL))
    monkeypatch.syspath_prepend(str(AI_CONTROL / "ai_models"))
    monkeypatch.setitem(sys.modules, "node_sdk", node_sdk)
    monkeypatch.setitem(sys.modules, "ai_control_node", types.ModuleType("ai_control_node"))
    monkeypatch.setitem(sys.modules, "ai_control_node.csv_debug", csv_debug)
    sys.modules.pop("ai_models.plugins.run_postech_docking_demo", None)
    return importlib.import_module("ai_models.plugins.run_postech_docking_demo")


def _pair(source):
    return {"encoder": "reloc3r_goal_pair", "source": source}


def test_camera_plan_orbbec_only_never_requires_usb(monkeypatch):
    plugin = _load_plugin(monkeypatch)
    plan = plugin._token_camera_plan({
        "wheel": {"encoder": "motion", "source": "encoder"},
        "reloc3r_pair_bottom": _pair("reloc3r_bottom"),
    })
    assert plan == {
        "use_orbbec": True,
        "use_usb": False,
        "need_orbbec_goal": True,
        "need_usb_goal": False,
    }
    assert plugin._validate_camera_mode(
        SimpleNamespace(demo_camera_mode="orbbec"), plan["use_usb"]
    ) == "orbbec"


def test_camera_plan_two_camera_requires_both_goals(monkeypatch):
    plugin = _load_plugin(monkeypatch)
    plan = plugin._token_camera_plan({
        "wheel": {"encoder": "motion", "source": "encoder"},
        "reloc3r_pair_bottom": _pair("reloc3r_bottom"),
        "reloc3r_pair_top": _pair("reloc3r_top"),
    })
    assert plan == {
        "use_orbbec": True,
        "use_usb": True,
        "need_orbbec_goal": True,
        "need_usb_goal": True,
    }
    assert plugin._validate_camera_mode(
        SimpleNamespace(demo_camera_mode="orbbec_usb"), plan["use_usb"]
    ) == "orbbec_usb"
    assert Path(plugin.GOAL_IMAGE_ORBBEC0_DEFAULT).is_file()
    assert Path(plugin.GOAL_IMAGE_USB0_DEFAULT).is_file()


def test_camera_mode_rejects_checkpoint_mismatch(monkeypatch):
    plugin = _load_plugin(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint sensors require 'orbbec_usb'"):
        plugin._validate_camera_mode(
            SimpleNamespace(demo_camera_mode="orbbec"), use_usb0=True)
    with pytest.raises(ValueError, match="checkpoint sensors require 'orbbec'"):
        plugin._validate_camera_mode(
            SimpleNamespace(demo_camera_mode="orbbec_usb"), use_usb0=False)
