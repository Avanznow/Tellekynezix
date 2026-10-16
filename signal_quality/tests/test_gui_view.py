"""GUI quality panel: view models and the real GUI5.py slots, without Qt, hardware or a display."""

from types import SimpleNamespace

import pytest

from signal_quality import QualityConfig, QualityGatedPredictor, withheld_result
from signal_quality.gui_view import preflight_view, quality_view
from signal_quality.report import preflight

from test_quality import backend, capture, gui_methods
from test_report import write_recording


def test_withheld_capture_shows_failing_channel_and_plain_reasons():
    sample = capture()
    sample["raw"][15] = -187500.02235174447
    result = QualityGatedPredictor().predict(predict=lambda raw: pytest.fail("must not predict"), **sample)
    view = quality_view(result)
    assert view["state"] == "withheld" and view["headline"].startswith("Prediction withheld")
    assert any("ADC rail" in r for r in view["reasons"]) and any("Flat signal" in r for r in view["reasons"])
    tiles = {c["label"]: c for c in view["channels"]}
    assert len(tiles) == 16 and tiles["Ch 15"]["status"] == "fail" and "ADC rail" in tiles["Ch 15"]["detail"]
    assert all(tiles[f"Ch {i}"]["status"] == "pass" for i in range(15))
    assert view["summary"] == "15/16 channels pass"


def test_accepted_and_contract_failures_are_distinguished():
    accepted = QualityGatedPredictor().predict(predict=lambda raw: "forward", **capture())
    view = quality_view(accepted)
    assert view["state"] == "accepted" and view["reasons"] == [] and view["summary"] == "16/16 channels pass"
    # Withheld before any measurement (e.g. unsupported headset): no invented channel tiles.
    view = quality_view(withheld_result("Neurosity quality contract needs raw timestamps"))
    assert view["state"] == "withheld" and view["channels"] == []
    assert view["reasons"] == ["Neurosity quality contract needs raw timestamps"]
    assert quality_view(withheld_result("no_assessed_input"))["state"] == "idle"
    assert quality_view(withheld_result("assessment_pending"))["state"] == "pending"


def test_preflight_view_grades_channels_and_lists_actions(tmp_path):
    result = preflight(write_recording(tmp_path / "rec.txt", dead=(4,), drop_packets=(100,)), QualityConfig())
    view = preflight_view(result)
    tiles = {c["label"]: c for c in view["channels"]}
    assert tiles["Ch 4"]["status"] == "fail" and "100% of windows" in tiles["Ch 4"]["detail"]
    assert tiles["Ch 0"]["status"] == "pass"
    assert view["state"] == "withheld" and view["reasons"][0].startswith("Usable without dead electrodes")
    assert "(Ch 4)" in view["reasons"][0]
    assert any("Re-seat" in r for r in view["reasons"]) and any("start-up" in r for r in view["reasons"])
    assert preflight_view(dict(path="x.txt", error="Empty recording"))["reasons"] == ["Empty recording"]


def test_read_my_mind_publishes_the_panel_for_every_decision():
    methods = gui_methods()
    b = backend(withheld_result("flat_channel; adc_saturation"))
    methods["readMyMind"](b)
    [view] = b.qualityViewChanged.values
    assert view["state"] == "withheld" and len(view["reasons"]) == 2
    accepted = QualityGatedPredictor().predict(predict=lambda raw: "left", **capture())
    b = backend(accepted)
    methods["readMyMind"](b)
    assert b.qualityViewChanged.values[-1]["state"] == "accepted" and b.actions == ["left"]


def _slot_backend(methods, chosen):
    b = backend(withheld_result("no_assessed_input"))
    b._choose_recording = lambda title: chosen
    return b


def test_check_recording_slot_shows_preflight_in_the_panel(tmp_path):
    methods = gui_methods()
    source = write_recording(tmp_path / "rec.txt", dead=(2,))
    b = _slot_backend(methods, str(source))
    methods["checkRecordingQuality"](b)
    [view] = b.qualityViewChanged.values
    assert {c["label"]: c["status"] for c in view["channels"]}["Ch 2"] == "fail"
    assert any("Re-seat" in line for line in b.logMessage.values)
    bad = tmp_path / "bad.txt"
    bad.write_text("nope\n")
    b = _slot_backend(methods, str(bad))
    methods["checkRecordingQuality"](b)
    assert b.qualityViewChanged.values[-1]["headline"] == "Recording could not be checked"
    b = _slot_backend(methods, None)  # dialog cancelled
    methods["checkRecordingQuality"](b)
    assert b.qualityViewChanged.values == []


def test_dashboard_slot_builds_and_opens_the_file(tmp_path, monkeypatch):
    methods = gui_methods()
    opened = []
    methods["QDesktopServices"] = SimpleNamespace(openUrl=opened.append)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    source = write_recording(tmp_path / "rec.txt")
    b = _slot_backend(methods, str(source))
    methods["openQualityDashboard"](b)
    out = tmp_path / "avatar_quality_dashboards" / "rec.quality.html"
    assert out.exists() and opened == [("file", str(out))]
    assert "4/4 windows pass" in b.logMessage.values[-1]
    bad = tmp_path / "bad.txt"
    bad.write_text("nope\n")
    b = _slot_backend(methods, str(bad))
    methods["openQualityDashboard"](b)
    assert "not built" in b.logMessage.values[-1] and len(opened) == 1


def test_preflight_grid_ignores_the_startup_window(tmp_path):
    view = preflight_view(preflight(write_recording(tmp_path / "rec.txt", startup_spike=True), QualityConfig()))
    assert {c["status"] for c in view["channels"]} == {"pass"}
    assert view["summary"] == "16/16 channels clean after start-up"
    assert any("start-up" in r for r in view["reasons"])
