"""Meaningful fault, withholding, adapter and independently labelled training tests.

Generated waveforms are unit fixtures only. Published replay outputs use raw
Data_clean recordings and never treat a detector's own decisions as labels.
"""

import ast
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from signal_quality import QualityAssessor, QualityConfig, QualityGatedPredictor, may_execute
from signal_quality.acquisition import brainflow_capture, file_sha256, read_openbci
from signal_quality.gate import withheld_result
from signal_quality.replay import replay
from signal_quality.train import QualityModel, reviewed_features, train


ROOT = Path(__file__).resolve().parents[2]


def capture(seconds=2, *, fs=125, c=16, start=1000):
    n = int(seconds * fs)
    t = np.arange(n) / fs
    raw = np.stack([-10000 - 100*i + (20+i) * np.sin(2*np.pi*(8+i/5)*t + i/9) for i in range(c)])
    return dict(raw=raw, sfreq=fs, channels=[f"EXG Channel {i}" for i in range(c)], units="uV",
                counters=(2*np.arange(n)) % 256, timestamps=start+t, counter_step=2)


def assess(sample, config=None):
    return QualityAssessor(config).assess(**sample)


def test_good_physical_eeg_features_are_finite_and_raw_is_not_modified():
    sample = capture()
    raw = sample["raw"].copy()
    report = assess(sample)
    assert report["usable"] and report["continuity_checked"] and report["bad_channels"] == []
    assert np.isfinite(list(report["window_features"].values())).all()
    assert "mains_frequency_unverified" in report["warnings"][-1]
    assert len(report["window_features"]) == 39
    np.testing.assert_array_equal(sample["raw"], raw)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fault,reason", [
    ("flat", "flat_channel"), ("rail", "adc_saturation"), ("amplitude", "excessive_amplitude"),
    ("jump", "excessive_sample_jump"), ("nan", "nonfinite_eeg"), ("overflow", "out_of_range_eeg"),
    ("packet", "packet_discontinuity"), ("reversed_time", "nonmonotonic_time"), ("gap", "acquisition_gap"),
    ("short", "insufficient_samples"), ("zero_time", "nonadvancing_time")])
def test_acquisition_faults_withhold_before_predictor_is_called(fault, reason):
    sample = capture()
    if fault == "flat": sample["raw"][3] = -10000
    elif fault == "rail": sample["raw"][3, :10] = 187500
    elif fault == "amplitude": sample["raw"][3] *= 100
    elif fault == "jump": sample["raw"][3, 100] += 700
    elif fault == "nan": sample["raw"][3, 0] = np.nan
    elif fault == "overflow": sample["raw"][3, 0] = 1e308
    elif fault == "packet": sample["counters"][100:] = (sample["counters"][100:] + 2) % 256
    elif fault == "reversed_time": sample["timestamps"][100] = sample["timestamps"][99] - .1
    elif fault == "gap": sample["timestamps"][100:] += 1
    elif fault == "zero_time": sample["timestamps"][:] = 0
    elif fault == "short":
        for key in ("raw", "counters", "timestamps"):
            sample[key] = sample[key][:, :100] if key == "raw" else sample[key][:100]
    called = []
    result = QualityGatedPredictor().predict(predict=lambda raw: called.append(raw) or "forward", **sample)
    assert result["status"] == "withheld" and result["prediction_label"] is None and not called
    assert reason in result["quality"]["reasons"] and not may_execute(result)
    json.dumps(result, allow_nan=False)


def test_line_noise_is_assessed_before_filters_and_only_rejected_when_configured():
    sample = capture()
    t = np.arange(sample["raw"].shape[1])/125
    sample["raw"][0] = -10000 + 100*np.sin(2*np.pi*50*t)
    assert assess(sample)["usable"]
    report = assess(sample, QualityConfig(mains_hz=50))
    assert not report["usable"] and "excessive_line_noise" in report["channel_reasons"]["EXG Channel 0"]


def test_batched_arrival_timestamps_do_not_claim_uniform_sample_spacing():
    sample = capture()
    sample["timestamps"] = 1000 + np.floor(np.arange(250)/10)/12.5
    assert assess(sample)["usable"]
    sample["timestamp_kind"] = "sample"
    assert "sample_clock_rate_mismatch" in assess(sample)["reasons"]


@pytest.mark.parametrize("change", [dict(units="zscore"), dict(channels=["x"]*16), dict(sfreq=0), dict(signal_stage="filtered")])
def test_ambiguous_acquisition_contracts_are_rejected(change):
    sample = capture()
    sample.update(change)
    with pytest.raises(ValueError): assess(sample)


def test_missing_continuity_and_synthetic_sources_cannot_dispatch():
    sample = capture()
    sample.pop("counters")
    result = QualityGatedPredictor().predict(predict=lambda raw: pytest.fail("should not run"), **sample)
    assert result["withholding_reason"] == "continuity_unverified"
    result = QualityGatedPredictor().predict(predict=lambda raw: pytest.fail("should not run"), source_kind="synthetic", **capture())
    assert result["withholding_reason"] == "synthetic_or_unverified_source"


def test_gate_binds_inference_to_immutable_assessed_samples_and_expires():
    clock = [20.0]
    sample = capture()
    expected = sample["raw"].copy()
    gate = QualityGatedPredictor(clock=lambda:clock[0])
    def predictor(raw):
        np.testing.assert_array_equal(raw, expected)
        assert not raw.flags.writeable
        return {"prediction_label":"landing", "confidence":.8}
    result = gate.predict(predict=predictor, captured_monotonic=20, **sample)
    assert result["prediction_label"] == "land" and may_execute(result, now=20)
    assert not may_execute(result, now=30) and not may_execute(result, now=19)
    sample["raw"][:] = 0
    rejected = gate.predict(predict=lambda raw:pytest.fail("must not call"), captured_monotonic=21, **dict(sample, stream_id="new"))
    assert not may_execute(rejected, now=21) and not may_execute(gate.last_result, now=21)


def test_expiration_during_slow_model_and_inference_errors_withhold():
    clock = [20.0]
    gate = QualityGatedPredictor(clock=lambda:clock[0])
    def slow(raw):
        clock[0] = 31
        return "forward"
    assert gate.predict(predict=slow, **capture())["withholding_reason"] == "input_expired_during_inference"
    def fail(raw): raise RuntimeError("missing model")
    assert gate.predict(predict=fail, **capture())["status"] == "withheld"
    assert gate.predict(predict=lambda raw:"Error", **capture())["status"] == "withheld"


def test_recovery_requires_new_good_captures_and_resets_on_bad_input():
    clock = [20.0]
    gate = QualityGatedPredictor(QualityAssessor(QualityConfig(recovery_good_windows=2)), clock=lambda:clock[0])
    def run(): return gate.predict(predict=lambda raw:"forward", stream_id="continuous-stream", captured_monotonic=clock[0], **capture())
    assert run()["withholding_reason"] == "awaiting_quality_recovery"
    assert run()["withholding_reason"] == "repeated_or_reversed_capture"
    clock[0] += 1
    assert run()["status"] == "withheld"  # Repeated capture reset the streak.
    clock[0] += 1
    assert run()["status"] == "accepted"
    gate.withhold("gap", stream_id="continuous-stream")
    clock[0] += 1
    assert run()["status"] == "withheld"


def test_board_adapter_excludes_auxiliary_rows_and_checks_schema():
    sample = capture()
    data = np.zeros((20,250))
    data[1:17] = sample["raw"]
    data[0] = sample["counters"]
    data[18] = sample["timestamps"]
    descriptor = dict(num_rows=20, eeg_channels=list(range(1,17)), package_num_channel=0, timestamp_channel=18, sampling_rate=125)
    actual = brainflow_capture(data, descriptor, real=True, counter_step=2)
    np.testing.assert_array_equal(actual["raw"], sample["raw"])
    assert actual["raw"].shape == (16,250)
    with pytest.raises(ValueError): brainflow_capture(data, dict(descriptor,eeg_channels=[0,1]), real=True,counter_step=2)


def _write_source(path, bad=False, *, start=1000):
    import csv
    sample = capture(seconds=4, start=start)
    if bad: sample["raw"][0] *= 100
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="") as stream:
        stream.write("%OpenBCI Raw EXG Data\n%Number of channels = 16\n%Sample Rate = 125 Hz\n%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n")
        writer = csv.writer(stream)
        writer.writerow(["Sample Index", *sample["channels"], "Timestamp", "Marker Channel"])
        for i in range(500): writer.writerow([sample["counters"][i],*sample["raw"][:,i],sample["timestamps"][i],0])


def test_real_schema_replay_has_no_invented_quality_ground_truth(tmp_path):
    source = tmp_path/"raw"/"G01"/"person01"/"unlabelled"/"capture.txt"
    _write_source(source)
    before = file_sha256(source)
    out = tmp_path/"replay"
    summary = replay(tmp_path/"raw",out)
    assert summary["counts"]["windows_assessed"] == 2 and summary["artifact_detection_accuracy"] is None
    assert file_sha256(source) == before
    rows = [json.loads(line) for line in (out/"windows.jsonl").read_text().splitlines()]
    assert all(row["reviewed_quality_label"] is None and not row["label_verified"] for row in rows)
    with pytest.raises(ValueError,match="independent verified"):
        reviewed_features(out/"windows.jsonl", QualityConfig())


def test_independently_reviewed_training_is_grouped_and_artifact_reloads(tmp_path):
    root = tmp_path/"raw"
    for person in range(1,10):
        for bad in (False,True):
            _write_source(root/"G01"/f"person{person:02d}"/"quality"/f"{bad}.txt",bad,start=1000+person*100+int(bad)*10)
    out = tmp_path/"replay"
    replay(root,out)
    rows = [json.loads(line) for line in (out/"windows.jsonl").read_text().splitlines()]
    evidence = tmp_path/"independent_review.txt"
    evidence.write_text("Unit-fixture independent reviewer classifications; not a production annotation")
    for row in rows:
        row.update(label_verified=True,reviewed_quality_label=int(Path(row["source_path"]).name=="False.txt"),
                   reviewed_by="Fixture reviewer",reviewed_at_utc="2026-09-30T20:00:00Z",
                   annotation_evidence=dict(path=str(evidence),sha256=file_sha256(evidence),locator=row["window_id"]))
    annotations = tmp_path/"reviewed.jsonl"
    annotations.write_text("".join(json.dumps(row)+"\n" for row in rows))
    summary = train(annotations,tmp_path/"models")
    assert len(summary["folds"])==3
    for fold in summary["folds"]:
        partitions = fold["split_participants"]
        assert not set(partitions["train"]) & set(partitions["test"])
        assert not set(partitions["validation"]) & set(partitions["test"])
    model = QualityModel.load(tmp_path/"models"/"fold_0"/"quality_model.joblib")
    report = QualityAssessor(model.config,quality_model=model).assess(**capture())
    assert report["learned_usable_probability"] is not None
    evidence.write_text("changed review")
    with pytest.raises(ValueError,match="evidence changed"): reviewed_features(annotations,QualityConfig())


class Signal:
    def __init__(self): self.values=[]
    def emit(self,value): self.values.append(value)


def gui_methods():
    # Test actual GUI method bodies without creating a Qt app, headset, network
    # connection or Tello instance. Hardware constructors are never run.
    tree = ast.parse((ROOT/"GUI5.py").read_text())
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="BrainwavesBackend")
    functions = []
    for node in cls.body:
        if isinstance(node,ast.FunctionDef) and node.name in ("readMyMind","executeAction","_queue_action","_queue_prediction","_drone_loop",
                                                               "_choose_recording","openQualityDashboard","checkRecordingQuality"):
            node.decorator_list=[]
            functions.append(node)
    from signal_quality.dashboard import build as build_quality_dashboard
    from signal_quality.gui_view import preflight_view, quality_view
    from signal_quality.report import preflight as preflight_recording
    import tempfile
    namespace = dict(may_execute=may_execute,withheld_result=withheld_result,quality_view=quality_view,
                     preflight_view=preflight_view,preflight_recording=preflight_recording,
                     build_quality_dashboard=build_quality_dashboard,Path=Path,tempfile=tempfile,
                     QUrl=SimpleNamespace(fromLocalFile=lambda path:("file",path)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),"GUI5.py","exec"),namespace)
    return namespace


def backend(result):
    b = SimpleNamespace(current_model="Random Forest",current_framework="PyTorch",current_prediction_label="old-forward",
                        current_prediction_result=result,quality_gate=QualityGatedPredictor(),flight_log=[],predictions_log=[],actions=[])
    for name in ("signalQualityChanged","qualityViewChanged","logMessage","flightLogUpdated","predictionsTableUpdated"): setattr(b,name,Signal())
    b.run_random_forest_pytorch=lambda:result
    b._queue_prediction=lambda result:b.actions.append(result["prediction_label"])
    return b


def test_gui_withheld_result_clears_old_prediction_and_never_dispatches():
    methods=gui_methods()
    b=backend(withheld_result("flat_channel"))
    methods["readMyMind"](b)
    assert b.actions==[] and b.current_prediction_label==""
    assert b.predictions_log[-1]["label"]=="WITHHELD"
    methods["executeAction"](b)
    assert b.actions==[]


def test_gui_accepted_result_dispatches_and_expired_result_does_not():
    methods=gui_methods()
    result=QualityGatedPredictor().predict(predict=lambda raw:"forward",**capture())
    b=backend(result)
    methods["readMyMind"](b)
    assert b.actions==["forward"]
    b.current_prediction_result["expires_monotonic"]=0
    methods["executeAction"](b)
    assert b.actions==["forward"] and b.current_prediction_label==""


def client_class():
    # Exercise actual controller methods with SDK constructors/logging replaced.
    # No test opens hardware or depends on native SDK imports.
    source=ROOT/"prediction-random-forest"/"tensorflow"/"client"/"brainflow1.py"
    tree=ast.parse(source.read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.ClassDef)]
    namespace=dict(Enum=Enum,BrainFlowInputParams=SimpleNamespace,
                   BoardShim=SimpleNamespace(enable_dev_board_logger=lambda:None),
                   np=np,time=time,QualityGatedPredictor=QualityGatedPredictor)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),"exec"),namespace)
    return namespace["bciConnection"]


def test_remote_client_never_posts_a_rejected_capture():
    client=client_class()()
    sample=capture()
    sample["raw"][0]=0
    client.quality_capture=dict(sample,source_kind="real")
    client.read_from_board=lambda:np.zeros((32,250))
    client.send_data_to_server=lambda raw:pytest.fail("rejected input must never be posted")
    assert client.bciConnectionController()["status"]=="withheld"


def test_remote_client_good_capture_is_posted_and_missing_label_withholds():
    client=client_class()()
    sample=capture()
    data=np.zeros((32,250))
    data[1:17]=sample["raw"]
    client.quality_capture=dict(sample,source_kind="real")
    client.read_from_board=lambda:data
    posted=[]
    client.send_data_to_server=lambda raw:posted.append(raw) or SimpleNamespace(raise_for_status=lambda:None,json=lambda:{"prediction_label":"right"})
    result=client.bciConnectionController()
    assert result["prediction_label"]=="right" and may_execute(result) and len(posted)==1
    np.testing.assert_array_equal(posted[0],data)
    client.send_data_to_server=lambda raw:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{})
    assert client.bciConnectionController()["status"]=="withheld"


def test_worker_rechecks_queued_approval_and_preserves_manual_commands():
    methods = gui_methods()
    accepted = QualityGatedPredictor().predict(predict=lambda raw:"forward", **capture())
    stale = dict(accepted, expires_monotonic=0)
    items = [("forward", None, accepted), ("right", None, stale), ("land", None, None)]
    class EndOfQueue(Exception): pass
    class FakeQueue:
        def __init__(self): self.finished=0
        def get(self):
            if not items: raise EndOfQueue()
            return items.pop(0)
        def task_done(self): self.finished+=1
    b = SimpleNamespace(current_prediction_result=withheld_result("new_capture_rejected"),
                        cmd_queue=FakeQueue(), _movement_distance_override=None,
                        actions=[], logMessage=Signal())
    b.getDroneAction=lambda action:b.actions.append(action)
    with pytest.raises(EndOfQueue): methods["_drone_loop"](b)
    assert b.actions==["land"] and b.cmd_queue.finished==3
    assert len(b.logMessage.values)==2


def test_prediction_approval_travels_with_queued_command():
    methods=gui_methods()
    accepted=QualityGatedPredictor().predict(predict=lambda raw:"left", **capture())
    entries=[]
    b=SimpleNamespace(current_prediction_result=accepted,cmd_queue=SimpleNamespace(put=entries.append))
    b._queue_action=lambda *args,**kw:methods["_queue_action"](b,*args,**kw)
    methods["_queue_prediction"](b,accepted)
    assert entries==[("left",None,accepted)]
    b.current_prediction_result=withheld_result("new_assessment")
    methods["_queue_prediction"](b,accepted)
    assert len(entries)==1
