"""Train quality classifiers from independently reviewed windows, never rule labels."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .acquisition import file_sha256, read_openbci
from .assessment import FEATURE_NAMES, QualityAssessor, QualityConfig, VERSION


class QualityModel:
    """Trusted local sklearn artifact with a fixed feature/quality contract."""
    def __init__(self, artifact):
        if artifact.get("version") != VERSION or artifact.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Unsupported quality model/feature contract")
        self.model = artifact["model"]
        self.policy_id = artifact["policy_id"]
        self.threshold = artifact["threshold"]
        self.config = QualityConfig(**artifact["config"])
        if self.config.policy_id != self.policy_id or not 0 < self.threshold < 1:
            raise ValueError("Quality model config/threshold mismatch")
        if list(self.model.classes_) != [0, 1]:
            raise ValueError("Quality labels must be 0=unacceptable, 1=acceptable")

    @classmethod
    def load(cls, path):
        # joblib/pickle files must come from this trusted local training workflow.
        return cls(joblib.load(path))

    def predict_usable(self, features):
        vector = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite quality features")
        return float(self.model.predict_proba(vector)[0, 1])


def reviewed_features(annotation_path, config):
    path = Path(annotation_path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Require reviewed quality annotations")
    assessor = QualityAssessor(config)
    X, y, groups, ids, recordings = [], [], [], [], []
    sources, evidence_hashes, occupied, owners = {}, {}, {}, {}
    for row in rows:
        if (row.get("label_verified") is not True or type(row.get("reviewed_quality_label")) is not int
                or row["reviewed_quality_label"] not in (0, 1)
                or not isinstance(row.get("reviewed_by"), str) or not row["reviewed_by"].strip()):
            raise ValueError("Require independent verified quality labels, reviewer identity; rule_usable is not ground truth")
        reviewed = datetime.fromisoformat(row.get("reviewed_at_utc", "").replace("Z", "+00:00"))
        if reviewed.tzinfo is None:
            raise ValueError("Review timestamp needs a timezone")
        evidence = row.get("annotation_evidence")
        if not isinstance(evidence, dict) or not evidence.get("locator"):
            raise ValueError("Require independent annotation evidence with a locator")
        source = (path.parent / row["source_path"]).resolve()
        proof = (path.parent / evidence["path"]).resolve()
        if proof == source or proof == path:
            raise ValueError("Annotations/raw EEG alone are not independent review evidence")
        if proof not in evidence_hashes:
            evidence_hashes[proof] = file_sha256(proof)
        if evidence_hashes[proof] != evidence.get("sha256"):
            raise ValueError("Quality review evidence changed")
        if source not in sources:
            sources[source] = (file_sha256(source), read_openbci(source))
        digest, capture = sources[source]
        if digest != row.get("source_sha256") or row.get("quality_policy_id") != config.policy_id:
            raise ValueError("Source/quality policy hash mismatch")
        participant = row.get("participant_id")
        if not isinstance(participant, str) or not participant.strip():
            raise ValueError("Require participant IDs")
        group = next((p for p in source.parts if re.fullmatch(r"G\d{2}", p, re.I)), None)
        person = next((p for p in source.parts if re.fullmatch(r"person\d{2}", p, re.I)), None)
        if group is None or person is None or participant != f"{group.upper()}/{person.lower()}":
            raise ValueError("Reviewed participant identity disagrees with recording path")
        if source in owners and owners[source] != participant:
            raise ValueError("One source cannot change participant")
        owners[source] = participant
        a, b = row.get("source_start_sample"), row.get("source_end_sample_exclusive")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= capture["raw"].shape[1]:
            raise ValueError("Invalid original quality window bounds")
        if b-a != int(round(config.min_seconds * capture["sfreq"])):
            raise ValueError("Reviewed window duration differs from replay")
        for left, right in occupied.setdefault(source, []):
            if max(a, left) < min(b, right):
                raise ValueError("Overlapping reviewed windows")
        occupied[source].append((a, b))
        window = {key:value for key,value in capture.items() if key != "source_kind"}
        for key in ("raw", "counters", "timestamps"):
            window[key] = capture[key][:, a:b] if key == "raw" else capture[key][a:b]
        actual = assessor.assess(**window)
        if actual["raw_sha256"] != row.get("raw_sha256") or actual["window_features"] is None:
            raise ValueError("Raw window mismatch or nonfinite EEG cannot train a feature classifier")
        X.append([actual["window_features"][name] for name in FEATURE_NAMES])
        y.append(row["reviewed_quality_label"])
        groups.append(participant)
        ids.append(row["window_id"])
        recordings.append(digest)
    if len(set(ids)) != len(ids) or set(y) != {0, 1}:
        raise ValueError("Require unique windows covering both independent quality labels")
    if len(set(groups)) < 5:
        raise ValueError("Need at least five reviewed participants for the fixed nested grouped protocol")
    return np.asarray(X), np.asarray(y), np.asarray(groups), np.asarray(ids), np.asarray(recordings)


def metrics(y, p, threshold):
    predicted = p >= threshold
    bad, good = y == 0, y == 1
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, average="macro", zero_division=0)),
                unacceptable_average_precision=float(average_precision_score(1-y, 1-p)),
                confusion_matrix=confusion_matrix(y, predicted, labels=[0,1]).tolist(),
                bad_window_acceptance=float(predicted[bad].mean()), usable_window_rejection=float((~predicted[good]).mean()),
                retained_coverage=float(predicted.mean()))


def train(annotations, out, config=None, *, seed=42):
    config = config or QualityConfig()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    X, y, groups, ids, recordings = reviewed_features(annotations, config)
    outer = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    reports = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quality-train-", dir=out.parent) as temporary:
        stage = Path(temporary) / "experiment"
        stage.mkdir()
        for fold, (development, test) in enumerate(outer.split(X, y, groups)):
            inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed+1)
            a, b = next(inner.split(X[development], y[development], groups[development]))
            training, validation = development[a], development[b]
            for positions in (training, validation, test):
                if set(y[positions]) != {0,1}:
                    raise ValueError("Every fixed partition must contain both quality labels; collect broader review coverage")
            for left, right in ((training, validation), (training, test), (validation, test)):
                if set(groups[left]) & set(groups[right]) or set(recordings[left]) & set(recordings[right]):
                    raise ValueError("Participant/duplicate-recording leakage in reviewed quality data")
            candidates = {
                "logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
                "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", n_jobs=1, random_state=seed),
            }
            best = None
            results = []
            for name, model in candidates.items():
                model.fit(X[training], y[training])
                probabilities = model.predict_proba(X[validation])[:,1]
                for threshold in (.3, .4, .5, .6, .7, .8, .9):
                    score = metrics(y[validation], probabilities, threshold)
                    results.append(dict(model=name, threshold=threshold, validation=score))
                    if best is None or score["macro_f1"] > best[0]:
                        best = (score["macro_f1"], name, model, threshold)
            _, name, model, threshold = best
            # Test is touched once, after validation model/threshold selection.
            probabilities = model.predict_proba(X[test])[:,1]
            target = stage / f"fold_{fold}"
            target.mkdir()
            artifact = dict(version=VERSION, model=model, policy_id=config.policy_id, config=config.__dict__,
                            feature_names=list(FEATURE_NAMES), threshold=threshold,
                            annotation_sha256=file_sha256(annotations), model_name=name,
                            training_window_ids=ids[training].tolist())
            joblib.dump(artifact, target / "quality_model.joblib")
            QualityModel.load(target / "quality_model.joblib")
            np.savez_compressed(target / "predictions.npz", ids=ids[test], y=y[test], usable_probability=probabilities)
            report = dict(fold=fold, selected_model=name, selected_threshold=threshold, validation_candidates=results,
                          test=metrics(y[test], probabilities, threshold),
                          split_participants={name:np.unique(groups[ix]).tolist() for name,ix in
                                              (("train",training),("validation",validation),("test",test))})
            (target / "report.json").write_text(json.dumps(report, indent=2))
            reports.append(report)
        summary = dict(folds=reports, independently_reviewed_windows=len(y), participants=len(set(groups)),
                       policy_id=config.policy_id, annotation_sha256=file_sha256(annotations),
                       label_definition={"0":"unacceptable", "1":"acceptable"},
                       note="Probability estimates are not claimed calibrated; do not select a deployment fold by its test score")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2))
        stage.rename(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = QualityConfig(**json.loads(Path(args.config).read_text())) if args.config else QualityConfig()
    print(json.dumps(train(args.annotations, args.out, config), indent=2))


if __name__ == "__main__":
    main()
