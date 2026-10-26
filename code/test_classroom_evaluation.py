"""课堂评价协议与历史患者协议的检查；只使用小型合成数据。"""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data import PreparedData, class_counts, indices_from_manifest, split_by_heartbeat
from train import evaluation_settings, main, parse_args


def synthetic_data(n=200):
    rng = np.random.default_rng(123)
    records = np.array([str(100 + (i // 50)) for i in range(n)])
    return PreparedData(rng.normal(size=(n, 300, 1)).astype(np.float32),
        np.arange(n, dtype=np.int64) % 5, records, records.copy(),
        np.arange(n, dtype=np.int64) * 360,
        {"selected_records": sorted(set(records.tolist()))})


class ClassroomEvaluationTests(unittest.TestCase):
    def test_exact_two_shuffle_algorithm_and_ratio(self):
        data = synthetic_data()
        with tempfile.TemporaryDirectory(prefix="ecg-classroom-split-") as directory:
            splits, manifest = split_by_heartbeat(data, Path(directory), seed=42)
            rng = np.random.RandomState(42)
            expected_order = np.arange(200)
            rng.shuffle(expected_order)
            permutation = rng.permutation(200)
            np.testing.assert_array_equal(splits["test"], expected_order[permutation[:40]])
            np.testing.assert_array_equal(splits["train"], expected_order[permutation[40:]])
            self.assertEqual((len(splits["train"]), len(splits["test"])), (160, 40))
            self.assertFalse(set(splits["train"]) & set(splits["test"]))
            self.assertGreater(manifest["patient_overlap_count"], 0)
            restored = indices_from_manifest(data, manifest)
            repeated, _ = split_by_heartbeat(data, Path(directory), seed=42)
            for name in ("train", "test"):
                np.testing.assert_array_equal(restored[name], splits[name])
                np.testing.assert_array_equal(repeated[name], splits[name])

    def test_reordered_samples_and_overlapping_rows_are_rejected(self):
        data = synthetic_data()
        with tempfile.TemporaryDirectory(prefix="ecg-classroom-identity-") as directory:
            _, manifest = split_by_heartbeat(data, Path(directory))
            reordered = synthetic_data()
            reordered.peaks = reordered.peaks[::-1].copy()
            with self.assertRaises(ValueError):
                indices_from_manifest(reordered, manifest)
            broken = json.loads(json.dumps(manifest))
            broken["indices"]["train"][0] = broken["indices"]["test"][0]
            with self.assertRaises(ValueError):
                indices_from_manifest(data, broken)

    def test_legacy_patient_manifest_is_still_patient_disjoint(self):
        data = synthetic_data(150)
        patients = {"train": ["100"], "val": ["101"], "test": ["102"]}
        manifest = {"patients": patients, "sample_counts": {
            name: {"total": 50, "classes": class_counts(data.y[np.isin(data.patients, ids)])}
            for name, ids in patients.items()}}
        restored = indices_from_manifest(data, manifest)
        self.assertEqual(len(restored["test"]), 50)
        broken = json.loads(json.dumps(manifest))
        broken["patients"]["test"] = ["100", "102"]
        with self.assertRaises(ValueError):
            indices_from_manifest(data, broken)

    def test_cli_protocol_defaults_and_selection_metrics(self):
        with patch.object(sys, "argv", ["train.py"]):
            classroom = parse_args()
        self.assertEqual(classroom.evaluation_protocol, "classroom")
        self.assertEqual((classroom.val_fraction, classroom.test_fraction), (0.0, 0.2))
        self.assertEqual(evaluation_settings("classroom")["name"], "test_accuracy")
        with patch.object(sys, "argv", ["train.py", "--evaluation-protocol", "patient"]):
            patient = parse_args()
        self.assertEqual((patient.val_fraction, patient.test_fraction), (0.15, 0.15))
        self.assertEqual(evaluation_settings("patient")["name"], "validation_macro_f1")

    def test_classroom_training_entry_on_synthetic_data_and_evaluation_restore(self):
        # 微型 CPU 检查保存逻辑，不读取项目原始数据。
        import evaluate
        data = synthetic_data(100)
        with tempfile.TemporaryDirectory(prefix="ecg-classroom-entry-") as directory:
            output_dir = Path(directory).resolve()
            self.assertTrue(output_dir.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            arguments = ["train.py", "--output-dir", str(output_dir), "--device", "cpu",
                "--epochs", "2", "--batch-size", "20", "--hidden-size", "8",
                "--num-layers", "1", "--classifier-hidden-sizes", "8"]
            with patch.object(sys, "argv", arguments), patch("train.prepare_dataset", return_value=data), \
                    patch("train.save_data_figures"), contextlib.redirect_stdout(io.StringIO()):
                main()
            run = next((output_dir / "runs").iterdir())
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["test_used_for_model_selection"])
            self.assertTrue(summary["test_evaluated"])
            self.assertEqual(summary["selection_metric"], "test_accuracy")
            self.assertEqual(summary["test"]["sample_count"], 20)
            self.assertNotIn("validation", summary)
            self.assertFalse((run / "validation_metrics.json").exists())
            with (run / "history.csv").open(encoding="utf-8-sig", newline="") as file:
                history = list(csv.DictReader(file))
            self.assertIn("test_accuracy", history[0])
            self.assertNotIn("val_accuracy", history[0])
            checkpoint_path = run / "best_model.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["selection_metrics"]["accuracy"],
                             max(float(row["test_accuracy"]) for row in history))
            eval_arguments = ["evaluate.py", "--checkpoint", str(checkpoint_path), "--device", "cpu"]
            with patch.object(sys, "argv", eval_arguments), \
                    patch("evaluate.prepare_dataset", return_value=data), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            repeated = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(repeated["accuracy"], summary["test"]["accuracy"])
            self.assertEqual(repeated["evaluation_protocol"], "classroom")


if __name__ == "__main__":
    unittest.main()
