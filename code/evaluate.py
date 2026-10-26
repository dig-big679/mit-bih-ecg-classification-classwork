"""恢复模型保存的心拍或患者划分并测试；不重新训练或选择模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import (CLASS_NAMES, DEFAULT_DATA_DIR, PreprocessConfig, prepare_dataset,
                  indices_from_manifest)
from models import build_model
from train import make_loader, resolve_device, run_epoch, save_evaluation, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError("Checkpoint class mapping differs from this code")
    config = PreprocessConfig(**checkpoint["preprocess_config"])
    seed_everything(config.seed)
    device = resolve_device(args.device)
    run_dir = checkpoint_path.parent
    data = prepare_dataset(args.data_dir, run_dir.parent.parent / "cache", config,
                           checkpoint["selected_records"])
    manifest = checkpoint["split_manifest"]
    indices = indices_from_manifest(data, manifest)["test"]
    protocol = checkpoint.get("evaluation_protocol", manifest.get("protocol", "patient"))
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    weights = checkpoint["class_weights"]
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    # 按 checkpoint 的真实输入维度组织数据；兼容修改前保存的逐点输入模型。
    loader = make_loader(data, indices, args.batch_size, device, input_size=model.input_size)
    metrics, labels, probabilities = run_epoch(model, loader, device, weights)
    metrics["development_subset"] = bool(checkpoint["development_subset"])
    metrics.update(evaluation_protocol=protocol,
                   test_used_for_model_selection=checkpoint.get("test_used_for_model_selection",
                                                               protocol == "classroom"))
    save_evaluation(run_dir, "test", metrics, labels, probabilities, data, indices)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(test=metrics, test_evaluated=True,
                   evaluation_protocol=protocol,
                   test_used_for_model_selection=metrics["test_used_for_model_selection"],
                   development_subset=bool(checkpoint["development_subset"]))
    save_json(summary_path, summary)
    print(f"Test samples: {len(indices)}\nTest accuracy: {metrics['accuracy']:.2%}"
          f"\nTest Macro-F1: {metrics['macro_f1']:.4f}\nResults: {run_dir}")
    print(f"Evaluation protocol: {protocol}")
    if protocol == "classroom":
        print("Classroom reproduction: test Accuracy was used for model selection.")
    if checkpoint["development_subset"]:
        print("DEVELOPMENT SUBSET: this checkpoint cannot provide full-data experiment results.")


if __name__ == "__main__":
    main()
