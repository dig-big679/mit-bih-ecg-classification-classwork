"""训练入口：默认复现课堂随机心拍 8:2 划分，按测试集 Accuracy 选模。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wfdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import (CLASS_NAMES, DEFAULT_DATA_DIR, PROJECT_ROOT, HeartbeatDataset,
                  PreprocessConfig, PreparedData, denoise_signal, prepare_dataset,
                  split_by_patient, split_by_heartbeat)
from models import build_model


def save_json(path: Path, content: dict):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def resolve_device(request: str) -> torch.device:
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but is not available in this Python environment")
    return torch.device("cuda" if request == "auto" and torch.cuda.is_available()
                        else "cpu" if request == "auto" else request)


def make_loader(data: PreparedData, indices: np.ndarray, batch_size: int,
                device: torch.device, shuffle: bool = False, seed: int = 42,
                workers: int = 0, input_size: int = 300) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(HeartbeatDataset(data, indices, input_size=input_size), batch_size=batch_size,
        shuffle=shuffle, num_workers=workers, pin_memory=device.type == "cuda",
        generator=generator)


def run_epoch(model, loader, device, weights=None, optimizer=None, clip_grad=1.0):
    training = optimizer is not None
    model.train(training)
    numerator, denominator = 0.0, 0.0
    labels_all, probabilities_all = [], []
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss_sum = F.cross_entropy(logits, labels, weight=weights, reduction="sum")
            divisor = weights[labels].sum() if weights is not None else labels.numel()
            loss = loss_sum / divisor
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss: inspect input data and learning rate")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            numerator += float(loss_sum.detach())
            denominator += float(divisor)
            labels_all.append(labels.detach().cpu().numpy())
            probabilities_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all)
    predictions = probabilities.argmax(axis=1)
    metrics = {"loss": numerator / denominator,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(5)),
                                   average="macro", zero_division=0)),
        "sample_count": len(labels)}
    return metrics, labels, probabilities


def save_evaluation(out_dir: Path, prefix: str, metrics: dict, labels: np.ndarray,
                    probabilities: np.ndarray, data: PreparedData, indices: np.ndarray):
    predictions = probabilities.argmax(axis=1)
    report = classification_report(labels, predictions, labels=list(range(5)),
        target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=list(range(5)))
    result = dict(metrics, class_names=list(CLASS_NAMES), classification_report=report,
                  confusion_matrix=matrix.tolist())
    save_json(out_dir / f"{prefix}_metrics.json", result)
    (out_dir / f"{prefix}_classification_report.txt").write_text(
        classification_report(labels, predictions, labels=list(range(5)),
            target_names=list(CLASS_NAMES), digits=4, zero_division=0), encoding="utf-8")
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["record", "patient", "peak_sample", "true_label", "predicted_label"]
                        + [f"prob_{c}" for c in CLASS_NAMES])
        for row, label, prediction, probability in zip(indices, labels, predictions, probabilities):
            writer.writerow([data.records[row], data.patients[row], int(data.peaks[row]),
                             CLASS_NAMES[int(label)], CLASS_NAMES[int(prediction)],
                             *map(float, probability)])
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float),
                           where=matrix.sum(axis=1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for ax, values, title, is_fraction in zip(axes, (matrix, normalized),
            ("Counts", "Row-normalized"), (False, True)):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if is_fraction else max(matrix.max(), 1))
        for i in range(5):
            for j in range(5):
                cutoff = 0.5 if is_fraction else matrix.max() / 2
                ax.text(j, i, f"{values[i, j]:.1%}" if is_fraction else str(values[i, j]),
                    ha="center", va="center", color="white" if values[i, j] > cutoff else "black")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_NAMES,
               yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True", title=title)
    fig.suptitle(f"{prefix.capitalize()} confusion matrix")
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return result


def evaluation_settings(protocol: str) -> dict:
    if protocol == "classroom":
        return {"split": "test", "metric": "accuracy", "name": "test_accuracy", "label": "Test"}
    if protocol == "patient":
        return {"split": "val", "metric": "macro_f1", "name": "validation_macro_f1",
                "label": "Validation"}
    raise ValueError(f"Unknown evaluation protocol: {protocol}")


def make_scheduler(optimizer, args):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    return None


def save_history(out_dir: Path, history: list[dict], evaluation_split: str = "val"):
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    has_lr = "lr" in history[0]
    fig, axes = plt.subplots(1, 4 if has_lr else 3,
                             figsize=(18 if has_lr else 14, 4), layout="constrained")
    for ax, metric, title in zip(axes[:3], ("loss", "accuracy", "macro_f1"),
                               ("Loss", "Accuracy", "Macro-F1")):
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        label = "Test (classroom selection)" if evaluation_split == "test" else "Validation"
        ax.plot(epochs, [row[f"{evaluation_split}_{metric}"] for row in history], label=label)
        ax.set(xlabel="Epoch", ylabel=title, title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    if has_lr:
        axes[3].plot([row["epoch"] for row in history], [row["lr"] for row in history])
        axes[3].set(xlabel="Epoch", ylabel="Learning rate", title="Learning rate used")
        axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[3].grid(alpha=0.2)
    fig.savefig(out_dir / "learning_curves.png", dpi=180)
    plt.close(fig)


def save_data_figures(data, config, data_dir, out_dir):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), layout="constrained")
    times = (np.arange(data.x.shape[1]) - config.left) / 360
    for class_index, ax in enumerate(axes):
        row = np.flatnonzero(data.y == class_index)[0]
        ax.plot(times, data.x[row, :, 0], linewidth=1)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set(ylabel="Z-score", title=f"{CLASS_NAMES[class_index]} | record "
               f"{data.records[row]} | peak {data.peaks[row]}")
    axes[-1].set_xlabel("Time relative to annotated R peak (s)")
    fig.savefig(out_dir / "heartbeat_examples.png", dpi=160)
    plt.close(fig)
    counts = [int((data.y == i).sum()) for i in range(5)]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.bar(CLASS_NAMES, counts, color="#3a7ca5")
    ax.bar_label(ax.containers[0])
    ax.set(ylabel="Heartbeats", xlabel="Class", title="Class distribution (selected data)")
    ax.set_ylim(0, max(counts) * 1.15)
    fig.savefig(out_dir / "class_distribution.png", dpi=160)
    plt.close(fig)
    rid = "106" if "106" in data.summary["selected_records"] else data.summary["selected_records"][0]
    raw = wfdb.rdrecord(str(Path(data_dir) / rid), physical=False,
                        channel_names=["MLII"]).d_signal[:, 0].astype(float)
    cleaned = denoise_signal(raw, config)
    start, stop = min(10000, max(0, len(raw) - 1800)), min(11800, len(raw))
    times = np.arange(start, stop) / 360
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, sharey=True, layout="constrained")
    for ax, values, title in zip(axes, (raw, cleaned), ("Raw", "After preprocessing")):
        ax.plot(times, values[start:stop], linewidth=0.8)
        ax.set(ylabel="ADC units", title=f"{title} | record {rid}")
    axes[-1].set_xlabel("Time from recording start (s)")
    fig.savefig(out_dir / "denoising_comparison.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--records", nargs="+", help="Optional record IDs, e.g. 100 101 106")
    parser.add_argument("--model", choices=("rnn", "lstm"), default="lstm")
    parser.add_argument("--input-size", type=int, choices=(1, 300), default=300,
                        help="300: classroom [B,1,300] (default); 1: previous [B,300,1]")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--recurrent-dropout", type=float, default=None,
                        help="Default: 0.2 for multiple layers, 0 for a single layer")
    parser.add_argument("--classifier-dropout", type=float, default=0.5)
    parser.add_argument("--classifier-hidden-sizes", type=int, nargs="+", default=[128, 64],
                        help="Classifier hidden widths, e.g. 128 64 (scheme B) or 64 (old head)")
    parser.add_argument("--pooling", choices=("last", "mean", "max", "meanmax"), default="last",
                        help="Aggregate the last state, or all time-step features")
    parser.add_argument("--loss", choices=("ce", "weighted"), default="ce")
    parser.add_argument("--weight-power", type=float, default=0.5,
                        help="Weighted CE: normalized (1 / training_count) ** power")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none",
                        help="none: fixed LR; cosine: anneal after each epoch over --epochs")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Cosine LR lower bound, reached after --epochs scheduler steps")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-protocol", choices=("classroom", "patient"),
                        default="classroom", help="classroom: random beats 80/20, test Accuracy "
                        "selection; patient: grouped 70/15/15, validation Macro-F1 selection")
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--max-beats-per-record", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Development check: at most 64 beats per record and one epoch")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Patient protocol only: evaluate held-out test after selection. "
                        "Classroom protocol already evaluates test data every epoch.")
    args = parser.parse_args()
    if args.val_fraction is None:
        args.val_fraction = 0.0 if args.evaluation_protocol == "classroom" else 0.15
    if args.test_fraction is None:
        args.test_fraction = 0.2 if args.evaluation_protocol == "classroom" else 0.15
    if args.evaluation_protocol == "classroom" and args.val_fraction != 0:
        parser.error("Classroom protocol has no independent validation split; use --val-fraction 0")
    if not 0 < args.test_fraction < 1:
        parser.error("test-fraction must be between 0 and 1")
    if args.evaluation_protocol == "patient" and not (
            0 < args.val_fraction < 1 and args.val_fraction + args.test_fraction < 1):
        parser.error("Patient protocol needs positive train/validation/test fractions")
    if args.recurrent_dropout is None:
        args.recurrent_dropout = 0.2 if args.num_layers > 1 else 0.0
    if any(width < 1 for width in args.classifier_hidden_sizes):
        parser.error("classifier-hidden-sizes must contain positive integers")
    if min(args.batch_size, args.epochs, args.patience) < 1 or args.lr <= 0 or args.clip_grad <= 0:
        parser.error("batch-size, epochs, patience, lr and clip-grad must be positive")
    if not math.isfinite(args.lr) or not math.isfinite(args.min_lr) or args.min_lr < 0:
        parser.error("lr must be finite and min-lr must be finite and nonnegative")
    if args.scheduler == "cosine" and args.min_lr > args.lr:
        parser.error("Cosine min-lr must not exceed the initial lr")
    if args.workers < 0 or args.weight_decay < 0 or args.weight_power < 0:
        parser.error("workers, weight-decay and weight-power must be nonnegative")
    if args.num_layers == 1 and args.recurrent_dropout != 0:
        parser.error("LSTM/RNN recurrent dropout needs num-layers >= 2")
    if args.input_size == 300 and args.pooling != "last":
        parser.error("Classroom input has one time step: use --pooling last. "
                     "Full-sequence pooling requires --input-size 1.")
    if args.smoke_test:
        args.epochs, args.max_beats_per_record = 1, 64
    return args


def main():
    args = parse_args()
    selection = evaluation_settings(args.evaluation_protocol)
    selection_split = selection["split"]
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    run_dir = output_dir / "runs" / (
        f"{datetime.now():%Y%m%d-%H%M%S}-{args.model}-{args.evaluation_protocol}-{uuid4().hex[:4]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results: {run_dir}\nDevice: {device}", flush=True)
    config = PreprocessConfig(denoise=not args.no_denoise,
        max_beats_per_record=args.max_beats_per_record, seed=args.seed)
    data = prepare_dataset(args.data_dir, output_dir / "cache", config, args.records)
    if args.evaluation_protocol == "classroom":
        splits, manifest = split_by_heartbeat(data, output_dir / "splits", args.seed,
                                             args.test_fraction)
    else:
        splits, manifest = split_by_patient(data, output_dir / "splits", args.seed,
                                            args.val_fraction, args.test_fraction)
    model_input_shape = [None, 1, 300] if args.input_size == 300 else [None, 300, 1]
    save_json(run_dir / "preprocessing_summary.json",
              dict(data.summary, cache_shape=list(data.x.shape), model_input_shape=model_input_shape))
    save_json(run_dir / "split_manifest.json", manifest)
    save_data_figures(data, config, args.data_dir, run_dir)
    model_config = dict(model_type=args.model, hidden_size=args.hidden_size,
        num_layers=args.num_layers, recurrent_dropout=args.recurrent_dropout,
        classifier_dropout=args.classifier_dropout, pooling=args.pooling, input_size=args.input_size,
        classifier_hidden_sizes=args.classifier_hidden_sizes)
    run_config = {name: str(value) if isinstance(value, Path) else value
                  for name, value in vars(args).items()}
    run_config.update(model_config=model_config, preprocess_config=asdict(config),
        model_input_shape=model_input_shape,
        device_used=str(device), torch_version=str(torch.__version__),
        cuda_device=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        development_subset=args.max_beats_per_record > 0,
        selection_metric=selection["name"], selection_split=selection_split,
        test_used_for_model_selection=args.evaluation_protocol == "classroom")
    save_json(run_dir / "config.json", run_config)
    print(f"Evaluation protocol: {args.evaluation_protocol}; selection: {selection['name']}", flush=True)
    print(json.dumps(manifest["sample_counts"], ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print(f"Preprocessing and {args.evaluation_protocol} split completed. No model was trained.")
        return
    train_loader = make_loader(data, splits["train"], args.batch_size, device, True,
                               args.seed, args.workers, input_size=args.input_size)
    selection_loader = make_loader(data, splits[selection_split], args.batch_size, device,
                                   workers=args.workers, input_size=args.input_size)
    model = build_model(model_config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model}\nTrainable parameters: {parameter_count:,}", flush=True)
    counts = np.bincount(data.y[splits["train"]], minlength=5)
    weights = None
    if args.loss == "weighted":
        if np.any(counts == 0):
            raise ValueError("Weighted loss needs all five classes in the training split")
        values = np.power(1.0 / counts, args.weight_power)
        weights = torch.tensor(values / values.mean(), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    scheduler_config = {"name": args.scheduler,
                        "T_max": args.epochs if scheduler is not None else None,
                        "eta_min": args.min_lr if scheduler is not None else None}
    print(f"Learning rate: {args.lr:g}; scheduler: {args.scheduler}"
          + (f"; T_max: {args.epochs}; min_lr: {args.min_lr:g}" if scheduler is not None else ""),
          flush=True)
    history, best_score, stale, best_epoch = [], -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics, _, _ = run_epoch(model, train_loader, device, weights, optimizer,
                                         args.clip_grad)
        selection_metrics, _, _ = run_epoch(model, selection_loader, device, weights)
        history.append(dict(epoch=epoch, lr=epoch_lr,
            **{f"train_{key}": train_metrics[key] for key in ("loss", "accuracy", "macro_f1")},
            **{f"{selection_split}_{key}": selection_metrics[key]
               for key in ("loss", "accuracy", "macro_f1")},
            seconds=time.perf_counter() - epoch_started))
        print(f"Epoch {epoch:02d}/{args.epochs} | lr {epoch_lr:.6g} | train loss {train_metrics['loss']:.4f} "
              f"acc {train_metrics['accuracy']:.4f} | {selection_split} loss "
              f"{selection_metrics['loss']:.4f} acc {selection_metrics['accuracy']:.4f} "
              f"macro-F1 {selection_metrics['macro_f1']:.4f}", flush=True)
        # 本轮全部参数更新后再调度；history 中记录本轮实际使用的学习率。
        if scheduler is not None:
            scheduler.step()
        if selection_metrics[selection["metric"]] > best_score + 1e-6:
            best_score, best_epoch, stale = selection_metrics[selection["metric"]], epoch, 0
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config, "preprocess_config": asdict(config),
                "class_names": list(CLASS_NAMES), "selected_records": data.summary["selected_records"],
                "split_manifest": manifest, "epoch": epoch,
                "evaluation_protocol": args.evaluation_protocol,
                "selection_split": selection_split, "selection_metric": selection["name"],
                "selection_metrics": selection_metrics,
                ("test_metrics" if selection_split == "test" else "validation_metrics"): selection_metrics,
                "test_used_for_model_selection": args.evaluation_protocol == "classroom",
                "class_weights": weights.detach().cpu().tolist() if weights is not None else None,
                "learning_rate": epoch_lr, "scheduler_config": scheduler_config,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "development_subset": args.max_beats_per_record > 0}, run_dir / "best_model.pth")
        else:
            stale += 1
        save_history(run_dir, history, evaluation_split=selection_split)
        if stale >= args.patience:
            print(f"Early stopping: no {selection['name']} improvement for {stale} epochs.")
            break
    checkpoint = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    selection_metrics, labels, probabilities = run_epoch(model, selection_loader, device, weights)
    result_name = "test" if selection_split == "test" else "validation"
    selection_metrics.update(evaluation_protocol=args.evaluation_protocol,
        test_used_for_model_selection=args.evaluation_protocol == "classroom")
    save_evaluation(run_dir, result_name, selection_metrics, labels, probabilities,
                    data, splits[selection_split])
    summary = {"best_epoch": best_epoch, result_name: selection_metrics,
        "scheduler_config": scheduler_config,
        "best_epoch_lr": history[best_epoch - 1]["lr"],
        "trainable_parameters": parameter_count, "training_seconds": time.perf_counter() - started,
        "development_subset": args.max_beats_per_record > 0,
        "evaluation_protocol": args.evaluation_protocol, "selection_metric": selection["name"],
        "test_used_for_model_selection": args.evaluation_protocol == "classroom",
        "test_evaluated": args.evaluation_protocol == "classroom" or args.evaluate_test}
    if args.evaluation_protocol == "patient" and args.evaluate_test:
        test_loader = make_loader(data, splits["test"], args.batch_size, device,
                                  workers=args.workers, input_size=args.input_size)
        test_metrics, labels, probabilities = run_epoch(model, test_loader, device, weights)
        test_metrics.update(evaluation_protocol="patient", test_used_for_model_selection=False)
        save_evaluation(run_dir, "test", test_metrics, labels, probabilities, data, splits["test"])
        summary["test"] = test_metrics
        print(f"Final test accuracy: {test_metrics['accuracy']:.2%}; "
              f"Macro-F1: {test_metrics['macro_f1']:.4f}", flush=True)
    save_json(run_dir / "summary.json", summary)
    print(f"Completed. Best epoch: {best_epoch}; {selection['label']} accuracy: "
          f"{selection_metrics['accuracy']:.2%}; Macro-F1: {selection_metrics['macro_f1']:.4f}")
    if args.max_beats_per_record:
        print("DEVELOPMENT SUBSET: these results are not full-data experiment results.")
    if args.evaluation_protocol == "patient" and not args.evaluate_test:
        print("Held-out test set was not evaluated. Use evaluate.py after finalizing the model.")
    elif args.evaluation_protocol == "classroom":
        print("CLASSROOM REPRODUCTION: random heartbeat split; test Accuracy selected the model. "
              "This is not a patient-independent held-out evaluation.")
    print(f"Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
