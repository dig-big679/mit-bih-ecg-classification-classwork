"""MIT-BIH 预处理、课堂随机心拍划分，以及可选的患者独立划分。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pywt
import torch
import wfdb
from torch.utils.data import Dataset


CLASS_NAMES = ("N", "A", "V", "L", "R")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
# MIT-BIH 中含起搏心拍的四条记录；102、104 同时没有 MLII 导联。
PACED_RECORDS = {"102", "104", "107", "217"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class PreprocessConfig:
    left: int = 99
    right: int = 201  # Python 右端点不包含在切片内，总长度 300。
    wavelet: str = "db5"
    wavelet_level: int = 9
    denoise: bool = True
    trim_first: int = 10  # 对应课堂代码：跳过最前 10 条标注。
    trim_last: int = 5
    max_beats_per_record: int = 0  # 0 表示使用全部；小规模检查才限制数量。
    seed: int = 42

    def __post_init__(self):
        if self.left < 0 or self.right <= 0 or self.wavelet_level < 1:
            raise ValueError("left >= 0, right > 0, wavelet_level >= 1 are required")
        if min(self.trim_first, self.trim_last, self.max_beats_per_record) < 0:
            raise ValueError("Trim counts and max_beats_per_record must be nonnegative")


@dataclass
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    records: np.ndarray
    patients: np.ndarray
    peaks: np.ndarray
    summary: dict


def patient_id(record_id: str) -> str:
    """201、202 来自同一患者，其余记录分别对应不同患者。"""
    return "201_202" if record_id in {"201", "202"} else record_id


def discover_records(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir).resolve()
    records = sorted(p.stem for p in data_dir.glob("*.hea") if p.stem.isdigit())
    if not records:
        raise FileNotFoundError(f"No numeric .hea records found in {data_dir}")
    missing = [str(data_dir / f"{rid}.{ext}") for rid in records
               for ext in ("dat", "atr") if not (data_dir / f"{rid}.{ext}").is_file()]
    if missing:
        raise FileNotFoundError("Missing record files:\n" + "\n".join(missing))
    return records


def denoise_signal(signal: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """沿用课堂 db5 小波方法；它不等同于完整的基线漂移校正。"""
    signal = np.asarray(signal, dtype=np.float64)
    if not config.denoise:
        return signal.copy()
    wavelet = pywt.Wavelet(config.wavelet)
    level = min(config.wavelet_level, pywt.dwt_max_level(len(signal), wavelet.dec_len))
    if level < 1:
        return signal.copy()
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(max(len(detail), 2)))
    # cA 保留；最高频的两层细节置零，其他细节做软阈值处理。
    for index in range(1, len(coeffs)):
        if index >= len(coeffs) - 2:
            coeffs[index] = np.zeros_like(coeffs[index])
        else:
            coeffs[index] = pywt.threshold(coeffs[index], threshold, mode="soft")
    return pywt.waverec(coeffs, wavelet)[:len(signal)]


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


def _signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _stratified_cap(labels: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if not limit or len(labels) <= limit:
        return np.arange(len(labels))
    # 开发用子集保留本条记录中出现的每个类别至少一个样本。
    present = np.unique(labels)
    if limit < len(present):
        raise ValueError("max_beats_per_record must cover all classes present in a record")
    mandatory = np.array([rng.choice(np.flatnonzero(labels == label)) for label in present])
    remaining = np.setdiff1d(np.arange(len(labels)), mandatory)
    chosen = rng.choice(remaining, size=limit - len(mandatory), replace=False)
    return np.sort(np.concatenate([mandatory, chosen]))


def prepare_dataset(data_dir: Path, cache_dir: Path, config: PreprocessConfig,
                    selected_records: list[str] | None = None,
                    include_paced: bool = False) -> PreparedData:
    """读取本地原始文件；不下载、不改动 data 下的任何文件。"""
    data_dir, cache_dir = Path(data_dir).resolve(), Path(cache_dir).resolve()
    all_records = discover_records(data_dir)
    requested = set(selected_records) if selected_records else set(all_records)
    unknown = requested - set(all_records)
    if unknown:
        raise ValueError(f"Unknown records: {sorted(unknown)}")
    records = sorted(requested if include_paced else requested - PACED_RECORDS)
    if not records:
        raise ValueError("No records remain after selection")
    sources = [(rid, ext, (data_dir / f"{rid}.{ext}").stat().st_size,
                (data_dir / f"{rid}.{ext}").stat().st_mtime_ns)
               for rid in records for ext in ("dat", "hea", "atr")]
    signature = _signature({"version": 1, "config": asdict(config), "sources": sources,
                            "data_dir": str(data_dir), "wfdb": wfdb.__version__,
                            "pywt": pywt.__version__})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"beats-{signature}.npz"
    if cache_path.exists():
        print(f"Loading preprocessed cache: {cache_path}", flush=True)
        with np.load(cache_path, allow_pickle=False) as saved:
            return PreparedData(saved["x"], saved["y"], saved["records"], saved["patients"],
                                saved["peaks"], json.loads(str(saved["summary"].item())))

    xs, ys, record_ids, patient_ids, peaks = [], [], [], [], []
    rng = np.random.default_rng(config.seed)
    summary = {"config": asdict(config), "cache_path": str(cache_path),
               "data_dir": str(data_dir), "class_names": list(CLASS_NAMES),
               "available_records": all_records, "selected_records": records,
               "excluded_records": sorted(set(all_records) - set(records)), "records": {}}
    for number, rid in enumerate(records, 1):
        base = str(data_dir / rid)
        header = wfdb.rdheader(base)
        if "MLII" not in header.sig_name:
            raise ValueError(f"Record {rid} has no MLII channel: {header.sig_name}")
        if header.fs != 360:
            raise ValueError(f"Record {rid}: expected 360 Hz, got {header.fs}")
        record = wfdb.rdrecord(base, physical=False, channel_names=["MLII"])
        signal = record.d_signal[:, 0].astype(np.float64)
        filtered = denoise_signal(signal, config)
        annotation = wfdb.rdann(base, "atr")
        end = len(annotation.sample) - config.trim_last if config.trim_last else len(annotation.sample)
        eligible_labels, eligible_peaks = [], []
        discarded = {"other_labels": 0, "boundary": 0, "nonfinite": 0}
        for peak, label in zip(annotation.sample[config.trim_first:end],
                               annotation.symbol[config.trim_first:end]):
            if label not in CLASS_TO_INDEX:
                discarded["other_labels"] += 1
                continue
            if peak - config.left < 0 or peak + config.right > len(filtered):
                discarded["boundary"] += 1
                continue
            beat = filtered[peak - config.left:peak + config.right]
            if not np.isfinite(beat).all():
                discarded["nonfinite"] += 1
                continue
            eligible_labels.append(CLASS_TO_INDEX[label])
            eligible_peaks.append(int(peak))
        labels = np.array(eligible_labels, dtype=np.int64)
        selected = _stratified_cap(labels, config.max_beats_per_record, rng)
        for index in selected:
            peak = eligible_peaks[index]
            beat = filtered[peak - config.left:peak + config.right]
            # 每个心拍独立标准化，不估计任何跨数据集的均值/方差。
            std = beat.std()
            normalized = (beat - beat.mean()) / max(float(std), 1e-8)
            xs.append(normalized.astype(np.float32)[:, None])
            ys.append(eligible_labels[index])
            peaks.append(peak)
            record_ids.append(rid)
            patient_ids.append(patient_id(rid))
        summary["records"][rid] = {"patient": patient_id(rid), "fs": header.fs,
            "signal_length": len(signal), "channels": header.sig_name,
            "eligible_class_counts": class_counts(labels),
            "used_class_counts": class_counts(labels[selected]), "discarded": discarded}
        print(f"[{number:02d}/{len(records)}] Record {rid}: {len(labels)} eligible, "
              f"{len(selected)} used", flush=True)
    if not xs:
        raise ValueError("No eligible heartbeats were extracted")
    prepared = PreparedData(np.stack(xs), np.array(ys, dtype=np.int64),
        np.array(record_ids), np.array(patient_ids), np.array(peaks, dtype=np.int64), summary)
    summary["shape"] = list(prepared.x.shape)
    summary["class_counts"] = class_counts(prepared.y)
    summary["sample_count"] = len(prepared.y)
    np.savez_compressed(cache_path, x=prepared.x, y=prepared.y, records=prepared.records,
        patients=prepared.patients, peaks=prepared.peaks,
        summary=np.array(json.dumps(summary, ensure_ascii=False)))
    cache_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return prepared


def split_by_patient(data: PreparedData, split_dir: Path, seed: int = 42,
                     val_fraction: float = 0.15, test_fraction: float = 0.15):
    """固定患者分组，按类别数量近似分层；不使用模型表现选择划分。"""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Invalid validation/test fractions")
    groups = sorted(set(data.patients.tolist()))
    if len(groups) < 3:
        raise ValueError("Patient-independent train/validation/test needs at least three patients")
    specification = {"version": 2, "seed": seed, "val_fraction": val_fraction,
                     "test_fraction": test_fraction,
                     "records": data.summary["selected_records"], "classes": CLASS_NAMES}
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"patients-{_signature(specification)}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        # 使用截取后、开发子集抽样前的标签计数，因此 smoke/full 使用相同患者划分。
        matrix = np.zeros((len(groups), len(CLASS_NAMES)), dtype=np.int64)
        for record in data.summary["records"].values():
            matrix[groups.index(record["patient"])] += [record["eligible_class_counts"][c]
                                                        for c in CLASS_NAMES]
        totals = matrix.sum(axis=0)
        if np.any(totals == 0):
            raise ValueError(f"Some target classes are absent: {totals.tolist()}")
        n_test = max(1, round(len(groups) * test_fraction))
        n_val = max(1, round(len(groups) * val_fraction))
        if n_test + n_val >= len(groups):
            raise ValueError("Too few patients for the requested split fractions")
        target = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction])
        rng = np.random.default_rng(seed)
        best, best_score = None, float("inf")
        for _ in range(8192):
            order = rng.permutation(len(groups))
            candidate = (order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test])
            counts = np.stack([matrix[index].sum(axis=0) for index in candidate])
            # 除类别覆盖外，避免验证集仅有极少数该类心拍。
            if np.any(counts < np.maximum(1, totals[None, :] * target[:, None] * 0.25)):
                continue
            score = (np.square(counts / totals[None, :] - target[:, None])
                     / target[:, None]).mean()
            score += np.square(counts.sum(axis=1) / totals.sum() - target).mean()
            if score < best_score:
                best, best_score = candidate, score
        if best is None:
            raise ValueError("Cannot create patient-disjoint splits covering all five classes. "
                             "Use more records or adjust the split fractions.")
        manifest = {"specification": specification, "patients": {
            name: sorted(groups[int(i)] for i in index)
            for name, index in zip(("train", "val", "test"), best)}}
        manifest["records"] = {name: sorted(rid for rid in data.summary["selected_records"]
                                            if patient_id(rid) in patients)
                               for name, patients in manifest["patients"].items()}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    splits = {name: np.flatnonzero(np.isin(data.patients, patients))
              for name, patients in manifest["patients"].items()}
    sets = [set(manifest["patients"][name]) for name in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Patient overlap detected in split manifest")
    if set.union(*sets) != set(groups):
        raise ValueError("Split manifest does not cover the selected patients")
    if sum(map(len, splits.values())) != len(data.y):
        raise ValueError("Split does not cover all samples")
    for name, index in splits.items():
        if len(index) == 0 or np.any(np.bincount(data.y[index], minlength=5) == 0):
            raise ValueError(f"Split {name} does not contain all five classes")
    manifest = dict(manifest, protocol="patient", path=str(manifest_path), sample_counts={
        name: {"total": len(index), "classes": class_counts(data.y[index])}
        for name, index in splits.items()})
    return splits, manifest


def sample_order_signature(data: PreparedData) -> str:
    """识别心拍行顺序，防止保存的行索引应用到不同的心拍集合。"""
    digest = hashlib.sha256()
    digest.update("\n".join(data.records.tolist()).encode("utf-8"))
    digest.update(np.asarray(data.peaks, dtype="<i8").tobytes())
    digest.update(np.asarray(data.y, dtype="<i8").tobytes())
    return digest.hexdigest()


def _check_index_partition(splits: dict[str, np.ndarray], sample_count: int):
    for name, indices in splits.items():
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0:
            raise ValueError(f"Split {name} needs a nonempty array of integer row indices")
        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError(f"Split {name} contains out-of-range sample indices")
    combined = np.concatenate(list(splits.values()))
    if len(combined) != sample_count or len(np.unique(combined)) != sample_count:
        raise ValueError("Split rows overlap or do not cover all samples exactly once")


def indices_from_manifest(data: PreparedData, manifest: dict) -> dict[str, np.ndarray]:
    """恢复 checkpoint 保存的划分；历史无 protocol 的文件按患者划分读取。"""
    protocol = manifest.get("protocol", "patient")
    if protocol == "classroom":
        if manifest.get("sample_order_sha256") != sample_order_signature(data):
            raise ValueError("Heartbeat order differs from the saved classroom split")
        if set(manifest["indices"]) != {"train", "test"}:
            raise ValueError("Classroom split must contain train and test only")
        splits = {name: np.asarray(indices) for name, indices in manifest["indices"].items()}
    elif protocol == "patient":
        patient_sets = [set(manifest["patients"][name]) for name in ("train", "val", "test")]
        if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Patient overlap detected in the saved patient-independent split")
        if set.union(*patient_sets) != set(data.patients.tolist()):
            raise ValueError("Saved patient split does not cover the current data")
        splits = {name: np.flatnonzero(np.isin(data.patients, manifest["patients"][name]))
                  for name in ("train", "val", "test")}
    else:
        raise ValueError(f"Unknown split protocol: {protocol}")
    _check_index_partition(splits, len(data.y))
    for name, indices in splits.items():
        expected = manifest["sample_counts"][name]
        if len(indices) != expected["total"] or class_counts(data.y[indices]) != expected["classes"]:
            raise ValueError(f"Split {name} sample/class counts differ from the saved manifest")
    return splits


def split_by_heartbeat(data: PreparedData, split_dir: Path, seed: int = 42,
                       test_fraction: float = 0.2):
    """课堂流程：先 shuffle 心拍行，再 permutation，前 20% 测试、其余训练。

    不分患者、不按类别分层，不额外保留验证集。同一患者可出现在两侧，
    但每个心拍行只属于一个集合。保存真实行索引用于后续精确恢复测试集。
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n_test = int(len(data.y) * test_fraction)
    if n_test == 0 or n_test == len(data.y):
        raise ValueError("Too few heartbeats for the requested train/test fractions")
    order_signature = sample_order_signature(data)
    specification = {"version": 1, "protocol": "classroom", "seed": seed,
                     "test_fraction": test_fraction, "sample_order_sha256": order_signature}
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"heartbeats-{_signature(specification)}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return indices_from_manifest(data, manifest), dict(manifest, path=str(manifest_path))
    rng = np.random.RandomState(seed)
    row_order = np.arange(len(data.y))
    rng.shuffle(row_order)
    shuffled_index = rng.permutation(len(data.y))
    splits = {"train": row_order[shuffled_index[n_test:]],
              "test": row_order[shuffled_index[:n_test]]}
    _check_index_partition(splits, len(data.y))
    patient_sets = {name: sorted(set(data.patients[index].tolist()))
                    for name, index in splits.items()}
    manifest = {"protocol": "classroom", "specification": specification,
        "sample_order_sha256": order_signature,
        "indices": {name: index.tolist() for name, index in splits.items()},
        "patients": patient_sets,
        "records": {name: sorted(set(data.records[index].tolist()))
                    for name, index in splits.items()},
        "patient_overlap_count": len(set(patient_sets["train"]) & set(patient_sets["test"])),
        "sample_counts": {name: {"total": len(index), "classes": class_counts(data.y[index])}
                          for name, index in splits.items()},
        "path": str(manifest_path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits, manifest


class HeartbeatDataset(Dataset):
    """缓存保持 [N, 300, 1]，按目标模型转换返回的单个心拍形状。"""
    def __init__(self, data: PreparedData, indices: np.ndarray, input_size: int = 300):
        if input_size not in {1, 300}:
            raise ValueError("input_size must be 300 (classroom) or 1 (pointwise sequence)")
        if data.x.ndim != 3 or data.x.shape[1:] != (300, 1):
            raise ValueError(f"Expected cached heartbeats [N, 300, 1], got {data.x.shape}")
        self.data, self.indices = data, np.asarray(indices)
        self.input_size = input_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        sample = self.data.x[row]
        if self.input_size == 300:
            sample = sample.T  # (300, 1) -> (1, 300)，采样点顺序保持不变。
        return torch.from_numpy(sample), torch.tensor(self.data.y[row], dtype=torch.long)
