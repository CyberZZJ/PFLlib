import copy
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EvaluationError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _project_root() -> str:
    current = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current, "..", "..", "..", ".."))


def _to_tensor_dataset(features: Sequence[Any], labels: Sequence[Any]) -> TensorDataset:
    if len(features) == 0 or len(labels) == 0:
        raise EvaluationError("EVAL_EMPTY_DATASET", "Public dataset is empty.")
    if len(features) != len(labels):
        raise EvaluationError(
            "EVAL_DATA_SHAPE_MISMATCH",
            f"Feature count {len(features)} does not match label count {len(labels)}."
        )

    feature_array = np.asarray(features)
    label_array = np.asarray(labels)

    if feature_array.dtype == object:
        try:
            feature_array = np.stack([np.asarray(row) for row in features], axis=0)
        except Exception as exc:
            raise EvaluationError(
                "EVAL_FEATURE_PARSE_FAILED",
                "Unable to convert feature rows into consistent numeric arrays."
            ) from exc

    feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32)
    label_tensor = torch.as_tensor(label_array, dtype=torch.int64)
    return TensorDataset(feature_tensor, label_tensor)


def _parse_feature_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            return json.loads(text)
        return value
    return value


def _load_csv_dataset(file_path: str, feature_key: str, label_key: str) -> TensorDataset:
    features: List[Any] = []
    labels: List[Any] = []
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            columns = reader.fieldnames or []
            if label_key not in columns:
                raise EvaluationError("EVAL_MISSING_LABEL_COLUMN", f"Missing label column '{label_key}'.")
            use_single_feature_col = feature_key in columns
            feature_columns = [feature_key] if use_single_feature_col else [col for col in columns if col != label_key]
            if not feature_columns:
                raise EvaluationError("EVAL_MISSING_FEATURE_COLUMN", "No feature columns found in CSV file.")

            for row in reader:
                if use_single_feature_col:
                    feature_row = _parse_feature_value(row[feature_key])
                else:
                    feature_row = [_parse_feature_value(row[col]) for col in feature_columns]
                features.append(feature_row)
                labels.append(int(row[label_key]))
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError("EVAL_CSV_LOAD_FAILED", f"Failed loading CSV dataset '{file_path}'.") from exc
    return _to_tensor_dataset(features, labels)


def _load_json_dataset(file_path: str, feature_key: str, label_key: str) -> TensorDataset:
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as exc:
        raise EvaluationError("EVAL_JSON_LOAD_FAILED", f"Failed loading JSON dataset '{file_path}'.") from exc

    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise EvaluationError("EVAL_JSON_SCHEMA_INVALID", "JSON dataset must be a list or contain a 'data' list.")

    features: List[Any] = []
    labels: List[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            raise EvaluationError("EVAL_JSON_SCHEMA_INVALID", "Each JSON row must be an object.")
        if label_key not in row:
            raise EvaluationError("EVAL_MISSING_LABEL_COLUMN", f"Missing label key '{label_key}' in JSON row.")
        if feature_key in row:
            feature_row = row[feature_key]
        else:
            feature_columns = [k for k in row.keys() if k != label_key]
            if not feature_columns:
                raise EvaluationError("EVAL_MISSING_FEATURE_COLUMN", "No feature keys found in JSON row.")
            feature_row = [row[k] for k in feature_columns]
        features.append(feature_row)
        labels.append(int(row[label_key]))

    return _to_tensor_dataset(features, labels)


def _feature_from_tfrecord(example: Any, feature_key: str, label_key: str) -> Tuple[Any, int]:
    feature_dict = example.features.feature
    if label_key not in feature_dict:
        raise EvaluationError("EVAL_MISSING_LABEL_COLUMN", f"Missing label feature '{label_key}' in TFRecord.")

    if feature_key in feature_dict:
        feature_names = [feature_key]
    else:
        feature_names = [name for name in feature_dict.keys() if name != label_key]
    if not feature_names:
        raise EvaluationError("EVAL_MISSING_FEATURE_COLUMN", "No feature entries found in TFRecord example.")

    def _extract_value(feat: Any) -> Any:
        if len(feat.float_list.value) > 0:
            values = list(feat.float_list.value)
        elif len(feat.int64_list.value) > 0:
            values = list(feat.int64_list.value)
        elif len(feat.bytes_list.value) > 0:
            decoded = [v.decode("utf-8") for v in feat.bytes_list.value]
            values = [json.loads(v) if v.startswith("[") and v.endswith("]") else v for v in decoded]
        else:
            values = []
        if len(values) == 1:
            return values[0]
        return values

    if len(feature_names) == 1:
        feature_value = _extract_value(feature_dict[feature_names[0]])
    else:
        feature_value = [_extract_value(feature_dict[name]) for name in feature_names]

    label_value = _extract_value(feature_dict[label_key])
    if isinstance(label_value, list):
        if len(label_value) != 1:
            raise EvaluationError("EVAL_LABEL_PARSE_FAILED", "Label feature should contain one value.")
        label_value = label_value[0]
    return feature_value, int(label_value)


def _load_tfrecord_dataset(file_path: str, feature_key: str, label_key: str) -> TensorDataset:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise EvaluationError(
            "EVAL_TFRECORD_BACKEND_MISSING",
            "TensorFlow is required for TFRecord evaluation but is not installed."
        ) from exc

    features: List[Any] = []
    labels: List[Any] = []
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feat, lbl = _feature_from_tfrecord(example, feature_key, label_key)
            features.append(feat)
            labels.append(lbl)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError("EVAL_TFRECORD_LOAD_FAILED", f"Failed loading TFRecord file '{file_path}'.") from exc

    return _to_tensor_dataset(features, labels)


def _load_npz_dataset(file_path: str, feature_key: str, label_key: str) -> TensorDataset:
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as exc:
        raise EvaluationError("EVAL_NPZ_LOAD_FAILED", f"Failed loading NPZ dataset '{file_path}'.") from exc

    if feature_key in data.files and label_key in data.files:
        features = data[feature_key]
        labels = data[label_key]
    elif "x" in data.files and "y" in data.files:
        features = data["x"]
        labels = data["y"]
    elif "data" in data.files:
        nested = data["data"]
        if nested.shape == ():
            nested = nested.item()
        if isinstance(nested, dict):
            if feature_key in nested and label_key in nested:
                features = nested[feature_key]
                labels = nested[label_key]
            elif "x" in nested and "y" in nested:
                features = nested["x"]
                labels = nested["y"]
            else:
                raise EvaluationError(
                    "EVAL_MISSING_REQUIRED_KEYS",
                    f"NPZ nested 'data' dict must contain '{feature_key}' and '{label_key}' or fallback keys 'x' and 'y'."
                )
        else:
            raise EvaluationError("EVAL_NPZ_SCHEMA_INVALID", "NPZ key 'data' must be a dictionary object.")
    else:
        raise EvaluationError(
            "EVAL_MISSING_REQUIRED_KEYS",
            f"NPZ file must contain keys '{feature_key}' and '{label_key}', or fallback keys 'x' and 'y', or a 'data' dictionary."
        )
    return _to_tensor_dataset(features, labels)


def _discover_public_dataset_file(dataset_name: str) -> str:
    public_dir = os.path.join(_project_root(), "dataset", dataset_name, "public")
    if not os.path.isdir(public_dir):
        raise EvaluationError(
            "EVAL_PUBLIC_DATASET_NOT_FOUND",
            f"Public dataset folder does not exist: {public_dir}"
        )

    supported_extensions = [".csv", ".json", ".tfrecord", ".npz"]
    candidates = []
    for filename in os.listdir(public_dir):
        lower_name = filename.lower()
        if any(lower_name.endswith(ext) for ext in supported_extensions):
            candidates.append(os.path.join(public_dir, filename))

    if not candidates:
        raise EvaluationError(
            "EVAL_PUBLIC_DATASET_NOT_FOUND",
            f"No public test file found in {public_dir}. Supported: CSV, JSON, TFRecord."
        )

    candidates.sort()
    preferred = [path for path in candidates if os.path.splitext(path)[1].lower() != ".npz"]
    if preferred:
        return preferred[0]
    return candidates[0]


def _load_public_dataset(dataset_name: str, feature_key: str, label_key: str) -> TensorDataset:
    file_path = _discover_public_dataset_file(dataset_name)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return _load_csv_dataset(file_path, feature_key, label_key)
    if ext == ".json":
        return _load_json_dataset(file_path, feature_key, label_key)
    if ext == ".tfrecord":
        return _load_tfrecord_dataset(file_path, feature_key, label_key)
    if ext == ".npz":
        return _load_npz_dataset(file_path, feature_key, label_key)
    raise EvaluationError("EVAL_UNSUPPORTED_DATASET_FORMAT", f"Unsupported dataset format: {ext}")


def evaluate_global_model(
    aggregated_weights: Dict[str, torch.Tensor],
    model_template: torch.nn.Module,
    dataset_name: str,
    device: str,
    batch_size: int,
    feature_key: str = "x",
    label_key: str = "y"
) -> Dict[str, float]:
    eval_model = copy.deepcopy(model_template)
    try:
        eval_model.load_state_dict(aggregated_weights, strict=True)
    except Exception as exc:
        raise EvaluationError(
            "EVAL_MODEL_STATE_MISMATCH",
            "Aggregated model weights do not match model architecture."
        ) from exc

    dataset = _load_public_dataset(dataset_name, feature_key, label_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    eval_model = eval_model.to(device)
    eval_model.eval()

    total = 0
    correct = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = eval_model(features)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if total == 0:
        raise EvaluationError("EVAL_EMPTY_DATASET", "Public dataset has no valid samples.")

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = correct / total

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }
