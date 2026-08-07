from __future__ import annotations

from typing import Iterable, Sequence


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _labels(y_true: Sequence[str], y_pred: Sequence[str] | None = None) -> list[str]:
    values = set(y_true)
    if y_pred is not None:
        values.update(y_pred)
    return sorted(values)


def per_class_prf(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] | None = None) -> dict[str, dict[str, float | int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = list(labels) if labels is not None else _labels(y_true, y_pred)
    out: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(t == label for t in y_true)}
    return out


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] | None = None) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = list(labels) if labels is not None else sorted(set(y_true))
    recalls = []
    for label in labels:
        support = sum(t == label for t in y_true)
        if support:
            recalls.append(sum(t == label and p == label for t, p in zip(y_true, y_pred)) / support)
    return _safe_div(sum(recalls), len(recalls))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] | None = None) -> float:
    stats = per_class_prf(y_true, y_pred, labels)
    return _safe_div(sum(float(v["f1"]) for v in stats.values()), len(stats))


def binary_f1(y_true: Sequence[str], y_pred: Sequence[str], positive: str = "anomalous") -> float:
    stats = per_class_prf(y_true, y_pred, [positive])
    return float(stats[positive]["f1"])


def joint_accuracy(anomaly_true: Sequence[str], anomaly_pred: Sequence[str], defect_true: Sequence[str], defect_pred: Sequence[str]) -> float:
    lengths = {len(anomaly_true), len(anomaly_pred), len(defect_true), len(defect_pred)}
    if len(lengths) != 1:
        raise ValueError("all inputs must have the same length")
    correct = sum(a_t == a_p and d_t == d_p for a_t, a_p, d_t, d_p in zip(anomaly_true, anomaly_pred, defect_true, defect_pred))
    return _safe_div(correct, len(anomaly_true))


def false_defect_rate(anomaly_true: Sequence[str], defect_pred: Sequence[str], normal_label: str = "normal") -> float:
    if len(anomaly_true) != len(defect_pred):
        raise ValueError("inputs must have the same length")
    normal_indices = [i for i, a in enumerate(anomaly_true) if a == normal_label]
    false_defects = sum(defect_pred[i] != normal_label for i in normal_indices)
    return _safe_div(false_defects, len(normal_indices))


def consistency_error(anomaly_pred: Sequence[str], defect_pred: Sequence[str], normal_label: str = "normal", anomalous_label: str = "anomalous") -> float:
    if len(anomaly_pred) != len(defect_pred):
        raise ValueError("inputs must have the same length")
    errors = 0
    for anomaly, defect in zip(anomaly_pred, defect_pred):
        if anomaly == normal_label and defect != normal_label:
            errors += 1
        elif anomaly == anomalous_label and defect == normal_label:
            errors += 1
    return _safe_div(errors, len(anomaly_pred))


def invalid_output_rate(valid_mask: Sequence[bool]) -> float:
    return _safe_div(sum(not value for value in valid_mask), len(valid_mask))


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] | None = None) -> dict[str, dict[str, int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = list(labels) if labels is not None else _labels(y_true, y_pred)
    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in zip(y_true, y_pred):
        if truth not in matrix:
            matrix[truth] = {p: 0 for p in labels}
        if pred not in matrix[truth]:
            for row in matrix.values():
                row.setdefault(pred, 0)
        matrix[truth][pred] += 1
    return matrix
