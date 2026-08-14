from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence


def _row_id(row: Mapping[str, object]) -> str:
    return str(row.get("id") or row.get("core_id") or row)


def _binary_answer(row: Mapping[str, object]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        value = answer.get("anomaly")
    else:
        value = answer
    value = str(value)
    if value not in {"normal", "anomalous"}:
        raise ValueError(f"unsupported binary answer: {value!r}")
    return value


def _round_robin_by_asset(
    rows: Sequence[Mapping[str, object]],
    n: int,
    *,
    seed: int,
) -> list[dict]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["asset_category"])].append(row)
    if not grouped:
        return []

    rng = random.Random(seed)
    for values in grouped.values():
        values.sort(key=_row_id)
        rng.shuffle(values)

    assets = sorted(grouped)
    selected: list[dict] = []
    cursor = 0
    active = assets[:]
    while len(selected) < n and active:
        next_active = []
        for asset in active:
            values = grouped[asset]
            if cursor < len(values):
                selected.append(dict(values[cursor]))
                if len(selected) == n:
                    break
            if cursor + 1 < len(values):
                next_active.append(asset)
        cursor += 1
        active = next_active
    return selected


def balanced_binary_sample(
    rows: Sequence[Mapping[str, object]],
    n: int = 100,
    *,
    seed: int = 42,
) -> list[dict]:
    """Return an exact 50/50 normal/anomalous sample when both classes are available.

    Within each binary class, rows are selected round-robin across asset categories
    so the sample does not collapse onto a few large assets.
    """
    if n <= 0:
        return []

    by_label: dict[str, list[Mapping[str, object]]] = {"normal": [], "anomalous": []}
    for row in rows:
        by_label[_binary_answer(row)].append(row)

    normal_n = n // 2
    anomalous_n = n - normal_n
    quotas = {"normal": normal_n, "anomalous": anomalous_n}

    for label, quota in quotas.items():
        if len(by_label[label]) < quota:
            raise ValueError(
                f"not enough {label} rows for balanced sample: need {quota}, "
                f"have {len(by_label[label])}"
            )

    selected = []
    for offset, label in enumerate(("normal", "anomalous")):
        selected.extend(
            _round_robin_by_asset(
                by_label[label],
                quotas[label],
                seed=seed + offset * 10_000,
            )
        )

    rng = random.Random(seed + 99_999)
    rng.shuffle(selected)
    return selected


def matched_one_shot_sample(
    zero_shot_rows: Sequence[Mapping[str, object]],
    one_shot_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Match one-shot rows to an already selected zero-shot sample by QA id."""
    by_id = {str(row["id"]): row for row in one_shot_rows}
    out = []
    for row in zero_shot_rows:
        row_id = str(row["id"])
        if row_id not in by_id:
            raise KeyError(f"missing one-shot row for id {row_id}")
        out.append(dict(by_id[row_id]))
    return out


def balanced_structured_sample(
    rows: Sequence[Mapping[str, object]],
    n: int = 100,
    *,
    seed: int = 42,
) -> list[dict]:
    """Balance structured QA across assets, then defect classes inside each asset."""
    if n >= len(rows):
        return [dict(r) for r in rows]
    if n <= 0:
        return []

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["asset_category"])].append(row)

    assets = sorted(grouped)
    base, extra = divmod(n, len(assets))
    selected: list[dict] = []

    for asset_index, asset in enumerate(assets):
        take = base + (1 if asset_index < extra else 0)
        label_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in grouped[asset]:
            answer = row.get("answer")
            if not isinstance(answer, dict):
                raise ValueError("structured row answer must be an object")
            label_groups[str(answer["defect_class"])].append(row)

        rng = random.Random(f"{seed}:{asset}")
        for values in label_groups.values():
            values.sort(key=_row_id)
            rng.shuffle(values)

        labels = sorted(label_groups)
        cursor = 0
        active = labels[:]
        pool: list[Mapping[str, object]] = []
        while len(pool) < take and active:
            next_active = []
            for label in active:
                values = label_groups[label]
                if cursor < len(values):
                    pool.append(values[cursor])
                    if len(pool) == take:
                        break
                if cursor + 1 < len(values):
                    next_active.append(label)
            cursor += 1
            active = next_active

        if len(pool) < take:
            raise ValueError(
                f"not enough structured rows for asset {asset}: need {take}, got {len(pool)}"
            )
        selected.extend(dict(r) for r in pool)

    return selected[:n]
