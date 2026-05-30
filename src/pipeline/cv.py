"""CV分割戦略。"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from pipeline.types import Rows


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: list[int]
    valid_idx: list[int]
    valid_groups: tuple[str, ...]


def make_cv_splits(
    train: Rows,
    *,
    strategy: str,
    group_col: str = "樹種",
    target_col: str = "含水率",
    n_splits: int = 5,
    seed: int = 42,
) -> list[Fold]:
    if strategy in {"none", "", None}:
        return []
    if strategy == "group_species":
        return group_kfold(train, group_col=group_col, n_splits=n_splits)
    if strategy == "leave_one_species":
        return leave_one_group_out(train, group_col=group_col)
    if strategy == "repeated_leave_one_species":
        return repeated_leave_one_group_out(
            train, group_col=group_col, n_repeats=n_splits, seed=seed
        )
    if strategy == "moisture_quantile":
        return moisture_quantile_folds(train, target_col=target_col, n_splits=n_splits)
    if strategy == "random":
        return random_kfold(train, n_splits=n_splits, seed=seed)
    raise ValueError(f"未知のCV戦略です: {strategy}")


def group_kfold(train: Rows, *, group_col: str, n_splits: int) -> list[Fold]:
    """groupを分割単位にしたKFold。

    scikit-learnへ依存せず、groupごとの件数がなるべく均等になるよう貪欲に割り当てる。
    """
    group_counts: dict[str, int] = {}
    for row in train:
        group = str(row[group_col])
        group_counts[group] = group_counts.get(group, 0) + 1

    ordered_groups = sorted(group_counts.items(), key=lambda item: item[1], reverse=True)
    fold_groups: list[list[str]] = [[] for _ in range(min(n_splits, len(ordered_groups)))]
    fold_sizes = [0 for _ in fold_groups]

    for group, count in ordered_groups:
        target_fold = fold_sizes.index(min(fold_sizes))
        fold_groups[target_fold].append(str(group))
        fold_sizes[target_fold] += int(count)

    all_idx = list(range(len(train)))
    folds: list[Fold] = []
    for fold_id, valid_groups in enumerate(fold_groups):
        valid_group_set = set(valid_groups)
        valid_idx = [i for i, row in enumerate(train) if str(row[group_col]) in valid_group_set]
        train_idx = [i for i in all_idx if i not in set(valid_idx)]
        folds.append(
            Fold(
                fold_id=fold_id,
                train_idx=train_idx,
                valid_idx=valid_idx,
                valid_groups=tuple(sorted(valid_groups)),
            )
        )
    return folds


def moisture_quantile_folds(
    train: Rows,
    *,
    target_col: str,
    n_splits: int,
) -> list[Fold]:
    """含水率の分位で分割（樹種非重複コンペ向け・外挿評価）。"""
    indexed = [(i, float(row[target_col])) for i, row in enumerate(train)]
    indexed.sort(key=lambda item: item[1])
    n = len(indexed)
    n_splits = max(2, min(n_splits, n))
    chunks: list[list[int]] = [[] for _ in range(n_splits)]
    for rank, (idx, _) in enumerate(indexed):
        chunks[rank % n_splits].append(idx)
    all_idx = list(range(len(train)))
    folds: list[Fold] = []
    for fold_id, valid_idx in enumerate(chunks):
        valid_idx = sorted(valid_idx)
        valid_set = set(valid_idx)
        folds.append(
            Fold(
                fold_id=fold_id,
                train_idx=[i for i in all_idx if i not in valid_set],
                valid_idx=valid_idx,
                valid_groups=(),
            )
        )
    return folds


def repeated_leave_one_group_out(
    train: Rows,
    *,
    group_col: str,
    n_repeats: int,
    seed: int,
) -> list[Fold]:
    """樹種LOOを n_repeats 回繰り返し（各回で検証樹種の順序をシャッフル）。"""
    rng = Random(seed)
    groups = sorted({str(row[group_col]) for row in train})
    all_idx = list(range(len(train)))
    folds: list[Fold] = []
    fold_id = 0
    for _rep in range(max(1, n_repeats)):
        order = groups[:]
        rng.shuffle(order)
        for group in order:
            valid_idx = [i for i, row in enumerate(train) if str(row[group_col]) == group]
            valid_set = set(valid_idx)
            folds.append(
                Fold(
                    fold_id=fold_id,
                    train_idx=[i for i in all_idx if i not in valid_set],
                    valid_idx=valid_idx,
                    valid_groups=(str(group),),
                )
            )
            fold_id += 1
    return folds


def leave_one_group_out(train: Rows, *, group_col: str) -> list[Fold]:
    groups = [str(row[group_col]) for row in train]
    all_idx = list(range(len(train)))
    folds: list[Fold] = []
    for fold_id, group in enumerate(sorted(set(groups))):
        valid_idx = [i for i, current in enumerate(groups) if current == group]
        valid_set = set(valid_idx)
        folds.append(
            Fold(
                fold_id=fold_id,
                train_idx=[i for i in all_idx if i not in valid_set],
                valid_idx=valid_idx,
                valid_groups=(str(group),),
            )
        )
    return folds


def random_kfold(train: Rows, *, n_splits: int, seed: int) -> list[Fold]:
    rng = Random(seed)
    indices = list(range(len(train)))
    rng.shuffle(indices)
    chunks = [indices[i::n_splits] for i in range(n_splits)]
    all_idx = list(range(len(train)))
    folds: list[Fold] = []
    for fold_id, valid_idx in enumerate(chunks):
        valid_idx = sorted(valid_idx)
        valid_set = set(valid_idx)
        train_idx = [i for i in all_idx if i not in valid_set]
        folds.append(
            Fold(
                fold_id=fold_id,
                train_idx=train_idx,
                valid_idx=valid_idx,
                valid_groups=(),
            )
        )
    return folds
