"""Evaluate species-aware weighted kNN without changing production search.

Hyperparameters are selected only on the calibration queries. The held-out
test split is evaluated once for the selected configuration.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from scripts.evaluate_species_prototypes import (  # noqa: E402
    group_raw_distances,
    normalize_and_fuse,
)
from scripts.tune_weights import evaluate, robust_normalize, split_queries  # noqa: E402


KNN_CONFIGS = (
    (10, 1),
    (10, 2),
    (20, 1),
    (20, 2),
    (30, 2),
    (50, 2),
    (50, 3),
)
BETAS = (0.0, 0.5, 1.0)


def weighted_species_scores(
    visual_distance: np.ndarray,
    label_indices: np.ndarray,
    species_counts: np.ndarray,
    *,
    neighbors: int,
    power: int,
    beta: float,
) -> np.ndarray:
    """Accumulate similarity evidence from the nearest visual neighbors."""

    image_count = len(label_indices)
    neighbors = min(neighbors, image_count - 1)
    nearest = np.argpartition(visual_distance, neighbors - 1, axis=1)[:, :neighbors]
    nearest_distance = np.take_along_axis(visual_distance, nearest, axis=1)
    evidence = np.clip(1.0 - nearest_distance, 0.0, 1.0) ** power

    scores = np.zeros((image_count, len(species_counts)), dtype=np.float32)
    rows = np.repeat(np.arange(image_count), neighbors)
    columns = label_indices[nearest].reshape(-1)
    np.add.at(scores, (rows, columns), evidence.reshape(-1))

    if beta > 0.0:
        # During leave-one-out evaluation, the query itself is not in the
        # gallery, so its own species has one fewer available image.
        gallery_counts = np.broadcast_to(species_counts, scores.shape).copy()
        gallery_counts[np.arange(image_count), label_indices] -= 1
        scores /= np.maximum(gallery_counts, 1.0) ** beta
    return scores


def species_aware_ranks(
    species_scores: np.ndarray,
    visual_distance: np.ndarray,
    label_indices: np.ndarray,
) -> np.ndarray:
    """Rank species by evidence and images within each species visually."""

    candidate_scores = species_scores[:, label_indices]
    ranks = np.empty_like(visual_distance, dtype=np.float32)
    ordinal = np.arange(len(label_indices), dtype=np.float32)
    for query_index in range(len(label_indices)):
        order = np.lexsort(
            (visual_distance[query_index], -candidate_scores[query_index])
        )
        ranks[query_index, order] = ordinal
    np.fill_diagonal(ranks, np.inf)
    return ranks


def prediction_metrics(
    scores: np.ndarray,
    label_indices: np.ndarray,
    queries: np.ndarray,
) -> dict[str, float]:
    order = np.argsort(-scores[queries], axis=1, kind="stable")
    truth = label_indices[queries]
    predictions = order[:, 0]
    return {
        "top1_accuracy": float(np.mean(predictions == truth)),
        "top3_accuracy": float(np.mean(np.any(order[:, :3] == truth[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(order[:, :5] == truth[:, None], axis=1))),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
    }


def selection_key(metrics) -> tuple[float, float, float, float]:
    """Method 5 targets P@1 first; mAP and broader precision break ties."""

    return metrics.p1, metrics.map10, metrics.p5, metrics.p10


def main() -> int:
    settings = get_settings()
    with SessionLocal() as session:
        records = list(
            session.scalars(
                select(ImageRecord)
                .options(joinedload(ImageRecord.feature))
                .join(ImageFeature)
                .where(
                    ImageRecord.species.is_not(None),
                    ImageFeature.feature_version == settings.feature_version,
                )
                .order_by(ImageRecord.id)
            ).all()
        )

    labels = np.asarray([str(record.species) for record in records], dtype=object)
    species = np.asarray(sorted(set(labels.tolist())), dtype=object)
    species_to_index = {name: index for index, name in enumerate(species)}
    label_indices = np.asarray([species_to_index[name] for name in labels], dtype=np.int32)
    species_counts = np.bincount(label_indices, minlength=len(species)).astype(np.float32)
    features = (
        np.asarray([record.feature.color_vector for record in records], dtype=np.float32),
        np.asarray([record.feature.texture_vector for record in records], dtype=np.float32),
        np.asarray([record.feature.shape_vector for record in records], dtype=np.float32),
    )
    calibration, test = split_queries(labels, settings.random_seed)
    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"species={len(species)}, calibration={len(calibration)}, test={len(test)}"
    )

    print("Computing the current visual distance matrix...")
    visual_groups = group_raw_distances(features, features, settings)
    visual_ranges = [robust_normalize(group)[1]["p95"] for group in visual_groups]
    visual_distance = normalize_and_fuse(visual_groups, visual_ranges, settings)
    np.fill_diagonal(visual_distance, np.inf)

    baseline_calibration = evaluate(visual_distance, labels, calibration)
    baseline_test = evaluate(visual_distance, labels, test)
    calibration_results: dict[str, dict] = {}
    best_config: tuple[int, int, float] | None = None
    best_metrics = None

    print(
        f"\n{'N':>4} {'p':>3} {'beta':>6} {'P@1':>8} {'P@5':>8} "
        f"{'P@10':>8} {'R@10':>8} {'mAP@10':>8}"
    )
    for neighbors, power in KNN_CONFIGS:
        for beta in BETAS:
            scores = weighted_species_scores(
                visual_distance,
                label_indices,
                species_counts,
                neighbors=neighbors,
                power=power,
                beta=beta,
            )
            ranks = species_aware_ranks(scores, visual_distance, label_indices)
            metrics = evaluate(ranks, labels, calibration)
            key = f"N={neighbors},p={power},beta={beta:g}"
            calibration_results[key] = {
                "neighbors": neighbors,
                "power": power,
                "beta": beta,
                "retrieval": asdict(metrics),
                "species_prediction": prediction_metrics(
                    scores, label_indices, calibration
                ),
            }
            print(
                f"{neighbors:4d} {power:3d} {beta:6.1f} {metrics.p1:8.4f} "
                f"{metrics.p5:8.4f} {metrics.p10:8.4f} {metrics.recall10:8.4f} "
                f"{metrics.map10:8.4f}"
            )
            if best_metrics is None or selection_key(metrics) > selection_key(best_metrics):
                best_config = (neighbors, power, beta)
                best_metrics = metrics

    assert best_config is not None and best_metrics is not None
    neighbors, power, beta = best_config
    selected_scores = weighted_species_scores(
        visual_distance,
        label_indices,
        species_counts,
        neighbors=neighbors,
        power=power,
        beta=beta,
    )
    selected_ranks = species_aware_ranks(
        selected_scores, visual_distance, label_indices
    )
    selected_test = evaluate(selected_ranks, labels, test)
    selected_prediction_test = prediction_metrics(selected_scores, label_indices, test)

    print(
        f"\nSelected on calibration P@1: N={neighbors}, p={power}, beta={beta:g}"
    )
    print(
        f"Test baseline: P@1={baseline_test.p1:.4f}, P@5={baseline_test.p5:.4f}, "
        f"P@10={baseline_test.p10:.4f}, R@10={baseline_test.recall10:.4f}, "
        f"mAP@10={baseline_test.map10:.4f}"
    )
    print(
        f"Test weighted kNN: P@1={selected_test.p1:.4f}, P@5={selected_test.p5:.4f}, "
        f"P@10={selected_test.p10:.4f}, R@10={selected_test.recall10:.4f}, "
        f"mAP@10={selected_test.map10:.4f}"
    )

    report = {
        "feature_version": settings.feature_version,
        "protocol": (
            "leave-one-out visual neighbors; tune N/p/beta on calibration only; "
            "rank species by weighted evidence and images within species visually"
        ),
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "species": len(species),
            "calibration": len(calibration),
            "test": len(test),
        },
        "selection_metric": "calibration P@1, then mAP@10/P@5/P@10",
        "baseline": {
            "calibration": asdict(baseline_calibration),
            "test": asdict(baseline_test),
        },
        "calibration_grid": calibration_results,
        "selected": {
            "neighbors": neighbors,
            "power": power,
            "beta": beta,
            "calibration_retrieval": asdict(best_metrics),
            "test_retrieval": asdict(selected_test),
            "test_species_prediction": selected_prediction_test,
        },
    }
    report_path = settings.scaler_dir / "weighted_knn_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
