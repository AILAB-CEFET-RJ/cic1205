#!/usr/bin/env python3
"""
Demonstrate several model selection strategies on the beer consumption dataset,
framed as a multi-class classification problem.

The script covers:
1. Evaluating on the training set (incorrect approach).
2. Two-way holdout (single train/test split).
3. Three-way holdout (train/validation/test split).
4. k-fold cross-validation.
5. Nested cross-validation for model/parameter selection.
6. Bootstrap-based generalisation error estimation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float


def load_features_and_target(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, derive multi-class labels, and return features/target."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    df["Class1"] = np.where(df["Litres"] > 29000, 1, 0)
    df["Class2"] = np.where(df["Litres"] > 25000, 1, 0)
    df["Class3"] = np.where(df["Litres"] > 22000, 1, 0)
    df["Class4"] = np.where(df["Litres"] > 10000, 1, 0)
    df["Class"] = (df["Class1"] + df["Class2"] + df["Class3"] + df["Class4"]).astype(int)

    feature_cols = ["AvgTemp", "MinTemp", "MaxTemp", "Rainfall_mm", "Weekend"]
    features = df[feature_cols]
    target = df["Class"]
    return features, target


def build_classifier(C: float = 1.0) -> Pipeline:
    """Create a multinomial logistic regression pipeline with scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=C, max_iter=1000, multi_class="multinomial")),
        ]
    )


def evaluate_model(model, X, y) -> ClassificationMetrics:
    """Fit the model and evaluate on the provided data."""
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro")
    return ClassificationMetrics(accuracy=accuracy, macro_f1=macro_f1)


def holdout_train_test(
    X,
    y,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ClassificationMetrics:
    """Evaluate a classifier using a single train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = build_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return ClassificationMetrics(
        accuracy=accuracy_score(y_test, predictions),
        macro_f1=f1_score(y_test, predictions, average="macro"),
    )


def three_way_holdout(
    X,
    y,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[float, ClassificationMetrics, ClassificationMetrics]:
    """Tune hyperparameters with a validation split and report validation/test metrics."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train_full,
    )

    C_grid = [0.01, 0.1, 1.0, 10.0]
    best_val_metrics: ClassificationMetrics | None = None
    best_model = None
    best_C = None

    for C in C_grid:
        candidate = build_classifier(C=C)
        candidate.fit(X_train, y_train)
        val_predictions = candidate.predict(X_val)
        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_val, val_predictions),
            macro_f1=f1_score(y_val, val_predictions, average="macro"),
        )
        if best_val_metrics is None or metrics.accuracy > best_val_metrics.accuracy:
            best_val_metrics = metrics
            best_model = candidate
            best_C = C

    test_predictions = best_model.predict(X_test)
    test_metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_test, test_predictions),
        macro_f1=f1_score(y_test, test_predictions, average="macro"),
    )
    return best_C, best_val_metrics, test_metrics


def kfold_cross_validation(
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Return mean accuracy and macro-F1 across k-fold splits."""
    model = build_classifier()
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    macro_f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return float(np.mean(accuracy_scores)), float(np.mean(macro_f1_scores))


def nested_cross_validation(
    X,
    y,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> Tuple[float, Iterable[dict]]:
    """Perform nested CV for hyperparameter selection and report outer accuracy."""
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    pipeline = build_classifier()
    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
    }

    outer_scores = []
    chosen_params = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring="accuracy",
        )
        grid.fit(X_train, y_train)

        predictions = grid.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outer_scores.append(accuracy)
        chosen_params.append(
            {
                "best_params": grid.best_params_,
                "inner_accuracy": grid.best_score_,
            }
        )

    return float(np.mean(outer_scores)), chosen_params


def bootstrap_accuracy(
    X,
    y,
    n_iterations: int = 200,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """Estimate accuracy and macro-F1 using bootstrap resampling."""
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    X_values = X.values
    y_values = y.values
    accuracies = []
    macro_f1_scores = []

    for _ in range(n_iterations):
        sample_indices = rng.integers(0, n_samples, size=n_samples)
        unique_oob_mask = np.ones(n_samples, dtype=bool)
        unique_oob_mask[np.unique(sample_indices)] = False
        oob_indices = np.where(unique_oob_mask)[0]
        if len(oob_indices) == 0:
            continue

        X_sample = X_values[sample_indices]
        y_sample = y_values[sample_indices]
        if len(np.unique(y_sample)) < 2:
            continue  # logistic regression needs at least two classes

        model = build_classifier()
        model.fit(X_sample, y_sample)
        X_oob = X_values[oob_indices]
        y_oob = y_values[oob_indices]
        predictions = model.predict(X_oob)

        accuracies.append(accuracy_score(y_oob, predictions))
        macro_f1_scores.append(f1_score(y_oob, predictions, average="macro"))

    if not accuracies:
        raise RuntimeError("Bootstrap produced no usable out-of-bag samples.")

    return (
        float(np.mean(accuracies)),
        float(np.std(accuracies)),
        float(np.mean(macro_f1_scores)),
        float(np.std(macro_f1_scores)),
    )


def main(csv_path: str) -> None:
    X, y = load_features_and_target(csv_path)
    print(f"Loaded {len(X)} records with {X.shape[1]} features from {csv_path}")
    print("Class distribution:")
    for cls, count in y.value_counts().sort_index().items():
        print(f"   Class {cls}: {count}")

    print("\n1) Train-set evaluation (incorrect baseline)")
    train_metrics = evaluate_model(build_classifier(), X, y)
    print(
        f"   Accuracy: {train_metrics.accuracy:.3f} | Macro-F1: {train_metrics.macro_f1:.3f}"
    )

    print("\n2) Two-way holdout (train/test split)")
    holdout_metrics = holdout_train_test(X, y)
    print(
        f"   Test Accuracy: {holdout_metrics.accuracy:.3f} | Test Macro-F1: {holdout_metrics.macro_f1:.3f}"
    )

    print("\n3) Three-way holdout (train/validation/test)")
    best_C, val_metrics, test_metrics = three_way_holdout(X, y)
    print(f"   Best validation C: {best_C}")
    print(
        f"   Validation Accuracy: {val_metrics.accuracy:.3f} | Validation Macro-F1: {val_metrics.macro_f1:.3f}"
    )
    print(
        f"   Test Accuracy: {test_metrics.accuracy:.3f} | Test Macro-F1: {test_metrics.macro_f1:.3f}"
    )

    print("\n4) k-fold cross-validation")
    cv_accuracy, cv_macro_f1 = kfold_cross_validation(X, y)
    print(f"   Mean Accuracy: {cv_accuracy:.3f} | Mean Macro-F1: {cv_macro_f1:.3f}")

    print("\n5) Nested cross-validation (model + hyperparameter selection)")
    nested_accuracy, chosen_params = nested_cross_validation(X, y)
    print(f"   Mean outer Accuracy: {nested_accuracy:.3f}")
    print("   Selected hyperparameters per outer fold:")
    for idx, params in enumerate(chosen_params, start=1):
        print(f"     Fold {idx}: {params}")

    print("\n6) Bootstrap-based estimation")
    (
        bootstrap_acc_mean,
        bootstrap_acc_std,
        bootstrap_f1_mean,
        bootstrap_f1_std,
    ) = bootstrap_accuracy(X, y)
    print(
        f"   Bootstrap Accuracy: {bootstrap_acc_mean:.3f} ± {bootstrap_acc_std:.3f} (std)"
    )
    print(
        f"   Bootstrap Macro-F1: {bootstrap_f1_mean:.3f} ± {bootstrap_f1_std:.3f} (std)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Illustrate model selection strategies.")
    parser.add_argument(
        "--csv",
        default="notebooks/BIP/Beerconsumption.csv",
        help="Path to the Beer consumption CSV file.",
    )
    args = parser.parse_args()
    main(args.csv)
