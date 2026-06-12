import argparse
import json
from pathlib import Path

import joblib
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import load_dataset, basic_cleaning, split_features_target
from .features import build_preprocessor


def build_model(config: dict):
    model_type = config["model"].get("type", "lightgbm").lower()
    params = config["model"].get("parameters", {})

    if model_type == "decision_tree":
        return DecisionTreeClassifier(random_state=params.get("random_state", 42))
    if model_type == "lightgbm":
        return LGBMClassifier(**params)
    raise ValueError(f"Unsupported model type: {model_type}")


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = load_dataset(config["data"]["raw_path"])
    df = basic_cleaning(df)

    drop_columns = config["data"].get("drop_columns", [])
    existing_drop_columns = [c for c in drop_columns if c in df.columns]
    if existing_drop_columns:
        df = df.drop(columns=existing_drop_columns)

    X, y = split_features_target(df, config["data"]["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["data"].get("random_state", 42),
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", build_model(config)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    output_cfg = config["output"]
    Path(output_cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(output_cfg["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_cfg["model_path"])
    with open(output_cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
