import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def main(predictions_path: str):
    df = pd.read_csv(predictions_path)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        raise ValueError("Predictions CSV must contain y_true and y_pred columns.")

    print(classification_report(df["y_true"], df["y_pred"], zero_division=0))

    cm = confusion_matrix(df["y_true"], df["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=45)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    main(args.predictions)
