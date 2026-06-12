from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset with safer defaults for mixed-type columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully empty columns and duplicate rows."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()
    return df


def split_features_target(df: pd.DataFrame, target_column: str):
    """Split dataframe into X and y."""
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
