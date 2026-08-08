import pandas as pd

def load_csv(path, required_columns=None):
    df = pd.read_csv(path)
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    return df


def load_supervised_csv(path, text_column, target_column):
    df = load_csv(path, required_columns=[text_column, target_column])
    df = df.dropna(subset=[text_column, target_column]).copy()
    df[text_column] = df[text_column].astype(str)
    df[target_column] = df[target_column].astype(str)
    return df
