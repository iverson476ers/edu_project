from __future__ import annotations

import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_answer_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df[["BMH", "OCR文字结果"]].copy()
    df = df.dropna(subset=["OCR文字结果"])
    df.columns = ["BMH", "text"]
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


def load_calibration_data(data_dir: str, question_id: str) -> pd.DataFrame:
    pattern = f"{data_dir}/calibration/calibration*_{question_id}.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No calibration file found matching: {pattern}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    calib = pd.concat(dfs, ignore_index=True)
    calib = calib[["BMH", "ZZCJ"]].copy()
    calib.columns = ["BMH", "label"]
    calib["label"] = calib["label"].astype(float)
    return calib.dropna()


def get_score_points(df: pd.DataFrame) -> list[float]:
    return sorted(df["label"].unique().tolist())


def merge_data(answer_df: pd.DataFrame, calib_df: pd.DataFrame) -> pd.DataFrame:
    merged = answer_df.merge(calib_df, on="BMH", how="inner")
    score_points = get_score_points(merged)
    score_to_idx = {s: i for i, s in enumerate(score_points)}
    merged["label_idx"] = merged["label"].map(score_to_idx)
    return merged[["BMH", "text", "label", "label_idx"]].reset_index(drop=True)


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=None
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ASAGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer=None, max_length: int = 1024):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.label_indices = df["label_idx"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            # Dummy mode: return text as-is for pipeline testing
            input_ids = text
            attention_mask = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": self.labels[idx],
            "label_idx": self.label_indices[idx],
            "text": self.texts[idx],
        }


def load_and_split_data(
    config: "TrainingConfig",
) -> tuple:
    """Load, merge, split data and return (train_df, test_df, score_points)."""
    from src.config import TrainingConfig

    answer_path = f"{config.data_dir}/answer/answer101_{config.question_id}.xlsx"

    answer_df = load_answer_data(answer_path)
    calib_df = load_calibration_data(config.data_dir, config.question_id)
    merged_df = merge_data(answer_df, calib_df)
    score_points = get_score_points(merged_df)
    train_df, test_df = split_data(merged_df, config.test_size, config.seed)

    print(f"Total labeled samples: {len(merged_df)}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Score points ({len(score_points)}): {score_points[:5]}...{score_points[-3:]}")

    return train_df, test_df, score_points
