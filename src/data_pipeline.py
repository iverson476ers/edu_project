from __future__ import annotations

import glob
import json

import numpy as np
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


def perturb_text(
    text: str, rng: np.random.Generator, perturb_prob: float = 0.5
) -> str:
    """Apply mild perturbation to a duplicated answer text.

    With probability perturb_prob, applies either or both:
    (1) insert 3-5 spaces at a random position
    (2) append a full stop at the end
    """
    if rng.random() > perturb_prob:
        return text
    if rng.random() < 0.5:
        n_spaces = int(rng.integers(3, 6))
        pos = int(rng.integers(0, len(text) + 1))
        text = text[:pos] + " " * n_spaces + text[pos:]
    if rng.random() < 0.5:
        text = text + "\u3002"
    return text


def oversample_rare_scores(
    df: pd.DataFrame,
    min_count: int = 100,
    perturb_prob: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Oversample score points below min_count with text perturbation.

    Original samples are kept untouched; only duplicated copies are
    perturbed to reduce exact-duplicate memorization.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("label"):
        if len(group) < min_count:
            n_dup = min_count - len(group)
            dup = group.sample(n=n_dup, replace=True, random_state=seed)
            dup = dup.copy()
            dup["text"] = dup["text"].apply(
                lambda t: perturb_text(t, rng, perturb_prob)
            )
            parts.append(dup)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


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


def load_question_config(data_dir: str, subject_id: str, question_id: str) -> dict:
    """Read question JSON config, e.g. {full_score, score_precision, ...}."""
    path = f"{data_dir}/question/question{subject_id}_{question_id}.json"
    with open(path, "r") as f:
        return json.load(f)


def load_validation_data(data_dir: str, subject_id: str, question_id: str) -> pd.DataFrame:
    """Load validation data from CSV. Expected columns: BMH, ZZCJ (label)."""
    pattern = f"{data_dir}/calibration/validation{subject_id}_{question_id}.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No validation file found matching: {pattern}")
    df = pd.read_csv(files[0])
    df = df[["BMH", "ZZCJ"]].copy()
    df.columns = ["BMH", "label"]
    df["label"] = df["label"].astype(float)
    return df.dropna()


def load_and_split_data(
    config: "TrainingConfig",
) -> tuple:
    """Load data and return (train_df, dev_df, test_df, score_points, question_config).

    train_df = (1 - dev_ratio) of calibration data (merged with answers).
    dev_df   = dev_ratio of calibration data for model selection & calibration.
    test_df  = external validation set loaded from CSV file.
    """
    from src.config import TrainingConfig

    answer_path = f"{config.data_dir}/answer/answer{config.subject_id}_{config.question_id}.xlsx"
    q_config = load_question_config(config.data_dir, config.subject_id, config.question_id)

    answer_df = load_answer_data(answer_path)
    calib_df = load_calibration_data(config.data_dir, config.question_id)
    full_calib = merge_data(answer_df, calib_df)
    score_points = get_score_points(full_calib)

    # Split calibration into train / dev (stratified by score)
    if config.dev_ratio > 0:
        train_df, dev_df = train_test_split(
            full_calib, test_size=config.dev_ratio, random_state=config.seed,
            stratify=full_calib["label_idx"]
        )
        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
    else:
        train_df = full_calib.reset_index(drop=True)
        dev_df = None

    # Load external validation set
    val_df = load_validation_data(config.data_dir, config.subject_id, config.question_id)
    test_df = merge_data(answer_df, val_df)

    print(f"Question: {q_config.get('subject_name', '')} {q_config.get('question_type', '')}")
    print(f"  full_score={q_config['full_score']}, precision={q_config['score_precision']}")
    print(f"Train (calibration): {len(train_df)}, Dev: {len(dev_df) if dev_df is not None else 0}, "
          f"Test (validation): {len(test_df)}")
    print(f"Score points ({len(score_points)}): {score_points[:5]}...{score_points[-3:]}")

    return train_df, dev_df, test_df, score_points, q_config
