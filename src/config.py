from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    model_name: str = "qwen/Qwen2.5-4B-Instruct"
    test_size: float = 0.2
    seed: int = 42
    output_dir: str = "./output"
    data_dir: str = "./data/101"
    question_id: str = "8"
    subject_id: str = "101"

    #hyper params
    max_length: int = 1024
    batch_size: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    tolerance: float = 0.0
    beta: float = 1.0                                  # margin penalty weight for regression_loss
    lambda_reg: float = 0.4                             # regression loss weight in coral_mix_loss
    head_lr_mult: float = 5.0                            # head LR multiplier relative to backbone LR
    # NEW fields:
    pooling: str = "mean"                              # "mean" | "last" | "attention" | "multi_layer"
    pooling_num_layers: int = 2                          # number of layers for multi_layer pooling
    head_hidden_sizes: list[int] = field(default_factory=list)  # e.g. [512, 128], empty=old behavior
    head_dropout: float = 0.1
    head_type: str = "coral"                           # "coral" | "regression" | "coral_mix"
    dev_ratio: float = 0.2                              # calibration split for dev set
    calibration_method: str = "linear"                  # "none" | "linear" | "isotonic"


def load_config(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**{k: v for k, v in data.items() if v is not None})
