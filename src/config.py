from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    model_name: str = "qwen/Qwen2.5-4B-Instruct"
    max_length: int = 1024
    batch_size: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 5
    test_size: float = 0.2
    seed: int = 42
    output_dir: str = "./output"
    data_dir: str = "./data/101"
    question_id: str = "8"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500


def load_config(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**{k: v for k, v in data.items() if v is not None})
