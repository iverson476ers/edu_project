from __future__ import annotations

import torch
from transformers import AutoTokenizer

from src.calibration import apply_calibration, snap_to_score_points
from src.model import OrdinalScorer, prediction_to_score, regression_to_score


def load_scorer(checkpoint_path: str, device: str = "cuda"):
    """Load a saved OrdinalScorer checkpoint.

    Returns (model, tokenizer, score_points, max_length, full_score, head_type).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    score_points = ckpt["score_points"]
    full_score = ckpt.get("full_score", float(score_points[-1]))
    head_type = getattr(config, "head_type", "coral")
    calibration = ckpt.get("calibration")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Rebuild model structure: backbone + LoRA + Head, then load weights
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.config.use_cache = False

    hidden_dim = backbone.config.hidden_size
    from src.pooling import build_pooling
    pooling_strategy = getattr(config, "pooling", "mean")
    pooling_num_layers = getattr(config, "pooling_num_layers", 4)
    pooling = build_pooling(pooling_strategy, hidden_dim, num_layers=pooling_num_layers)
    head_config = {
        "hidden_sizes": getattr(config, "head_hidden_sizes", []),
        "dropout": getattr(config, "head_dropout", 0.1),
    }
    model = OrdinalScorer(backbone, hidden_dim, len(score_points),
                          pooling=pooling, head_config=head_config,
                          head_type=head_type)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, score_points, config.max_length, full_score, head_type, calibration


def predict(
    text: str,
    model: OrdinalScorer,
    tokenizer,
    score_points: list[float],
    device: str = "cuda",
    max_length: int = 2048,
    full_score: float | None = None,
    calibration: dict | None = None,
) -> float | tuple[float, float]:
    """Predict score for a single text.

    For coral_mix head, returns (coral_score, reg_score).
    """
    encoded = tokenizer(
        text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        head_type = getattr(model, "head_type", "coral")
        if head_type == "coral_mix":
            ordinal_logits, reg_value = output
            probs = torch.sigmoid(ordinal_logits)
            coral_score = prediction_to_score(probs.cpu(), score_points)[0]
            reg_score = round(regression_to_score(reg_value.cpu(), full_score)[0] / 0.5) * 0.5
            coral_score = snap_to_score_points(apply_calibration([coral_score], calibration), score_points)[0]
            reg_score = snap_to_score_points(apply_calibration([reg_score], calibration), score_points)[0]
            return coral_score, reg_score
        elif head_type == "regression":
            scores = regression_to_score(output.cpu(), full_score)
        else:
            probs = torch.sigmoid(output)
            scores = prediction_to_score(probs.cpu(), score_points)
    scores = snap_to_score_points(apply_calibration(scores, calibration), score_points)
    return scores[0]


def predict_batch(
    texts: list[str],
    model: OrdinalScorer,
    tokenizer,
    score_points: list[float],
    device: str = "cuda",
    max_length: int = 2048,
    full_score: float | None = None,
    calibration: dict | None = None,
) -> list[float] | tuple[list[float], list[float]]:
    """Predict scores for a batch of texts in one forward pass.

    For coral_mix head, returns (coral_scores, reg_scores).
    """
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        head_type = getattr(model, "head_type", "coral")
        if head_type == "coral_mix":
            ordinal_logits, reg_value = output
            probs = torch.sigmoid(ordinal_logits).cpu()
            coral_scores = prediction_to_score(probs, score_points)
            reg_scores = [round(s / 0.5) * 0.5 for s in regression_to_score(reg_value.cpu(), full_score)]
            coral_scores = snap_to_score_points(apply_calibration(coral_scores, calibration), score_points)
            reg_scores = snap_to_score_points(apply_calibration(reg_scores, calibration), score_points)
            return coral_scores, reg_scores
        elif head_type == "regression":
            scores = regression_to_score(output.cpu(), full_score)
        else:
            probs = torch.sigmoid(output).cpu()
            scores = prediction_to_score(probs, score_points)
    scores = snap_to_score_points(apply_calibration(scores, calibration), score_points)
    return scores


# ============================================================
# CLI entry point: python -m src.infer --checkpoint ... --input ... --output ...
# ============================================================
if __name__ == "__main__":
    import argparse
    import sys

    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Batch inference for ASAG scoring")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input xlsx file (must have BMH and OCR文字结果 columns)")
    parser.add_argument("--output", type=str, default="result.xlsx",
                        help="Output xlsx file (default: result.xlsx)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference (default: 16)")
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, score_points, max_length, full_score, head_type, calibration = load_scorer(
        args.checkpoint, device=args.device
    )
    print(f"Model loaded, head={head_type}, score_points={len(score_points)}, max_length={max_length}")

    # Read input
    df = pd.read_excel(args.input)
    if "BMH" not in df.columns or "OCR文字结果" not in df.columns:
        print("Error: input xlsx must have 'BMH' and 'OCR文字结果' columns")
        sys.exit(1)
    print(f"Input: {len(df)} rows")

    # Separate valid texts and empty ones
    texts = []
    indices = []         # row indices for valid texts
    skip_entries = []    # (row_index, prediction, status) for skipped rows

    for i, (_, row) in enumerate(df.iterrows()):
        text = row["OCR文字结果"]
        if pd.isna(text) or str(text).strip() == "":
            skip_entries.append((i, None, None, "跳过(空文本)"))
        else:
            texts.append(str(text))
            indices.append(i)

    # Run batched inference
    is_coral_mix = head_type == "coral_mix"
    predictions = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Inference"):
        batch_texts = texts[start:start + args.batch_size]
        try:
            batch_result = predict_batch(
                batch_texts, model, tokenizer, score_points,
                device=args.device, max_length=max_length,
                full_score=full_score,
                calibration=calibration,
            )
            if is_coral_mix:
                coral_scores, reg_scores = batch_result
                for idx in range(len(batch_texts)):
                    predictions.append((indices[start + idx], coral_scores[idx], reg_scores[idx], "成功"))
            else:
                for idx in range(len(batch_texts)):
                    predictions.append((indices[start + idx], batch_result[idx], None, "成功"))
            torch.cuda.empty_cache()
        except Exception as e:
            for idx in range(len(batch_texts)):
                predictions.append((indices[start + idx], None, None, f"失败: {e}"))

    # Merge results back in original order
    all_results = [None] * len(df)
    for row_idx, coral_score, reg_score, status in skip_entries:
        all_results[row_idx] = (coral_score, reg_score, status)
    for row_idx, coral_score, reg_score, status in predictions:
        all_results[row_idx] = (coral_score, reg_score, status)

    # Save output
    if is_coral_mix:
        df["预测分数(coral)"] = [r[0] for r in all_results]
        df["预测分数(reg)"] = [r[1] for r in all_results]
    else:
        df["预测分数"] = [r[0] for r in all_results]
    df["处理状态"] = [r[2] for r in all_results]
    df.to_excel(args.output, index=False)

    success = sum(1 for r in all_results if r[2] == "成功")
    skip = sum(1 for r in all_results if "跳过" in r[2])
    fail = sum(1 for r in all_results if "失败" in r[2])
    print(f"Done. 成功={success} 跳过={skip} 失败={fail}")
    print(f"Saved to: {args.output}")
