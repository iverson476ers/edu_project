from __future__ import annotations

import torch
from transformers import AutoTokenizer

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
    pooling = build_pooling(pooling_strategy, hidden_dim)
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

    return model, tokenizer, score_points, config.max_length, full_score, head_type


def predict(
    text: str,
    model: OrdinalScorer,
    tokenizer,
    score_points: list[float],
    device: str = "cuda",
    max_length: int = 2048,
    full_score: float | None = None,
) -> float:
    """Predict score for a single text."""
    encoded = tokenizer(
        text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        if getattr(model, "head_type", "coral") == "regression":
            scores = regression_to_score(logits.cpu(), full_score)
        else:
            probs = torch.sigmoid(logits)
            scores = prediction_to_score(probs.cpu(), score_points)
    return scores[0]


def predict_batch(
    texts: list[str],
    model: OrdinalScorer,
    tokenizer,
    score_points: list[float],
    device: str = "cuda",
    max_length: int = 2048,
    full_score: float | None = None,
) -> list[float]:
    """Predict scores for a batch of texts in one forward pass."""
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
        logits = model(input_ids, attention_mask)
        if getattr(model, "head_type", "coral") == "regression":
            scores = regression_to_score(logits.cpu(), full_score)
        else:
            probs = torch.sigmoid(logits).cpu()
            scores = prediction_to_score(probs, score_points)
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
    model, tokenizer, score_points, max_length, full_score, head_type = load_scorer(
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
            skip_entries.append((i, None, "跳过(空文本)"))
        else:
            texts.append(str(text))
            indices.append(i)

    # Run batched inference
    predictions = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Inference"):
        batch_texts = texts[start:start + args.batch_size]
        try:
            batch_scores = predict_batch(
                batch_texts, model, tokenizer, score_points,
                device=args.device, max_length=max_length,
                full_score=full_score,
            )
            for idx in range(len(batch_texts)):
                predictions.append((indices[start + idx], batch_scores[idx], "成功"))
            torch.cuda.empty_cache()
        except Exception as e:
            for idx in range(len(batch_texts)):
                predictions.append((indices[start + idx], None, f"失败: {e}"))

    # Merge results back in original order
    all_results = [None] * len(df)
    for row_idx, score, status in skip_entries:
        all_results[row_idx] = (score, status)
    for row_idx, score, status in predictions:
        all_results[row_idx] = (score, status)

    # Save output
    df["预测分数"] = [r[0] for r in all_results]
    df["处理状态"] = [r[1] for r in all_results]
    df.to_excel(args.output, index=False)

    success = sum(1 for r in all_results if r[1] == "成功")
    skip = sum(1 for r in all_results if "跳过" in r[1])
    fail = sum(1 for r in all_results if "失败" in r[1])
    print(f"Done. 成功={success} 跳过={skip} 失败={fail}")
    print(f"Saved to: {args.output}")

