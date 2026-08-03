import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.config import TrainingConfig
from src.data_pipeline import load_and_split_data, ASAGDataset, oversample_rare_scores
from src.evaluate import compute_metrics
from src.model import OrdinalScorer, coral_loss, coral_mix_loss, prediction_to_score, regression_loss, regression_to_score


def setup_model_and_tokenizer(
    config: TrainingConfig, num_classes: int
):
    """Load Qwen 4B with 4-bit QLoRA + CoralHead. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    backbone.config.use_cache = False  # disable KV cache for training

    hidden_dim = backbone.config.hidden_size
    # Old: model = OrdinalScorer(backbone, hidden_dim, num_classes)
    # New:
    from src.pooling import build_pooling
    pooling = build_pooling(config.pooling, hidden_dim, num_layers=config.pooling_num_layers)
    head_config = {
        "hidden_sizes": config.head_hidden_sizes,
        "dropout": config.head_dropout,
    }
    model = OrdinalScorer(backbone, hidden_dim, num_classes,
                          pooling=pooling, head_config=head_config,
                          head_type=getattr(config, "head_type", "coral"))

    # Enable gradient checkpointing
    backbone.gradient_checkpointing_enable()

    return model, tokenizer


def get_dataloader(df, tokenizer, config, shuffle=True):
    ds = ASAGDataset(df, tokenizer, config.max_length)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)


def validate(model, dataloader, score_points, device, full_score, tolerance=0.0, beta=1.0, lambda_reg=0.4, return_predictions=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0
    head_type = getattr(model, "head_type", "coral")
    is_regression = head_type == "regression"
    is_coral_mix = head_type == "coral_mix"
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_indices = batch["label_idx"].to(device)
            labels_batch = batch["label"].to(device)
            output = model(input_ids, attention_mask)

            if is_coral_mix:
                ordinal_logits, reg_value = output
                loss = coral_mix_loss(ordinal_logits, reg_value, label_indices,
                                      labels_batch, model.num_classes, full_score,
                                      lambda_reg)
                probs = torch.sigmoid(ordinal_logits)
                preds = prediction_to_score(probs.cpu(), score_points)
            elif is_regression:
                loss = regression_loss(output, labels_batch, full_score, beta)
                preds = regression_to_score(output.cpu(), full_score)
                # Snap to 0.5 grid
                preds = [round(p / 0.5) * 0.5 for p in preds]
            else:
                loss = coral_loss(output, label_indices, model.num_classes)
                probs = torch.sigmoid(output)
                preds = prediction_to_score(probs.cpu(), score_points)

            total_loss += loss.item() * input_ids.size(0)
            num_samples += input_ids.size(0)
            all_preds.extend(preds)
            all_labels.extend(labels_batch.tolist())
    metrics = compute_metrics(all_preds, all_labels, tolerance=tolerance if tolerance > 0 else None)
    metrics["val_loss"] = total_loss / num_samples

    # Per-score-point error monitoring
    from collections import defaultdict
    per_score = defaultdict(list)
    for p, l in zip(all_preds, all_labels):
        per_score[l].append(abs(p - l))
    per_score_mae = {}
    for score in sorted(per_score.keys()):
        vals = per_score[score]
        mae = sum(vals) / len(vals)
        per_score_mae[str(score)] = round(mae, 3)
        print(f"  score {score:>5.1f} (n={len(vals):>4}): MAE={mae:.3f}")
    metrics["per_score_mae"] = per_score_mae

    if return_predictions:
        return metrics, all_preds, all_labels
    return metrics


def train(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    # 1. Load data
    train_df, dev_df, test_df, score_points, q_config = load_and_split_data(config)
    num_classes = len(score_points)
    full_score = float(q_config["full_score"])

    # Oversample rare score points with perturbation (train only)
    if config.oversample_min_count > 0:
        train_df = oversample_rare_scores(
            train_df,
            min_count=config.oversample_min_count,
            perturb_prob=config.oversample_perturb_prob,
            seed=config.seed,
        )
        print(f"Train after oversampling: {len(train_df)}")

    # 2. Setup model
    model, tokenizer = setup_model_and_tokenizer(config, num_classes)
    model.to(device)

    # 3. Build dataloaders
    train_loader = get_dataloader(train_df, tokenizer, config, shuffle=True)
    dev_loader = get_dataloader(dev_df, tokenizer, config, shuffle=False) if dev_df is not None else None
    test_loader = get_dataloader(test_df, tokenizer, config, shuffle=False)

    # 4. Optimizer & scheduler (split LR: head > backbone)
    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head.") or name.startswith("pooling."):
            head_params.append(param)
        else:
            lora_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": config.learning_rate},
        {"params": head_params, "lr": config.learning_rate * config.head_lr_mult},
    ], weight_decay=config.weight_decay)
    print(f"Optimizer: lora_params={len(lora_params)}, head_params={len(head_params)}, "
          f"lora_lr={config.learning_rate}, head_lr={config.learning_rate * config.head_lr_mult}")
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. Training loop
    os.makedirs(config.output_dir, exist_ok=True)
    global_step = 0
    best_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_indices = batch["label_idx"].to(device)

            output = model(input_ids, attention_mask)
            head_type = getattr(model, "head_type", "coral")
            if head_type == "coral_mix":
                ordinal_logits, reg_value = output
                loss = coral_mix_loss(ordinal_logits, reg_value, label_indices,
                                      batch["label"].to(device), num_classes,
                                      full_score, config.lambda_reg)
            elif head_type == "regression":
                loss = regression_loss(output, batch["label"].to(device), full_score, config.beta)
            else:
                loss = coral_loss(output, label_indices, num_classes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % config.logging_steps == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

            if global_step % config.eval_steps == 0:
                metrics = validate(model, dev_loader or test_loader, score_points, device, full_score, config.tolerance, config.beta, config.lambda_reg)
                print(f"Eval @ step {global_step}: {metrics}")
                if metrics["acc"] > best_acc:
                    best_acc = metrics["acc"]
                    torch.save(
                        {"model_state_dict": model.state_dict(), "config": config, "score_points": score_points, "full_score": full_score},
                        os.path.join(config.output_dir, "best_model.pt"),
                    )
                    print(f"New best model saved! acc={best_acc:.4f}")
                model.train()

        # End of epoch eval
        metrics = validate(model, dev_loader or test_loader, score_points, device, full_score, config.tolerance, config.beta, config.lambda_reg)
        print(f"Epoch {epoch+1}/{config.epochs}: {metrics}")
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "score_points": score_points, "full_score": full_score},
                os.path.join(config.output_dir, "best_model.pt"),
            )

        epoch_avg_loss = epoch_loss / len(train_loader)
        val_loss = metrics.get("val_loss", float("nan"))
        print(f"Epoch {epoch+1} train_loss={epoch_avg_loss:.4f}, val_loss={val_loss:.4f}")

    print(f"Training complete. Best accuracy: {best_acc:.4f}")

    # Save final model checkpoint (always)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "score_points": score_points, "full_score": full_score},
        os.path.join(config.output_dir, "final_model.pt"),
    )
    print(f"Final model saved to {os.path.join(config.output_dir, 'final_model.pt')}")

    # Run final validation on test set
    final_metrics = validate(model, test_loader, score_points, device, full_score, config.tolerance, config.beta, config.lambda_reg)
    final_metrics["best_accuracy"] = best_acc
    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Final metrics: {final_metrics}")
    print(f"Metrics saved to {os.path.join(config.output_dir, 'metrics.json')}")

    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    from src.config import load_config
    cfg = load_config(args.config)
    train(cfg)
