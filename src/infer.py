import torch
from transformers import AutoTokenizer

from src.model import OrdinalScorer, prediction_to_score


def load_scorer(checkpoint_path: str, device: str = "cuda"):
    """Load a saved OrdinalScorer checkpoint. Returns (model, tokenizer, score_points)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    score_points = ckpt["score_points"]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Rebuild model structure and load weights
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
    from peft import PeftModel
    # Note: backbone loaded as base; LoRA weights are in the saved state_dict
    hidden_dim = backbone.config.hidden_size
    model = OrdinalScorer(backbone, hidden_dim, len(score_points))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, score_points


def predict(
    text: str, model: OrdinalScorer, tokenizer, score_points: list[float], device: str = "cuda"
) -> float:
    encoded = tokenizer(
        text, max_length=2048, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        scores = prediction_to_score(probs.cpu(), score_points)
    return scores[0]


def predict_batch(
    texts: list[str], model: OrdinalScorer, tokenizer, score_points: list[float], device: str = "cuda"
) -> list[float]:
    results = []
    for text in texts:
        results.append(predict(text, model, tokenizer, score_points, device))
    return results
