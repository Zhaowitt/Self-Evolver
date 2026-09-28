#!/usr/bin/env python3
"""Shallow LoRA SFT for the Controller, token-identical to the EasyR1/GRPO prompt path.

Loss covers only the assistant turn as the chat template renders it; after training the
held-out split is greedily decoded and gated, then the adapter is merged for ``CONTROLLER_MODEL_PATH``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.controller.parser import parse_controller_response  # noqa: E402
from src.controller.schema import MAX_SELECTED_SKILLS  # noqa: E402

IGNORE_INDEX = -100


def load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = {
        "max_sequence_length": 13312,
        "epochs": 2,
        "learning_rate": 1.0e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "seed": 1,
        "max_new_tokens": 256,
        "min_json_valid_rate": 0.95,
        "min_legal_selection_rate": 1.0,
    }
    return {**defaults, **data}


def assistant_suffix(tokenizer, prompt: str, response: str) -> str:
    """Text the chat template appends for the assistant turn, up to end-of-turn."""
    user = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(
        [*user, {"role": "assistant", "content": response}],
        add_generation_prompt=False,
        tokenize=False,
    )
    eos = tokenizer.eos_token or ""
    if full_text.startswith(prompt_text) and response in full_text[len(prompt_text):]:
        suffix = full_text[len(prompt_text):]
        if eos and eos in suffix:
            suffix = suffix[: suffix.index(eos) + len(eos)]
        return suffix
    # Template renders history differently from the generation prompt: use the raw completion.
    return response + eos


def encode_example(tokenizer, row: Dict[str, Any], max_length: int) -> Dict[str, List[int]] | None:
    user = [{"role": "user", "content": row["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(
        assistant_suffix(tokenizer, row["prompt"], row["response"]),
        add_special_tokens=False,
    )["input_ids"]
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        # Never truncate: a clipped prompt or target teaches the wrong mapping.
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + response_ids,
    }


def score_prediction(text: str, row: Dict[str, Any]) -> Dict[str, bool]:
    signal = parse_controller_response(text)
    target = json.loads(row["response"])["selected_skill_ids"]
    candidates = set(row["candidate_skill_ids"])
    selected = signal.selected_skill_ids
    json_valid = not signal.parse_error
    legal = json_valid and len(selected) <= MAX_SELECTED_SKILLS and set(selected) <= candidates
    return {
        "json_valid": json_valid,
        "legal_selection": legal,
        "exact_match": legal and sorted(selected) == sorted(target),
        "any_overlap": legal and (bool(set(selected) & set(target)) or (not target and not selected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "controller_sft.yaml")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override config model_path.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    cfg = load_config(args.config)
    model_path = args.model or cfg["model_path"]
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_rows = _jsonl(args.train)
    val_rows = _jsonl(args.val)
    if not train_rows or not val_rows:
        raise SystemExit("SFT needs non-empty --train and --val files")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_all(rows):
        encoded, dropped = [], []
        for row in rows:
            item = encode_example(tokenizer, row, cfg["max_sequence_length"])
            (encoded.append(item) if item else dropped.append(row["instance_id"]))
        return encoded, dropped

    train_data, train_dropped = encode_all(train_rows)
    val_data, val_dropped = encode_all(val_rows)
    if train_dropped or val_dropped:
        print(f"Dropped overlong rows: train={train_dropped} val={val_dropped}", file=sys.stderr)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(cfg["lora_rank"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=cfg["lora_target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(
        len(train_data) / (cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"])
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type="cosine",
        logging_steps=max(1, steps_per_epoch // 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=dtype == torch.bfloat16,
        report_to="none",
        seed=int(cfg["seed"]),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
    )
    trainer.train()

    # Greedy generation-based acceptance on the held-out split.
    model = trainer.model
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    scores = []
    samples = []
    for row in val_rows:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(cfg["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(score_prediction(text, row))
        if len(samples) < 5:
            samples.append({"instance_id": row["instance_id"], "target": row["response"], "output": text})

    def rate(key: str) -> float:
        return round(sum(item[key] for item in scores) / len(scores), 4)

    metrics = {
        "model_path": model_path,
        "train_rows": len(train_data),
        "val_rows": len(val_rows),
        "dropped_overlong": {"train": train_dropped, "val": val_dropped},
        "json_valid_rate": rate("json_valid"),
        "legal_selection_rate": rate("legal_selection"),
        "exact_match_rate": rate("exact_match"),
        "overlap_rate": rate("any_overlap"),
        "best_eval_loss": trainer.state.best_metric,
        "config": cfg,
        "samples": samples,
    }
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        metrics["skill_bank_hash"] = manifest.get("skill_bank_hash")
    (args.output_dir / "sft_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: metrics[k] for k in metrics if k.endswith("_rate")}, indent=2))

    model.save_pretrained(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    failed = []
    if metrics["json_valid_rate"] < float(cfg["min_json_valid_rate"]):
        failed.append(f"json_valid_rate {metrics['json_valid_rate']} < {cfg['min_json_valid_rate']}")
    if metrics["legal_selection_rate"] < float(cfg["min_legal_selection_rate"]):
        failed.append(
            f"legal_selection_rate {metrics['legal_selection_rate']} < {cfg['min_legal_selection_rate']}"
        )
    if failed:
        raise SystemExit("SFT acceptance gate failed: " + "; ".join(failed))
    # Only a model that passed the gate is merged, so stage 4 never reuses a rejected one.
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(args.output_dir / "merged"), safe_serialization=True)
        tokenizer.save_pretrained(str(args.output_dir / "merged"))
        print(f"Merged Controller for GRPO: CONTROLLER_MODEL_PATH={args.output_dir / 'merged'}")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
