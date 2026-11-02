#!/usr/bin/env python3
"""Fit a code-policy success head while preserving the trained VLA router."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from success_router_server import CheckpointPredictor, OUTPUT_CONTRACT, input_record


INPUT_FIELDS = ("task_text", "images", "state_text", "output_contract")


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else payload["instances"]


def scene_key(row):
    instance = row.get("instance", row)
    return str(instance["task"]), str(instance["setting"]), int(instance["seed"])


def normalize_row(row, directory):
    if type(row.get("code_success")) is not bool:
        raise ValueError("Every training row requires a boolean code_success from an executed code policy.")
    source = row.get("input", row)
    if "observation" in source:
        record = {"observation": source["observation"], "instruction": source.get("instruction")}
    else:
        record = {field: source[field] for field in INPUT_FIELDS}
        record["images"] = [str((directory / image).resolve()) for image in record["images"]]
        if len(record["images"]) != 3:
            raise ValueError("Router input requires three initial camera images in head, left, right order.")
        if record["output_contract"] != OUTPUT_CONTRACT:
            raise ValueError("Router output_contract differs from the source RoboTwin training contract.")
    normalized = {
        "instance": dict(zip(("task", "setting", "seed"), scene_key(row))),
        "input": record,
        "code_success": row["code_success"],
    }
    if "vla_selected_model" in row or "vla_success" in row:
        if type(row.get("vla_success")) is not bool or not isinstance(row.get("vla_selected_model"), str):
            raise ValueError("Paired VLA labels require vla_selected_model and boolean vla_success.")
        normalized.update(vla_selected_model=row["vla_selected_model"], vla_success=row["vla_success"])
    if "policy" in row:
        normalized["policy"] = row["policy"]
    return normalized


def prepare_data(train_path, dev_path, test_path):
    test_rows = read_rows(test_path)
    test_keys = {scene_key(row) for row in test_rows}
    datasets, summaries = {}, {}
    for split, path in (("train", train_path), ("dev", dev_path)):
        original = read_rows(path)
        retained = [row for row in original if scene_key(row) not in test_keys]
        records = [normalize_row(row, Path(path).resolve().parent) for row in retained]
        keys = [scene_key(row) for row in records]
        if not records:
            raise ValueError(f"No {split} rows remain after excluding frozen test instances.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate {split} scenes; provide one code-policy outcome per scene.")
        datasets[split] = records
        successes = sum(row["code_success"] for row in records)
        summaries[split] = {
            "source_rows": len(original), "rows": len(records),
            "excluded_test_overlap": len(original) - len(records),
            "successes": successes, "failures": len(records) - successes,
            "paired_vla_rows": sum("vla_success" in row for row in records),
            "settings": dict(Counter(row["instance"]["setting"] for row in records)),
        }
    if {scene_key(row) for row in datasets["train"]} & {scene_key(row) for row in datasets["dev"]}:
        raise ValueError("Train and dev share task/setting/seed scenes.")
    summary = {
        "train": summaries["train"], "dev": summaries["dev"],
        "frozen_test_instances": len(test_rows), "frozen_test_unique_scenes": len(test_keys),
        "test_overlap_after_exclusion": 0,
        "checkpoint_selection": (
            "maximum_paired_dev_routing_success_then_minimum_code_bce"
            if summaries["dev"]["paired_vla_rows"] == summaries["dev"]["rows"] else "minimum_dev_code_bce"
        ),
    }
    return datasets, summary


def model_record(row):
    source = row["input"]
    if "observation" in source:
        return input_record(source)
    return dict(source, id=":".join(map(str, scene_key(row))))


def extract_features(predictor, rows, split):
    torch = predictor.torch
    features, logits = [], []

    def capture(module, arguments):
        features.append(arguments[0].detach().float().cpu())

    handle = predictor.model.head.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                prediction, _ = predictor.training.forward_row(
                    predictor.model, model_record(row), predictor.processor,
                    predictor.args, predictor.device,
                )
                logits.append(prediction.detach().float().cpu())
                if (index + 1) % 25 == 0 or index + 1 == len(rows):
                    print(json.dumps({"event": "features", "split": split,
                                      "rows": index + 1, "total": len(rows)}), flush=True)
    finally:
        handle.remove()
    return {
        "features": torch.cat(features),
        "vla_logits": torch.cat(logits),
        "labels": torch.tensor([row["code_success"] for row in rows], dtype=torch.float32),
    }


def attach_paired_dev(dev, rows, model_ids):
    import torch

    if not all("vla_success" in row for row in rows):
        return
    selected = [model_ids[index] for index in dev["vla_logits"].argmax(dim=1).tolist()]
    for row, model_id in zip(rows, selected):
        if row["vla_selected_model"] != model_id:
            raise ValueError(
                f"VLA baseline candidate differs from checkpoint argmax for {scene_key(row)}: "
                f"ran {row['vla_selected_model']}, predicted {model_id}."
            )
    dev["vla_success"] = torch.tensor([row["vla_success"] for row in rows], dtype=torch.float32)


def dev_metrics(code_logits, dev):
    import torch
    from torch.nn import functional

    metrics = {
        "dev_code_bce": functional.binary_cross_entropy_with_logits(code_logits, dev["labels"]).item(),
    }
    if "vla_success" in dev:
        choose_code = code_logits.sigmoid() > dev["vla_logits"].sigmoid().max(dim=1).values
        successes = torch.where(choose_code, dev["labels"], dev["vla_success"])
        metrics.update(
            dev_routing_success=successes.mean().item(),
            dev_code_selections=int(choose_code.sum().item()),
        )
    return metrics


def selection_key(metrics):
    return metrics.get("dev_routing_success", 0.0), -metrics["dev_code_bce"]


def train_head(train, dev, epochs, batch_size, learning_rate, seed):
    import torch
    from torch import nn
    from torch.nn import functional

    torch.manual_seed(seed)
    code_head = nn.Linear(train["features"].shape[1], 1, dtype=torch.float32)
    nn.init.zeros_(code_head.weight)
    nn.init.zeros_(code_head.bias)
    optimizer = torch.optim.AdamW(code_head.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    best_metrics, best_state, best_epoch = None, None, None
    curve = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train["labels"]), generator=generator)
        total_loss = 0.0
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = code_head(train["features"][indices]).squeeze(-1)
            loss = functional.binary_cross_entropy_with_logits(logits, train["labels"][indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        with torch.no_grad():
            dev_logits = code_head(dev["features"]).squeeze(-1)
            metrics = dev_metrics(dev_logits, dev)
        selected = best_metrics is None or selection_key(metrics) > selection_key(best_metrics)
        if selected:
            best_metrics, best_epoch = metrics, epoch
            best_state = {key: value.detach().clone() for key, value in code_head.state_dict().items()}
        curve.append({"epoch": epoch, "train_bce": total_loss / len(order),
                      **metrics, "selected": selected})
    code_head.load_state_dict(best_state)
    return code_head, {
        "best_epoch": best_epoch, **best_metrics, "curve": curve,
        "checkpoint_selection": ("maximum_paired_dev_routing_success_then_minimum_code_bce"
                                 if "vla_success" in dev else "minimum_dev_code_bce"),
    }


def combined_head(original, code_head):
    import torch

    return {
        "weight": torch.cat((original["weight"].cpu(), code_head.weight.detach().cpu()), dim=0),
        "bias": torch.cat((original["bias"].cpu(), code_head.bias.detach().cpu()), dim=0),
    }


def save_checkpoint(source, output, code_head, code_id, model_path, summary, training, arguments):
    import torch

    source, output = Path(source), Path(output)
    config = json.loads((source / "router_config.json").read_text())
    plan = json.loads((source / "training_plan.json").read_text())
    original = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    combined = combined_head(original, code_head)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("adapter", "processor"):
        shutil.copytree(source / directory, output / directory)
    torch.save(combined, output / "head.pt")
    config["model_ids"] = [*config["model_ids"], code_id]
    config["metadata"] = {
        **config["metadata"], "model_path": model_path,
        "source_checkpoint": str(source.resolve()),
        "source_training": dict(config["metadata"]),
        "code_head": {
            "candidate_id": code_id, "trained": True,
            "supervision": "code_success boolean from actual code-policy episode outcomes",
            "encoder_frozen": True, "original_vla_heads_frozen": True,
            "checkpoint_selection": training["checkpoint_selection"],
            "best_epoch": training["best_epoch"], "dev_code_bce": training["dev_code_bce"],
            "dev_routing_success": training.get("dev_routing_success"),
            "routing_validation": ("paired_initial_choice" if "dev_routing_success" in training
                                   else "code_probability_only_no_paired_vla_outcomes"),
            "data": summary,
        },
    }
    plan["model_ids"] = config["model_ids"]
    plan["code_head_training"] = arguments
    for filename, value in (("router_config.json", config), ("training_plan.json", plan),
                            ("models.json", {"model_ids": config["model_ids"]})):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path,
                        help="Optional external model.py/train.py directory; defaults to bundled router sources")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--test-instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-id", default="code")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets, summary = prepare_data(args.train_jsonl, args.dev_jsonl, args.test_instances)
    config = json.loads((args.checkpoint / "router_config.json").read_text())
    if args.code_id in config["model_ids"]:
        raise ValueError(f"Candidate {args.code_id} already has a trained head.")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive.")
    print(json.dumps({"event": "data", **summary}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    import torch

    started = time.monotonic()
    model_path = args.model_path or config["metadata"]["model_path"]
    identity = {
        "checkpoint": str(args.checkpoint.resolve()), "model_path": model_path,
        "source_metadata": config["metadata"], "datasets": datasets,
    }
    cache_path = args.feature_cache or args.output.with_name(args.output.name + ".features.pt")
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache["identity"] != identity:
            raise ValueError("Feature cache belongs to different inputs or a different source checkpoint.")
        features = cache["features"]
    else:
        predictor = CheckpointPredictor(args.training_root, args.checkpoint, model_path, args.device)
        predictor.model.requires_grad_(False)
        features = {split: extract_features(predictor, rows, split) for split, rows in datasets.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": identity, "features": features}, cache_path)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    attach_paired_dev(features["dev"], datasets["dev"], config["model_ids"])
    code_head, training = train_head(features["train"], features["dev"], args.epochs,
                                     args.batch_size, args.lr, args.seed)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    final_config = save_checkpoint(args.checkpoint, args.output, code_head, args.code_id,
                                   model_path, summary, training, arguments)
    with torch.no_grad():
        probabilities = code_head(features["dev"]["features"]).squeeze(-1).sigmoid().tolist()
    with (args.output / "predictions-dev.jsonl").open("w") as stream:
        for index, (row, probability) in enumerate(zip(datasets["dev"], probabilities)):
            prediction = {"instance": row["instance"], "code_success": row["code_success"],
                          "code_probability": probability}
            if "vla_success" in features["dev"]:
                vla_probability = features["dev"]["vla_logits"][index].sigmoid().max().item()
                choose_code = probability > vla_probability
                prediction.update(vla_selected_model=row["vla_selected_model"], vla_success=row["vla_success"],
                                  vla_probability=vla_probability, selected_model=args.code_id if choose_code else row["vla_selected_model"],
                                  selected_success=row["code_success"] if choose_code else row["vla_success"])
            stream.write(json.dumps(prediction) + "\n")
    result = {
        "status": "completed", "checkpoint": str(args.output.resolve()),
        "trained_candidates": final_config["model_ids"], "data": summary,
        "training": training, "feature_cache": str(cache_path.resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "training"}), flush=True)


if __name__ == "__main__":
    main()
