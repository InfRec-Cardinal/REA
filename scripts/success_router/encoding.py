"""Multimodal input encoding shared by success-router inference and head training."""
from contextlib import nullcontext
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "Read the robot task, its initial camera observations, current robot state, "
    "and required action output. Represent the operations needed to complete the task."
)


def load_processor(args):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path, local_files_only=True)
    template = (Path(args.chat_template_path).read_text() if args.chat_template_path else
                processor.chat_template or processor.tokenizer.chat_template)
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return processor


def encode_record(row, processor, args):
    """The identical mode-independent multimodal input used by both routers."""
    from PIL import Image

    images = []
    for path in row["images"]:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    state = row.get("state_text") or ""
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    task_ids = processor.tokenizer.encode(row["task_text"], add_special_tokens=False)
    retained = len(task_ids)
    task_text = row["task_text"]
    contract = json.dumps(row["output_contract"], ensure_ascii=False, separators=(",", ":"))
    while True:
        content = []
        for index in range(len(images)):
            content.extend([{"type": "text", "text": f"Initial camera {index + 1}: "},
                            {"type": "image"}, {"type": "text", "text": "\n"}])
        content.append({"type": "text", "text":
                        f"Robot task:\n{task_text}\n\nInitial robot state:\n{state}\n\nRequired action output:\n{contract}"})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        options = dict(text=prefix, return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=True)
        if images:
            limit = min(args.max_image_tokens, args.image_token_budget // len(images))
            factor = processor.image_processor.patch_size * processor.image_processor.merge_size
            options.update(images=images, images_kwargs={"size": {
                "shortest_edge": min(64, limit) * factor**2, "longest_edge": limit * factor**2,
            }})
        encoded = processor(**options)
        tokens = int(encoded["input_ids"].shape[1])
        if tokens <= args.seq_length:
            break
        remove = tokens - args.seq_length + 16
        retained = max(0, retained - remove)
        if not retained:
            raise ValueError(f"Initial state and images exceed sequence length for {row['id']}.")
        left = (retained + 1) // 2
        right = retained // 2
        task_text = (processor.tokenizer.decode(task_ids[:left], skip_special_tokens=False)
                     + "\n[Task text cropped]\n"
                     + (processor.tokenizer.decode(task_ids[-right:], skip_special_tokens=False) if right else ""))
    features = {key: encoded[key] for key in (
        "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw") if key in encoded}
    return features, {
        "input_tokens": tokens, "images": len(images), "cropped_task_tokens": len(task_ids) - retained,
        "image_tokens": sum(int(grid.prod()) // processor.image_processor.merge_size**2
                            for grid in encoded.get("image_grid_thw", [])),
    }


def forward_row(model, row, processor, args, device):
    import torch

    features, info = encode_record(row, processor, args)
    inputs = {key: value.to(device) for key, value in features.items()}
    cast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with cast:
        logits = model(**inputs)
    return logits, info
