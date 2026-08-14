from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _adapter_kind(spec: Mapping[str, Any]) -> str:
    return str(spec.get("adapter", "pipeline"))


def load_hf_adapter(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Load one Hugging Face VLM using its supported inference interface."""
    kind = _adapter_kind(spec)
    if kind == "pipeline":
        return _load_pipeline_adapter(spec)
    if kind == "internvl_chat":
        return _load_internvl_adapter(spec)
    if kind == "minicpm_chat":
        return _load_minicpm_adapter(spec)
    if kind == "llava_processor":
        return _load_llava_adapter(spec)
    raise ValueError(f"Unsupported HF adapter: {kind}")


def infer_hf(adapter: Mapping[str, Any], image_paths: Sequence[Path], prompt: str) -> str:
    kind = str(adapter["kind"])
    if kind == "pipeline":
        return _infer_pipeline(adapter, image_paths, prompt)
    if kind == "qwen35_processor":
        return _infer_qwen35(adapter, image_paths, prompt)
    if kind == "internvl_chat":
        return _infer_internvl(adapter, image_paths, prompt)
    if kind == "minicpm_chat":
        return _infer_minicpm(adapter, image_paths, prompt)
    if kind == "llava_processor":
        return _infer_llava(adapter, image_paths, prompt)
    raise ValueError(f"Unsupported loaded HF adapter: {kind}")


def _make_chat(image_paths: Sequence[Path], prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "url": str(Path(path))} for path in image_paths]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_generated_text(out: Any) -> str:
    item = out[0] if isinstance(out, list) else out
    if isinstance(item, list):
        item = item[0]
    value = item.get("generated_text", "") if isinstance(item, dict) else item
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for entry in reversed(value):
            if not isinstance(entry, dict) or entry.get("role") != "assistant":
                continue
            content = entry.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [
                    str(part["text"])
                    for part in content
                    if isinstance(part, dict) and part.get("text")
                ]
                if texts:
                    return "\n".join(texts).strip()

        texts: list[str] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(str(part["text"]))
        return "\n".join(texts).strip()
    return str(value).strip()


def _load_pipeline_adapter(spec: Mapping[str, Any]) -> dict[str, Any]:
    model_id = str(spec["model_id"])
    if model_id.startswith("Qwen/Qwen3.5-"):
        return _load_qwen35_adapter(spec)

    from transformers import pipeline

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        device_map="auto",
        dtype="auto",
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
    )
    return {"kind": "pipeline", "runner": pipe, "model_id": model_id}


def _infer_pipeline(adapter: Mapping[str, Any], image_paths: Sequence[Path], prompt: str) -> str:
    out = adapter["runner"](
        text=_make_chat(image_paths, prompt),
        max_new_tokens=80,
        do_sample=False,
    )
    return _extract_generated_text(out)


def _load_qwen35_adapter(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 direct inference requires CUDA for this benchmark.")

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(spec["model_id"])
    model = AutoModelForMultimodalLM.from_pretrained(
        spec["model_id"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    return {
        "kind": "qwen35_processor",
        "model": model,
        "processor": processor,
        "dtype": dtype,
    }


def _move_batch_to_model(inputs: Mapping[str, Any], model: Any, dtype: Any) -> dict[str, Any]:
    device = model.device
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        is_float = getattr(value, "is_floating_point", lambda: False)()
        moved[key] = value.to(device=device, dtype=dtype) if is_float else value.to(device=device)
    return moved


def _infer_qwen35(adapter: Mapping[str, Any], image_paths: Sequence[Path], prompt: str) -> str:
    if not image_paths:
        raise ValueError("Qwen3.5 inference requires at least one image.")

    processor = adapter["processor"]
    model = adapter["model"]
    messages = _make_chat(image_paths, prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    moved = _move_batch_to_model(inputs, model, adapter["dtype"])
    input_len = moved["input_ids"].shape[-1]
    outputs = model.generate(
        **moved,
        max_new_tokens=80,
        do_sample=False,
    )
    generated = outputs[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _ensure_transformers5_custom_model_compat(model_base_cls: Any | None = None) -> None:
    """Provide the v5 tied-weight default missing from older remote-code VLMs.

    Transformers 5.x accesses ``all_tied_weights_keys`` while loading weights.
    Some 4.x-era custom model classes (including InternVLChatModel/MiniCPMV)
    never ran the newer ``post_init()`` path, so the instance has no attribute.
    A class-level empty dict is the minimal inference-only compatibility default;
    native models that initialize their own value still override it.
    """
    if model_base_cls is None:
        from transformers.modeling_utils import PreTrainedModel

        model_base_cls = PreTrainedModel

    existing = model_base_cls.__dict__.get("all_tied_weights_keys")
    if not isinstance(existing, dict):
        model_base_cls.all_tied_weights_keys = {}


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Any,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    tiles_per_row = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_internvl_image(image_file: Path, input_size: int = 448, max_num: int = 6) -> Any:
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    image = Image.open(image_file).convert("RGB")
    images = _dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    return torch.stack([transform(tile) for tile in images])


def _load_internvl_adapter(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("InternVL direct inference requires CUDA for this benchmark.")

    _ensure_transformers5_custom_model_compat()
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(
        spec["model_id"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"], trust_remote_code=True, use_fast=False
    )
    return {
        "kind": "internvl_chat",
        "model": model,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "max_total_tiles": int(spec.get("max_total_tiles", 6)),
    }


def _internvl_per_image_tile_budget(max_total_tiles: int, image_count: int) -> int:
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    return max(1, max_total_tiles // image_count)


def _infer_internvl(adapter: Mapping[str, Any], image_paths: Sequence[Path], prompt: str) -> str:
    import torch

    if not image_paths:
        raise ValueError("InternVL inference requires at least one image.")

    max_total_tiles = int(adapter["max_total_tiles"])
    per_image_max = _internvl_per_image_tile_budget(max_total_tiles, len(image_paths))
    pixel_batches = [
        _load_internvl_image(Path(path), max_num=per_image_max)
        for path in image_paths
    ]
    num_patches_list = [batch.size(0) for batch in pixel_batches]
    pixel_values = torch.cat(pixel_batches, dim=0).to(adapter["dtype"]).cuda()

    if len(image_paths) == 1:
        image_prefix = "<image>\n"
        num_patches_arg = None
    else:
        image_prefix = "".join(
            f"Image-{index + 1}: <image>\n" for index in range(len(image_paths))
        )
        num_patches_arg = num_patches_list

    generation_config = {"max_new_tokens": 80, "do_sample": False}
    kwargs: dict[str, Any] = {}
    if num_patches_arg is not None:
        kwargs["num_patches_list"] = num_patches_arg

    response = adapter["model"].chat(
        adapter["tokenizer"],
        pixel_values,
        image_prefix + prompt,
        generation_config,
        **kwargs,
    )
    return str(response).strip()


def _load_minicpm_adapter(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _ensure_transformers5_custom_model_compat()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        spec["model_id"],
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], trust_remote_code=True)
    return {"kind": "minicpm_chat", "model": model, "tokenizer": tokenizer}


def _infer_minicpm(adapter: Mapping[str, Any], image_paths: Sequence[Path], prompt: str) -> str:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    msgs = [{"role": "user", "content": [*images, prompt]}]
    answer = adapter["model"].chat(
        msgs=msgs,
        tokenizer=adapter["tokenizer"],
        enable_thinking=False,
        stream=False,
    )
    if isinstance(answer, str):
        return answer.strip()
    if hasattr(answer, "__iter__"):
        return "".join(str(part) for part in answer).strip()
    return str(answer).strip()


def _load_llava_adapter(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("LLaVA-OneVision direct inference requires CUDA for this benchmark.")

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(spec["model_id"])
    model = AutoModelForMultimodalLM.from_pretrained(
        spec["model_id"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    return {
        "kind": "llava_processor",
        "model": model,
        "processor": processor,
        "dtype": dtype,
    }


def _infer_llava(adapter: Mapping[str, Any], image_paths: Sequence[Path], prompt: str) -> str:
    from PIL import Image

    if not image_paths:
        raise ValueError("LLaVA inference requires at least one image.")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    conversation = [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in images),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    processor = adapter["processor"]
    model = adapter["model"]
    formatted_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    inputs = processor(
        images=images,
        text=formatted_prompt,
        return_tensors="pt",
    )
    moved = _move_batch_to_model(inputs, model, adapter["dtype"])

    input_len = moved["input_ids"].shape[-1]
    outputs = model.generate(
        **moved,
        max_new_tokens=80,
        do_sample=False,
    )
    generated = outputs[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()
