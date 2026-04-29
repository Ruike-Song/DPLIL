"""
Collect last-token hidden states from multiple decoder layers in a single forward pass.

The last token is the decision point for answer generation in decoder-only VLMs.
We hook multiple layers simultaneously and extract the last-token hidden vector
per (image, question_lang) pair. This produces training data for the SAE and
material for cross-lingual divergence analysis.

Supports multi-GPU sharding and post-hoc merge.
"""
import torch
import json
import time
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class MultiLayerHook:
    """Hook multiple layers simultaneously; extract last token from each."""

    def __init__(self):
        self.states = {}
        self._handles = []

    def register(self, model, layer_indices, layer_accessor):
        for idx in layer_indices:
            layer = layer_accessor(model, idx)

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    hs = output[0].detach()
                    if hs.dim() == 3:
                        hs = hs.squeeze(0)
                    self.states[layer_idx] = hs[-1].float().cpu()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(idx))
            self._handles.append(handle)
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_last_token_states(self):
        return dict(self.states)

    def clear(self):
        self.states.clear()


def build_input(processor, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], images=[image], return_tensors="pt")


def get_layer_accessor(model_type):
    """Return a function that accesses decoder layer by index for the given model type."""
    accessors = {
        "qwen": lambda model, idx: model.model.language_model.layers[idx],
        "llava": lambda model, idx: model.model.language_model.layers[idx],
        "internvl": lambda model, idx: model.language_model.model.layers[idx],
    }
    if model_type not in accessors:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(accessors)}")
    return accessors[model_type]


def collect(args):
    device = f"cuda:{args.gpu_id}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, processor = load_model(args.model_path, args.model_type, device)

    print("Loading dataset...")
    with open(args.data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.n_samples and args.n_samples < len(dataset):
        dataset = dataset[:args.n_samples]

    if args.total_gpus > 1:
        chunk = (len(dataset) + args.total_gpus - 1) // args.total_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, len(dataset))
        dataset = dataset[start:end]
        print(f"[GPU {args.gpu_id}] Shard: samples {start}-{end}")

    layers = [int(x) for x in args.layers.split(",")]
    langs = [x.strip() for x in args.langs.split(",")]
    print(f"Layers: {layers}, Langs: {langs}, Samples: {len(dataset)}")

    layer_accessor = get_layer_accessor(args.model_type)
    hook = MultiLayerHook().register(model, layers, layer_accessor)

    per_layer_lang = {l: {lang: [] for lang in langs} for l in layers}
    metadata = []
    n_skipped = 0
    t_start = time.time()

    for item in tqdm(dataset, desc=f"GPU{args.gpu_id}"):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            n_skipped += 1
            continue

        valid = True
        item_states = {lang: {} for lang in langs}
        for lang in langs:
            q = item["questions"].get(lang)
            if not q:
                valid = False
                break
            inputs = build_input(processor, image, q)
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
            hook.clear()
            with torch.no_grad():
                model(**inputs_gpu)
            last_states = hook.get_last_token_states()
            if len(last_states) != len(layers):
                valid = False
                break
            item_states[lang] = last_states

        if not valid:
            n_skipped += 1
            continue

        for lang in langs:
            for l in layers:
                per_layer_lang[l][lang].append(item_states[lang][l])
        metadata.append({"id": item["id"], "image_path": item["image_path"]})

    hook.remove()
    elapsed = time.time() - t_start
    print(f"Collection done: {len(metadata)} samples, {n_skipped} skipped, {elapsed:.0f}s")

    suffix = f"_gpu{args.gpu_id}" if args.total_gpus > 1 else ""
    for l in layers:
        for lang in langs:
            if per_layer_lang[l][lang]:
                t = torch.stack(per_layer_lang[l][lang])
                torch.save(t, save_dir / f"layer{l}_{lang}{suffix}.pt")

    all_vectors = []
    for l in layers:
        for lang in langs:
            if per_layer_lang[l][lang]:
                all_vectors.append(torch.stack(per_layer_lang[l][lang]))
    if all_vectors:
        combined = torch.cat(all_vectors, dim=0)
        torch.save(combined, save_dir / f"all_combined{suffix}.pt")
        print(f"Saved combined: {combined.shape}")

    meta_info = {
        "layers": layers, "languages": langs,
        "n_samples": len(metadata), "n_skipped": n_skipped,
        "elapsed_seconds": elapsed, "metadata": metadata,
    }
    with open(save_dir / f"metadata{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)


def merge(args):
    save_dir = Path(args.save_dir)
    layers = [int(x) for x in args.layers.split(",")]
    langs = [x.strip() for x in args.langs.split(",")]

    print(f"Merging {args.total_gpus} shards...")
    for l in layers:
        for lang in langs:
            shards = []
            for gpu_id in range(args.total_gpus):
                fpath = save_dir / f"layer{l}_{lang}_gpu{gpu_id}.pt"
                if fpath.exists():
                    shards.append(torch.load(fpath, weights_only=True))
            if shards:
                merged = torch.cat(shards, dim=0)
                torch.save(merged, save_dir / f"layer{l}_{lang}.pt")
                print(f"  layer{l}/{lang} merged: {merged.shape}")

    print("Merge done.")


def load_model(model_path, model_type, device):
    """Load VLM and processor. Adapt import based on model_type."""
    if model_type in ("qwen", "qwen8b"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_type == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_type == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True
        )
        model.eval()
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["qwen", "qwen8b", "llava", "internvl"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layer indices to hook")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    if args.merge:
        merge(args)
    else:
        collect(args)
