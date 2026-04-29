"""
Causal alignment experiment: replace LMF features with English reference values.

For each sample:
  1. Run EN query with collection hook -> capture z_en (SAE activations)
  2. Run non-EN queries with alignment hook -> replace selected features with z_en

Conditions:
  - baseline:    no intervention
  - align_bot10: replace bottom 10% consistency features with EN values
  - align_bot20: replace bottom 20% features
  - align_all:   replace ALL active features (upper bound)
"""
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sae_model import SparseAutoencoder
from config import SHORT_ANSWER_SUFFIX, normalize_answer


class AlignmentHook:
    """Hook that collects EN SAE activations or aligns non-EN activations."""

    def __init__(self, sae, alignment_indices, device):
        self.sae = sae
        self.alignment_indices = alignment_indices
        self.device = device
        self.mode = "off"
        self.collected_z = None
        self.donor_z = None
        self._handle = None

    def hook_fn(self, module, input, output):
        if self.mode == "off":
            return output

        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        seq_len = hidden.shape[1] if hidden.dim() == 3 else hidden.shape[0]
        if seq_len <= 1:
            return output

        if self.mode == "collect":
            with torch.no_grad():
                last_h = (hidden[:, -1, :] if hidden.dim() == 3 else hidden[-1:]).float()
                self.collected_z = self.sae.encode(last_h)
            return output

        elif self.mode == "align" and self.donor_z is not None:
            with torch.no_grad():
                if hidden.dim() == 3:
                    last_h = hidden[:, -1, :].float()
                else:
                    last_h = hidden[-1:].float()
                z = self.sae.encode(last_h)
                z[:, self.alignment_indices] = self.donor_z[:, self.alignment_indices]
                h_new = self.sae.decode(z)
                modified = hidden.clone()
                if hidden.dim() == 3:
                    modified[:, -1, :] = h_new.to(hidden.dtype)
                else:
                    modified[-1, :] = h_new[0].to(hidden.dtype)
            return (modified,) + output[1:] if is_tuple else modified

        return output

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()


def load_sae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(cfg["input_dim"], cfg["hidden_dim"]).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def get_alignment_indices(analysis_path, condition):
    data = torch.load(analysis_path, weights_only=True)
    consistency = data["combined_consistency"]
    active_mask = data["active_mask"]
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    active_consistency = consistency[active_indices]
    sorted_order = active_consistency.argsort()
    n_active = len(active_indices)

    if condition == "align_bot10":
        k = int(n_active * 0.10)
    elif condition == "align_bot20":
        k = int(n_active * 0.20)
    elif condition == "align_all":
        return active_indices
    else:
        return torch.tensor([], dtype=torch.long)

    return active_indices[sorted_order[:k]]


def generate_answer(model, processor, image, question, lang, device):
    prompt = question + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def compute_metrics(results, langs):
    n = len(results)
    if n == 0:
        return {}
    non_en = [l for l in langs if l != "en"]
    per_lang_acc = {}
    for lang in langs:
        correct = sum(1 for r in results if r["correct"].get(lang, False))
        per_lang_acc[lang] = correct / n
    non_en_mean = float(np.mean([per_lang_acc.get(l, 0) for l in non_en]))
    consistency = sum(1 for r in results if r["all_consistent"]) / n
    return {
        "n_samples": n,
        "consistency_rate": consistency,
        "per_lang_accuracy": per_lang_acc,
        "accuracy_gap": float(per_lang_acc.get("en", 0) - non_en_mean),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--analysis_path", type=str, required=True,
                        help="Path to crosslingual_analysis.pt")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True,
                        choices=["baseline", "align_bot10", "align_bot20", "align_all"])
    parser.add_argument("--intervention_layer", type=int, required=True)
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    langs = args.langs.split(",")
    non_en = [l for l in langs if l != "en"]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if args.n_samples:
        dataset = dataset[:args.n_samples]

    if args.condition == "baseline":
        results = []
        for item in tqdm(dataset, desc="baseline"):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception:
                continue
            entry = {"id": item["id"], "answers": {}, "gt": item.get("answer", "")}
            for lang in langs:
                q = item["questions"].get(lang)
                if q:
                    entry["answers"][lang] = generate_answer(
                        model, processor, image, q, lang, device
                    )
            norm = {l: normalize_answer(a) for l, a in entry["answers"].items()}
            entry["normalized"] = norm
            entry["all_consistent"] = len(set(norm.values())) == 1
            gt_norm = normalize_answer(entry["gt"])
            entry["correct"] = {l: (a == gt_norm) for l, a in norm.items()}
            results.append(entry)
    else:
        sae = load_sae(args.sae_path, device)
        indices = get_alignment_indices(args.analysis_path, args.condition).to(device)
        print(f"Aligning {len(indices)} features at layer {args.intervention_layer}")

        hook = AlignmentHook(sae, indices, device)
        layer_module = model.model.language_model.layers[args.intervention_layer]
        hook.register(layer_module)

        results = []
        for item in tqdm(dataset, desc="alignment"):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception:
                continue
            en_q = item["questions"].get("en")
            if not en_q:
                continue

            entry = {"id": item["id"], "answers": {}, "gt": item.get("answer", "")}
            hook.mode = "collect"
            hook.collected_z = None
            entry["answers"]["en"] = generate_answer(
                model, processor, image, en_q, "en", device
            )

            if hook.collected_z is not None:
                hook.mode = "align"
                hook.donor_z = hook.collected_z
                for lang in non_en:
                    q = item["questions"].get(lang)
                    if q:
                        entry["answers"][lang] = generate_answer(
                            model, processor, image, q, lang, device
                        )
                hook.donor_z = None
            hook.mode = "off"

            norm = {l: normalize_answer(a) for l, a in entry["answers"].items()}
            entry["normalized"] = norm
            entry["all_consistent"] = len(set(norm.values())) == 1
            gt_norm = normalize_answer(entry["gt"])
            entry["correct"] = {l: (a == gt_norm) for l, a in norm.items()}
            results.append(entry)

        hook.remove()

    metrics = compute_metrics(results, langs)
    print(f"\n{args.condition}: gap={metrics.get('accuracy_gap', 0):.3f}")

    with open(save_dir / f"results_{args.condition}.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "raw": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
