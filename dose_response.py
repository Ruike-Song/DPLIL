"""
Dose-response sweep: causal alignment with varying feature percentages.

Uses phi-based feature ranking (instance-conditioned variance) to select
LMF/LIVF/Random feature sets at different quantiles, then measures the
gap reduction when aligning each set to English reference values.

Establishes the causal specificity gradient: LMF > Random > LIVF.
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


class DoseResponseHook:
    """Hook that collects EN SAE activations or aligns non-EN activations."""

    def __init__(self, sae, alignment_indices, device):
        self.sae = sae
        self.alignment_indices = alignment_indices
        self.device = device
        self.mode = "off"
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

        last_hidden = hidden[:, -1, :] if hidden.dim() == 3 else hidden[-1:]

        with torch.no_grad():
            z = self.sae.encode(last_hidden.float())

            if self.mode == "collect":
                self.donor_z = z.clone()
                return output

            if self.mode == "align" and self.donor_z is not None:
                z_mod = z.clone()
                z_mod[:, self.alignment_indices] = self.donor_z[:, self.alignment_indices]
                delta = self.sae.decode(z_mod) - self.sae.decode(z)
                hidden_new = hidden.clone()
                if hidden.dim() == 3:
                    hidden_new[:, -1, :] = hidden[:, -1, :] + delta.to(hidden.dtype)
                else:
                    hidden_new[-1:] = hidden[-1:] + delta.to(hidden.dtype)
                return (hidden_new,) + output[1:] if is_tuple else hidden_new

        return output

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--phi_dir", type=str, required=True,
                        help="Directory with phi_layer*.npy and feature_ranks_layer*.json")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layer indices for intervention")
    parser.add_argument("--q_values", type=str, default="5,10,20,40",
                        help="Percentile values for feature selection")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    langs = args.langs.split(",")
    non_en = [l for l in langs if l != "en"]
    layers = [int(x) for x in args.layers.split(",")]
    q_values = [int(x) for x in args.q_values.split(",")]

    dataset = json.load(open(args.data_path, encoding="utf-8"))
    if args.num_samples:
        dataset = dataset[:args.num_samples]

    print("Loading model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    ckpt = torch.load(args.sae_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(cfg["input_dim"], cfg["hidden_dim"]).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phi_dir = Path(args.phi_dir)

    for layer in layers:
        print(f"\nLayer {layer}")
        ranks = json.load(open(phi_dir / f"feature_ranks_layer{layer}.json"))
        phi = np.load(phi_dir / f"phi_layer{layer}.npy")
        M = len(phi)

        for q in q_values:
            lmvf_idx = ranks[f"LMVF_top{q}"]
            livf_idx = ranks[f"LIVF_bot{q}"]
            np.random.seed(42)
            rand_idx = np.random.choice(M, size=len(lmvf_idx), replace=False).tolist()

            for label, indices in [
                (f"LMVF_q{q}", lmvf_idx),
                (f"LIVF_q{q}", livf_idx),
                (f"Random_q{q}", rand_idx),
            ]:
                print(f"  {label} (n={len(indices)})")
                hook = DoseResponseHook(sae, indices, device)
                hook.register(model.model.language_model.layers[layer])

                results = []
                for sample in tqdm(dataset, desc=label):
                    image = Image.open(sample["image_path"]).convert("RGB")
                    gt = normalize_answer(sample["answer"])
                    entry = {"id": sample["id"], "correct": {}}

                    en_q = sample["questions"]["en"]
                    prompt_en = en_q + SHORT_ANSWER_SUFFIX["en"]
                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_en},
                    ]}]
                    text_en = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inputs_en = processor(text=[text_en], images=[image], return_tensors="pt").to(device)

                    hook.mode = "collect"
                    with torch.no_grad():
                        out_ids = model.generate(**inputs_en, max_new_tokens=32, do_sample=False)
                    input_len = inputs_en["input_ids"].shape[1]
                    en_norm = normalize_answer(
                        processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
                    )
                    entry["correct"]["en"] = en_norm == gt

                    for lang in non_en:
                        q_lang = sample["questions"].get(lang, "")
                        if not q_lang:
                            continue
                        prompt_l = q_lang + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
                        msgs_l = [{"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_l},
                        ]}]
                        text_l = processor.apply_chat_template(msgs_l, tokenize=False, add_generation_prompt=True)
                        inputs_l = processor(text=[text_l], images=[image], return_tensors="pt").to(device)

                        hook.mode = "align"
                        with torch.no_grad():
                            out_ids = model.generate(**inputs_l, max_new_tokens=32, do_sample=False)
                        input_len = inputs_l["input_ids"].shape[1]
                        ans_norm = normalize_answer(
                            processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
                        )
                        entry["correct"][lang] = ans_norm == gt
                    results.append(entry)

                hook.mode = "off"
                hook.remove()

                n = len(results)
                accs = {l: sum(1 for r in results if r["correct"].get(l, False)) / n for l in langs}
                non_en_mean = np.mean([accs[l] for l in non_en])
                gap = accs["en"] - non_en_mean
                print(f"    gap={gap:.3f}")

                with open(out_dir / f"{label}_L{layer}_summary.json", "w") as f:
                    json.dump({"condition": label, "layer": layer, "gap": float(gap),
                               "per_lang_acc": accs}, f, indent=2)


if __name__ == "__main__":
    main()
