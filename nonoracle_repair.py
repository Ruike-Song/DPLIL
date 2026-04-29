"""
Non-oracle repair methods:
  1. Mean-shift correction — learns a per-language correction vector from
     training data, applies it at test time without English oracle states
  2. INLP projection — iteratively removes language-predictive directions

Mean-shift is the primary method; INLP serves as comparison baseline.
"""
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from config import SHORT_ANSWER_SUFFIX, normalize_answer


def compute_mean_shift_vectors(states_dir, layer_idx, langs, train_ratio=0.8):
    """Compute delta_l = mean(h_en) - mean(h_l) for each non-EN language."""
    states = {}
    for lang in langs:
        fpath = Path(states_dir) / f"layer{layer_idx}_{lang}.pt"
        states[lang] = torch.load(fpath, map_location="cpu", weights_only=True).float()

    N = states["en"].shape[0]
    n_train = int(N * train_ratio)
    en_mean = states["en"][:n_train].mean(dim=0)

    non_en = [l for l in langs if l != "en"]
    deltas = {}
    for lang in non_en:
        lang_mean = states[lang][:n_train].mean(dim=0)
        deltas[lang] = en_mean - lang_mean
    return deltas, n_train


def compute_inlp_projection(states_dir, layer_idx, langs, n_iterations=5,
                             train_ratio=0.8):
    """Iterative Null-space Projection: remove language-predictive directions."""
    states = {}
    for lang in langs:
        fpath = Path(states_dir) / f"layer{layer_idx}_{lang}.pt"
        states[lang] = torch.load(fpath, map_location="cpu", weights_only=True).float()

    N = states["en"].shape[0]
    n_train = int(N * train_ratio)

    stack = torch.stack([states[l][:n_train] for l in langs], dim=0)
    mean = stack.mean(dim=0, keepdim=True)
    centered = stack - mean

    X = torch.cat([centered[i] for i in range(len(langs))], dim=0).numpy()
    y = np.concatenate([np.full(n_train, i) for i in range(len(langs))])
    D = X.shape[1]
    P = np.eye(D)

    for iteration in range(n_iterations):
        X_proj = X @ P
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X_proj, y)
        acc = clf.score(X_proj, y)
        print(f"  INLP iteration {iteration + 1}: acc={acc:.4f}")

        W = clf.coef_
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        n_remove = min(len(langs) - 1, Vt.shape[0])
        for d in Vt[:n_remove]:
            d = d / np.linalg.norm(d)
            P = P - np.outer(P @ d, d)
        if acc < 0.30:
            break

    return torch.from_numpy(P).float(), n_train


class NonOracleHook:
    """Hook that applies mean-shift or projection correction at the last token."""

    def __init__(self, correction_type, device, **kwargs):
        self.correction_type = correction_type
        self.device = device
        self.current_lang = None
        self.active = False
        self._handle = None

        if correction_type == "mean_shift":
            self.deltas = {l: v.to(device) for l, v in kwargs["deltas"].items()}
        elif correction_type == "projection":
            self.P = kwargs["projection"].to(device)

    def hook_fn(self, module, input, output):
        if not self.active or self.current_lang == "en" or self.current_lang is None:
            return output

        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        if hidden.dim() == 3:
            last = hidden[:, -1:, :]
        elif hidden.dim() == 2:
            last = hidden[-1:]
        else:
            return output

        with torch.no_grad():
            if self.correction_type == "mean_shift":
                delta = self.deltas.get(self.current_lang)
                if delta is not None:
                    corrected = (last.float() + delta.unsqueeze(0)).to(last.dtype)
                else:
                    corrected = last
            elif self.correction_type == "projection":
                corrected = (last.float() @ self.P.T).to(last.dtype)
            else:
                corrected = last

            hidden_new = hidden.clone()
            if hidden.dim() == 3:
                hidden_new[:, -1:, :] = corrected
            else:
                hidden_new[-1:] = corrected

        return (hidden_new,) + output[1:] if is_tuple else hidden_new

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()


def run_evaluation(model, processor, dataset, hooks, langs, device, out_dir, label):
    results = []
    non_en = [l for l in langs if l != "en"]

    for sample in tqdm(dataset, desc=label):
        image = Image.open(sample["image_path"]).convert("RGB")
        gt = normalize_answer(sample["answer"])
        entry = {"id": sample["id"], "gt": gt, "answers": {}, "normalized": {}, "correct": {}}

        for lang in langs:
            for h in hooks:
                h.current_lang = lang
                h.active = True

            q = sample["questions"].get(lang, "")
            if not q:
                continue
            prompt = q + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            ans = processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
            norm = normalize_answer(ans)
            entry["answers"][lang] = ans
            entry["normalized"][lang] = norm
            entry["correct"][lang] = norm == gt

        norms = [entry["normalized"][l] for l in langs if l in entry["normalized"]]
        entry["all_consistent"] = len(set(norms)) == 1
        results.append(entry)

    for h in hooks:
        h.active = False

    n = len(results)
    accs = {l: sum(1 for r in results if r["correct"].get(l, False)) / n for l in langs}
    non_en_mean = np.mean([accs[l] for l in non_en])

    summary = {
        "condition": label, "n_samples": n,
        "per_lang_acc": {l: float(v) for l, v in accs.items()},
        "gap": float(accs["en"] - non_en_mean),
    }
    print(f"  {label}: gap={summary['gap']:.3f}")

    with open(out_dir / f"raw_{label}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--states_dir", type=str, required=True,
                        help="Directory with pre-collected hidden states")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["mean_shift", "projection"])
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated intervention layer indices")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    langs = args.langs.split(",")
    intervention_layers = [int(x) for x in args.layers.split(",")]

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

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in intervention_layers:
        print(f"\nMethod: {args.method}, Layer: {layer}")

        if args.method == "mean_shift":
            deltas, n_train = compute_mean_shift_vectors(
                args.states_dir, layer, langs
            )
            hook = NonOracleHook("mean_shift", device, deltas=deltas)
        else:
            P, n_train = compute_inlp_projection(args.states_dir, layer, langs)
            hook = NonOracleHook("projection", device, projection=P)

        hook.register(model.model.language_model.layers[layer])
        run_evaluation(model, processor, dataset, [hook], langs, device,
                       out_dir, f"{args.method}_L{layer}")
        hook.remove()


if __name__ == "__main__":
    main()
