"""
Position diagnostic: compare cross-lingual divergence at different token positions.
  - Visual token positions (image tokens)
  - Last token position (answer generation decision point)
  - Post-visual text token positions

This establishes that language divergence is concentrated at the decision point
(last token), not at visual token positions — ruling out the alternative
explanation that visual encoding itself is language-conditioned.
"""
import torch
import json
import itertools
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


class MultiPositionHook:
    def __init__(self, image_token_id):
        self.hidden_states = None
        self.image_token_id = image_token_id
        self._handle = None

    def hook_fn(self, module, input, output):
        self.hidden_states = output[0].detach()

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()

    def get_states_at(self, input_ids, position_type="visual"):
        if self.hidden_states is None:
            return None
        hs = self.hidden_states.squeeze(0)
        ids = input_ids.squeeze(0)

        if position_type == "visual":
            mask = (ids == self.image_token_id)
            return hs[mask].float().cpu() if mask.any() else None
        elif position_type == "last":
            return hs[-1:].float().cpu()
        elif position_type == "post_visual":
            vis_mask = (ids == self.image_token_id)
            if not vis_mask.any():
                return None
            last_vis = vis_mask.nonzero().max().item()
            post_vis = hs[last_vis + 1:]
            return post_vis.float().cpu() if post_vis.shape[0] > 0 else None


def build_input(processor, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], images=[image], return_tensors="pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--image_token_id", type=int, required=True)
    parser.add_argument("--layers", type=str, required=True,
                        help="Comma-separated layer indices to test")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    layers_to_test = [int(x) for x in args.layers.split(",")]
    langs = args.langs.split(",")
    positions = ["visual", "last", "post_visual"]

    print("Loading model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path) as f:
        dataset = json.load(f)
    if args.n_samples:
        dataset = dataset[:args.n_samples]

    results = {}
    for layer_idx in layers_to_test:
        print(f"\nLayer {layer_idx}")
        hook = MultiPositionHook(args.image_token_id)
        hook.register(model.model.language_model.layers[layer_idx])

        pos_states = {pos: {lang: [] for lang in langs} for pos in positions}

        for item in tqdm(dataset, desc=f"L{layer_idx}"):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception:
                continue

            valid = True
            per_lang = {pos: {} for pos in positions}
            for lang in langs:
                q = item["questions"].get(lang)
                if not q:
                    valid = False
                    break
                inputs = build_input(processor, image, q)
                input_ids = inputs["input_ids"].to(device)
                inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    model(**inputs_gpu)
                for pos in positions:
                    s = hook.get_states_at(input_ids, pos)
                    if s is None:
                        valid = False
                        break
                    per_lang[pos][lang] = s
                if not valid:
                    break

            if not valid:
                continue
            for pos in positions:
                for lang in langs:
                    pos_states[pos][lang].append(per_lang[pos][lang])

        hook.remove()

        layer_results = {}
        for pos in positions:
            for lang in langs:
                if pos_states[pos][lang]:
                    pos_states[pos][lang] = torch.cat(pos_states[pos][lang], dim=0)
                else:
                    pos_states[pos][lang] = torch.empty(0)

            n_tok = pos_states[pos][langs[0]].shape[0]
            if n_tok == 0:
                continue

            pairs = list(itertools.combinations(langs, 2))
            cos_vals, l2_vals = [], []
            for l1, l2 in pairs:
                s1, s2 = pos_states[pos][l1], pos_states[pos][l2]
                min_n = min(s1.shape[0], s2.shape[0])
                s1, s2 = s1[:min_n], s2[:min_n]
                cos_vals.append(
                    torch.nn.functional.cosine_similarity(s1, s2, dim=-1).mean().item()
                )
                l2_vals.append((s1 - s2).norm(dim=-1).mean().item())

            layer_results[pos] = {
                "n_tokens": n_tok,
                "avg_cos_sim": float(np.mean(cos_vals)),
                "avg_l2_dist": float(np.mean(l2_vals)),
                "divergence": 1.0 - float(np.mean(cos_vals)),
            }
            print(f"  {pos:>12}: cos={np.mean(cos_vals):.6f}  L2={np.mean(l2_vals):.4f}")

        results[layer_idx] = layer_results
        torch.cuda.empty_cache()

    from pathlib import Path
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
