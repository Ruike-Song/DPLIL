"""
Control experiments to rule out alternative explanations:
  1. Same-language paraphrase — divergence is language-specific, not prompt sensitivity
  2. Translation-to-English — divergence comes from query language, not translation quality
  3. No-image — visual information is essential for non-EN answers
  4. Shuffled-image — language bias partially independent of visual content correctness
"""
import torch
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import SHORT_ANSWER_SUFFIX, normalize_answer


PARAPHRASE_TEMPLATES_EN = [
    lambda q: q,
    lambda q: f"Looking at this image, {q[0].lower()}{q[1:]}" if q else q,
    lambda q: f"Based on what you see, {q[0].lower()}{q[1:]}" if q else q,
    lambda q: f"Please tell me: {q}" if q else q,
    lambda q: f"Can you determine: {q}" if q else q,
]

PARAPHRASE_TEMPLATES_ZH = [
    lambda q: q,
    lambda q: f"看这张图片，{q}",
    lambda q: f"根据你所看到的，{q}",
    lambda q: f"请告诉我：{q}",
    lambda q: f"你能判断一下：{q}",
]


def generate_answer(model, processor, image, question, device,
                    lang="en", max_new_tokens=32):
    prompt = question + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content = [{"type": "image", "image": image}] + content

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def run_paraphrase(model, processor, dataset, langs, device, out_dir):
    results = []
    for sample in tqdm(dataset, desc="Paraphrase control"):
        image = Image.open(sample["image_path"]).convert("RGB")
        entry = {"id": sample["id"], "paraphrases": {}}

        for lang, templates in [("en", PARAPHRASE_TEMPLATES_EN), ("zh", PARAPHRASE_TEMPLATES_ZH)]:
            q_orig = sample["questions"].get(lang, "")
            if not q_orig:
                continue
            norms = []
            for template in templates:
                ans = generate_answer(model, processor, image, template(q_orig), device, lang=lang)
                norms.append(normalize_answer(ans))
            entry["paraphrases"][lang] = {"all_consistent": len(set(norms)) == 1}

        cross_norms = []
        for lang in langs:
            q = sample["questions"].get(lang, "")
            if q:
                ans = generate_answer(model, processor, image, q, device, lang=lang)
                cross_norms.append(normalize_answer(ans))
        entry["cross_lingual_consistent"] = len(set(cross_norms)) == 1
        results.append(entry)

    n = len(results)
    en_cons = sum(1 for r in results if r["paraphrases"].get("en", {}).get("all_consistent", False)) / n
    zh_cons = sum(1 for r in results if r["paraphrases"].get("zh", {}).get("all_consistent", False)) / n
    cross_cons = sum(1 for r in results if r.get("cross_lingual_consistent", False)) / n
    print(f"  EN paraphrase: {en_cons:.3f}, ZH: {zh_cons:.3f}, Cross-lingual: {cross_cons:.3f}")

    with open(out_dir / "paraphrase_summary.json", "w") as f:
        json.dump({"en": en_cons, "zh": zh_cons, "cross_lingual": cross_cons}, f, indent=2)


def run_no_image(model, processor, dataset, langs, device, out_dir):
    results = []
    for sample in tqdm(dataset, desc="No-image control"):
        gt = normalize_answer(sample["answer"])
        entry = {"id": sample["id"], "correct": {}}
        for lang in langs:
            q = sample["questions"].get(lang, "")
            if not q:
                continue
            ans = generate_answer(model, processor, None, q, device, lang=lang)
            entry["correct"][lang] = normalize_answer(ans) == gt
        results.append(entry)

    n = len(results)
    accs = {l: sum(1 for r in results if r["correct"].get(l, False)) / n for l in langs}
    non_en = [l for l in langs if l != "en"]
    gap = accs.get("en", 0) - np.mean([accs[l] for l in non_en])
    print(f"  Accs: {accs}, Gap: {gap:.3f}")

    with open(out_dir / "no_image_summary.json", "w") as f:
        json.dump({"per_lang_acc": accs, "gap": float(gap)}, f, indent=2)


def run_shuffled_image(model, processor, dataset, langs, device, out_dir):
    results = []
    for idx, sample in enumerate(tqdm(dataset, desc="Shuffled-image")):
        gt = normalize_answer(sample["answer"])
        wrong_idx = (idx + 1) % len(dataset)
        wrong_image = Image.open(dataset[wrong_idx]["image_path"]).convert("RGB")
        entry = {"id": sample["id"], "correct": {}}
        for lang in langs:
            q = sample["questions"].get(lang, "")
            if not q:
                continue
            ans = generate_answer(model, processor, wrong_image, q, device, lang=lang)
            entry["correct"][lang] = normalize_answer(ans) == gt
        results.append(entry)

    n = len(results)
    accs = {l: sum(1 for r in results if r["correct"].get(l, False)) / n for l in langs}
    non_en = [l for l in langs if l != "en"]
    gap = accs.get("en", 0) - np.mean([accs[l] for l in non_en])
    print(f"  Accs: {accs}, Gap: {gap:.3f}")

    with open(out_dir / "shuffled_image_summary.json", "w") as f:
        json.dump({"per_lang_acc": accs, "gap": float(gap)}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["paraphrase", "no_image", "shuffled_image", "all"])
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    langs = args.langs.split(",")

    dataset = json.load(open(args.data_path, encoding="utf-8"))
    if args.num_samples:
        dataset = dataset[:args.num_samples]

    print(f"Loading model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [args.experiment] if args.experiment != "all" else \
        ["paraphrase", "no_image", "shuffled_image"]

    for exp in experiments:
        print(f"\nRunning: {exp}")
        if exp == "paraphrase":
            run_paraphrase(model, processor, dataset, langs, device, out_dir)
        elif exp == "no_image":
            run_no_image(model, processor, dataset, langs, device, out_dir)
        elif exp == "shuffled_image":
            run_shuffled_image(model, processor, dataset, langs, device, out_dir)


if __name__ == "__main__":
    main()
