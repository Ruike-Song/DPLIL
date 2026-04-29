"""
Phenomenon confirmation: for each (image, question) pair, query the VLM in
multiple languages and compare answer consistency.

Demonstrates that the same visual question yields different answers depending
on query language — motivating the study of language-modulated visual features.
"""
import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import SHORT_ANSWER_SUFFIX, LANG_NAMES, normalize_answer


def generate_answer(model, processor, image, question, device,
                    lang="en", max_new_tokens=32):
    prompt = question + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    answer = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    print("Loading dataset...")
    with open(args.data_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset[:args.max_samples]

    langs = args.langs.split(",")
    print(f"  Using {len(dataset)} samples, Languages: {langs}")

    results = []
    for item in tqdm(dataset, desc="Pre-experiment"):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            continue

        answers = {}
        for lang in langs:
            q = item["questions"].get(lang)
            if not q:
                continue
            answers[lang] = generate_answer(model, processor, image, q, device, lang=lang)

        gt = item.get("answer", "")
        norm_answers = {l: normalize_answer(a) for l, a in answers.items()}
        gt_norm = normalize_answer(gt)
        ref_norm = norm_answers.get(langs[0], "")

        entry = {
            "id": item["id"],
            "gt_answer": gt,
            "answers": answers,
            "consistent": all(norm_answers.get(l, "") == ref_norm for l in langs[1:]),
            "correct": {l: (a == gt_norm) for l, a in norm_answers.items()},
        }
        results.append(entry)

    n = len(results)
    consistent_count = sum(1 for r in results if r["consistent"])
    per_lang_correct = {l: 0 for l in langs}
    for r in results:
        for l in langs:
            if r["correct"].get(l, False):
                per_lang_correct[l] += 1

    summary = {
        "num_samples": n,
        "languages": langs,
        "consistency_rate": consistent_count / n if n > 0 else 0,
        "per_lang_accuracy": {l: per_lang_correct[l] / n for l in langs},
    }

    with open(save_dir / "consistency.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nConsistency: {summary['consistency_rate']:.1%}")
    for l in langs:
        print(f"  {LANG_NAMES.get(l, l)}: {summary['per_lang_accuracy'][l]:.1%}")


if __name__ == "__main__":
    main()
