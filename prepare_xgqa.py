"""
Build a unified multilingual VQA dataset from xGQA + GQA.
xGQA provides the same questions in multiple languages;
GQA provides the images (Visual Genome) and ground truth answers.

Output: data/multilingual_vqa/dataset.json
"""
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def load_xgqa_questions(xgqa_dir: Path, langs: list[str]) -> dict:
    """Load xGQA questions for specified languages.
    Returns {qid: {lang: question_text, image_id, answer, ...}}."""
    all_questions = {}
    for lang in langs:
        fpath = xgqa_dir / f"testdev_balanced_questions_{lang}.json"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping {lang}")
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {lang}: {len(data)} questions loaded")
        for qid, item in data.items():
            if qid not in all_questions:
                all_questions[qid] = {
                    "image_id": item["imageId"],
                    "answer": item["answer"],
                    "full_answer": item.get("fullAnswer", ""),
                    "types": item.get("types", {}),
                    "questions": {},
                }
            all_questions[qid]["questions"][lang] = item["question"]
    return all_questions


def find_image_path(image_id: str, image_dir: Path) -> str | None:
    for ext in [".jpg", ".jpeg", ".png"]:
        path = image_dir / f"{image_id}{ext}"
        if path.exists():
            return str(path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgqa_dir", type=str, required=True,
                        help="Path to xGQA zero_shot data directory")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to GQA image directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the dataset JSON")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de",
                        help="Comma-separated language codes")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to keep (None = keep all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    langs = args.langs.split(",")
    image_dir = Path(args.image_dir)

    print(f"Languages: {langs}")
    print("Loading xGQA questions...")
    all_questions = load_xgqa_questions(Path(args.xgqa_dir), langs)
    print(f"  Total unique question IDs: {len(all_questions)}")

    complete = {
        qid: item for qid, item in all_questions.items()
        if all(lang in item["questions"] for lang in langs)
    }
    print(f"  With all {len(langs)} languages: {len(complete)}")

    print(f"Checking images in {image_dir}...")
    dataset = []
    missing_images = 0
    for qid, item in tqdm(complete.items(), desc="Matching images"):
        img_path = find_image_path(item["image_id"], image_dir)
        if img_path is None:
            missing_images += 1
            continue
        structural_type = item["types"].get("structural", "other")
        dataset.append({
            "id": qid,
            "image_id": item["image_id"],
            "image_path": img_path,
            "questions": item["questions"],
            "answer": item["answer"],
            "full_answer": item["full_answer"],
            "answer_type": structural_type,
        })
    print(f"  Valid samples (with images): {len(dataset)}")
    print(f"  Missing images: {missing_images}")

    if args.num_samples and args.num_samples < len(dataset):
        random.seed(args.seed)
        by_type = {}
        for d in dataset:
            t = d["answer_type"]
            by_type.setdefault(t, []).append(d)
        per_type = max(1, args.num_samples // len(by_type))
        sampled = []
        for t, items in by_type.items():
            k = min(per_type, len(items))
            sampled.extend(random.sample(items, k))
        if len(sampled) < args.num_samples:
            remaining = [d for d in dataset if d not in sampled]
            need = min(args.num_samples - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, need))
        dataset = sampled[:args.num_samples]
        print(f"  After sampling: {len(dataset)}")

    out_path = save_dir / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
