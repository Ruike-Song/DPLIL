"""
Cross-model mean-shift repair experiment.
Validates that the mean-shift intervention generalizes across different
VLM architectures by testing on multiple models at their respective
language routing zones.
"""
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import SHORT_ANSWER_SUFFIX, normalize_answer


def get_layer(model, model_type, layer_idx):
    """Access a decoder layer by index for the given model type."""
    if model_type == "llava":
        return model.model.language_model.layers[layer_idx]
    elif model_type == "internvl":
        return model.language_model.model.layers[layer_idx]
    elif model_type in ("qwen", "qwen8b"):
        return model.model.language_model.layers[layer_idx]
    raise ValueError(f"Unknown model type: {model_type}")


class MeanShiftHook:
    def __init__(self, langs, device):
        self.device = device
        self.langs = langs
        self.non_en = [l for l in langs if l != "en"]
        self.current_lang = None
        self.active = False
        self.deltas = {}
        self._handle = None
        self.collecting = False
        self.collected_states = {l: [] for l in langs}

    def collection_hook(self, module, input, output):
        if not self.collecting or self.current_lang is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() == 3:
            state = hidden[:, -1, :].detach().cpu().float()
        else:
            state = hidden[-1:].detach().cpu().float()
        self.collected_states[self.current_lang].append(state)
        return output

    def correction_hook(self, module, input, output):
        if not self.active or self.current_lang == "en" or self.current_lang is None:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.dim() == 3:
            last = hidden[:, -1:, :]
        else:
            last = hidden[-1:]

        with torch.no_grad():
            delta = self.deltas.get(self.current_lang)
            if delta is not None:
                corrected = (last.float() + delta.unsqueeze(0)).to(last.dtype)
            else:
                corrected = last
            hidden_new = hidden.clone()
            if hidden.dim() == 3:
                hidden_new[:, -1:, :] = corrected
            else:
                hidden_new[-1:] = corrected

        return (hidden_new,) + output[1:] if is_tuple else hidden_new

    def register_collection(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.collection_hook)
        return self

    def register_correction(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.correction_hook)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()

    def compute_deltas(self, train_ratio=0.8):
        en_states = torch.cat(self.collected_states["en"], dim=0)
        n_train = int(en_states.shape[0] * train_ratio)
        en_mean = en_states[:n_train].mean(dim=0)
        for lang in self.non_en:
            lang_states = torch.cat(self.collected_states[lang], dim=0)
            lang_mean = lang_states[:n_train].mean(dim=0)
            self.deltas[lang] = (en_mean - lang_mean).to(self.device)


def load_model_and_processor(model_path, model_type, device):
    if model_type == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_type in ("qwen", "qwen8b"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_type == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map=device, trust_remote_code=True
        )
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.eval()
    return model, processor


def generate_answer(model, processor, model_type, image, prompt, device):
    if model_type == "llava":
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    elif model_type in ("qwen", "qwen8b"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    elif model_type == "internvl":
        pixel_values = _load_internvl_image(image, model)
        generation_config = {"max_new_tokens": 32, "do_sample": False}
        return model.chat(processor, pixel_values, prompt, generation_config)
    else:
        return ""

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()


def _load_internvl_image(image, model):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0).to(model.device, dtype=model.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["llava", "internvl", "qwen", "qwen8b"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer indices for mean-shift intervention")
    parser.add_argument("--langs", type=str, default="en,zh,ko,de")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--calibration_samples", type=int, default=None,
                        help="Number of samples for calibration phase")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    langs = args.langs.split(",")
    non_en = [l for l in langs if l != "en"]

    dataset = json.load(open(args.data_path, encoding="utf-8"))
    if args.num_samples:
        dataset = dataset[:args.num_samples]
    calib_n = args.calibration_samples or int(len(dataset) * 0.8)

    print(f"Model: {args.model_type}, Layers: {args.layers}")
    model, processor = load_model_and_processor(args.model_path, args.model_type, device)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(desc):
        results = []
        for sample in tqdm(dataset, desc=desc):
            image = Image.open(sample["image_path"]).convert("RGB")
            gt = normalize_answer(sample["answer"])
            entry = {"id": sample["id"], "gt": gt, "answers": {}, "correct": {}}
            for lang in langs:
                q = sample["questions"].get(lang, "")
                if not q:
                    continue
                prompt = q + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
                if hasattr(hook, 'current_lang'):
                    hook.current_lang = lang
                ans = generate_answer(model, processor, args.model_type, image, prompt, device)
                norm = normalize_answer(ans)
                entry["answers"][lang] = ans
                entry["correct"][lang] = norm == gt
            results.append(entry)
        n = len(results)
        accs = {l: sum(1 for r in results if r["correct"].get(l, False)) / n for l in langs}
        non_en_mean = np.mean([accs[l] for l in non_en])
        return {"gap": float(accs["en"] - non_en_mean), "per_lang_acc": accs, "n": n}, results

    if not args.skip_baseline:
        class DummyHook:
            current_lang = None
        hook = DummyHook()
        summary, raw = evaluate("baseline")
        print(f"Baseline: gap={summary['gap']:.3f}")
        with open(out_dir / "baseline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    for layer_idx in args.layers:
        print(f"\nMean-shift at Layer {layer_idx}")
        hook = MeanShiftHook(langs, device)
        layer_module = get_layer(model, args.model_type, layer_idx)
        hook.register_collection(layer_module)
        hook.collecting = True

        for sample in tqdm(dataset[:calib_n], desc=f"Calibrating L{layer_idx}"):
            image = Image.open(sample["image_path"]).convert("RGB")
            for lang in langs:
                hook.current_lang = lang
                q = sample["questions"].get(lang, "")
                if not q:
                    continue
                prompt = q + SHORT_ANSWER_SUFFIX.get(lang, SHORT_ANSWER_SUFFIX["en"])
                generate_answer(model, processor, args.model_type, image, prompt, device)

        hook.collecting = False
        hook.remove()
        hook.compute_deltas()

        hook.register_correction(layer_module)
        hook.active = True
        summary, raw = evaluate(f"mean_shift_L{layer_idx}")
        hook.active = False
        hook.remove()

        print(f"  gap={summary['gap']:.3f}")
        with open(out_dir / f"mean_shift_L{layer_idx}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
