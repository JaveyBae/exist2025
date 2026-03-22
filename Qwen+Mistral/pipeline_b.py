import argparse
import ast
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from dataset import TASK6_LABELS, create_data_splits


TASK6_HUMAN_LABELS = {
    "IDEOLOGICAL-INEQUALITY": "ideological and inequality",
    "STEREOTYPING-DOMINANCE": "stereotyping and dominance",
    "OBJECTIFICATION": "objectification",
    "SEXUAL-VIOLENCE": "sexual violence",
    "MISOGYNY-NON-SEXUAL-VIOLENCE": "misogyny and non-sexual violence",
}

TASK6_HUMAN_TO_CANON = {v: k for k, v in TASK6_HUMAN_LABELS.items()}


@dataclass
class MemeRow:
    sample_id: str
    lang: str
    text: str
    image_path: str
    labels_task1: List[str]
    labels_task2: List[List[str]]
    task4: int
    task6: List[int]
    task4_soft: float
    task6_soft: List[float]


def parse_list(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if not value:
        return default
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def resolve_image_path(memes_dir: str, sample_id: str) -> str:
    for ext in (".jpeg", ".jpg", ".png", ".webp"):
        p = os.path.join(memes_dir, f"{sample_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(memes_dir, f"{sample_id}.jpeg")


def load_rows(data_dir: str, split_ids: List[str]) -> List[MemeRow]:
    csv_path = os.path.join(data_dir, "training", "memes_training.csv")
    memes_dir = os.path.join(data_dir, "training", "memes")

    by_id = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("id", "")).strip()
            if sid:
                by_id[sid] = row

    out: List[MemeRow] = []
    for sid in split_ids:
        if sid not in by_id:
            continue
        row = by_id[sid]
        labels_task1 = [str(x).upper() for x in parse_list(row.get("labels_task1"), default=[])]
        labels_task2 = parse_list(row.get("labels_task2"), default=[])

        task4 = int(float(row.get("task1_label", 0))) if str(row.get("task1_label", "")).strip() != "" else 0

        # Soft target for Task4: prefer annotator distribution, then task1_soft, then hard fallback.
        if labels_task1:
            yes = sum(1 for x in labels_task1 if str(x).upper() == "YES")
            task4_soft = yes / max(1, len(labels_task1))
        else:
            task1_soft = parse_list(row.get("task1_soft"), default=[])
            if isinstance(task1_soft, list) and len(task1_soft) >= 2:
                task4_soft = float(task1_soft[1])
            else:
                task4_soft = float(task4)

        task2_hard = parse_list(row.get("task2_hard"), default=[])
        if len(task2_hard) == 6:
            task6 = [1 if float(v) > 0 else 0 for v in task2_hard[1:6]]
        elif len(task2_hard) == 5:
            task6 = [1 if float(v) > 0 else 0 for v in task2_hard]
        else:
            task6 = [0] * len(TASK6_LABELS)

        # Soft target for Task6: prefer annotator frequencies, then task2_soft, then hard fallback.
        task6_soft = [0.0] * len(TASK6_LABELS)
        if isinstance(labels_task2, list) and labels_task2:
            valid_ann = 0
            counts = [0] * len(TASK6_LABELS)
            for ann in labels_task2:
                if not isinstance(ann, list):
                    continue
                valid_ann += 1
                ann_set = {str(x).upper() for x in ann}
                for i, lab in enumerate(TASK6_LABELS):
                    if lab in ann_set:
                        counts[i] += 1
            if valid_ann > 0:
                task6_soft = [c / valid_ann for c in counts]
            else:
                task6_soft = [float(x) for x in task6]
        else:
            task2_soft = parse_list(row.get("task2_soft"), default=[])
            if isinstance(task2_soft, list) and len(task2_soft) == 6:
                task6_soft = [float(v) for v in task2_soft[1:6]]
            elif isinstance(task2_soft, list) and len(task2_soft) == 5:
                task6_soft = [float(v) for v in task2_soft]
            else:
                task6_soft = [float(x) for x in task6]

        out.append(
            MemeRow(
                sample_id=sid,
                lang=str(row.get("lang", "es")),
                text=str(row.get("text", "")),
                image_path=resolve_image_path(memes_dir, sid),
                labels_task1=labels_task1,
                labels_task2=labels_task2 if isinstance(labels_task2, list) else [],
                task4=task4,
                task6=task6,
                task4_soft=task4_soft,
                task6_soft=task6_soft,
            )
        )

    return out


def task4_agreement(labels_task1: List[str]) -> float:
    if not labels_task1:
        return 0.0
    yes = sum(1 for x in labels_task1 if str(x).upper() == "YES")
    no = len(labels_task1) - yes
    return max(yes, no) / max(1, len(labels_task1))


def task6_label_agreement(labels_task2: List[List[str]], target_label: str) -> float:
    if not labels_task2:
        return 0.0
    n = 0
    pos = 0
    for ann in labels_task2:
        if not isinstance(ann, list):
            continue
        n += 1
        ann_set = {str(x).upper() for x in ann}
        if target_label in ann_set:
            pos += 1
    if n == 0:
        return 0.0
    return pos / n


def select_prototypes_task4(rows: List[MemeRow], k_per_class: int) -> Dict[str, List[MemeRow]]:
    yes_rows = [r for r in rows if r.task4 == 1]
    no_rows = [r for r in rows if r.task4 == 0]

    yes_rows.sort(key=lambda r: task4_agreement(r.labels_task1), reverse=True)
    no_rows.sort(key=lambda r: task4_agreement(r.labels_task1), reverse=True)

    return {
        "YES": yes_rows[:k_per_class],
        "NO": no_rows[:k_per_class],
    }


def select_prototypes_task4_full_consensus(rows: List[MemeRow], k_per_class: int) -> Dict[str, List[MemeRow]]:
    yes_rows = [r for r in rows if r.task4 == 1 and task4_agreement(r.labels_task1) >= 1.0]
    no_rows = [r for r in rows if r.task4 == 0 and task4_agreement(r.labels_task1) >= 1.0]
    if len(yes_rows) < k_per_class or len(no_rows) < k_per_class:
        return select_prototypes_task4(rows, k_per_class)
    return {"YES": yes_rows[:k_per_class], "NO": no_rows[:k_per_class]}


def select_prototypes_task6(rows: List[MemeRow], k_per_label: int) -> Dict[str, List[MemeRow]]:
    out = {}
    for i, label in enumerate(TASK6_LABELS):
        pos_rows = [r for r in rows if r.task6[i] == 1]
        pos_rows.sort(key=lambda r: task6_label_agreement(r.labels_task2, label), reverse=True)
        out[label] = pos_rows[:k_per_label]
    return out


def select_prototypes_task6_full_consensus(rows: List[MemeRow], k_per_label: int) -> Dict[str, List[MemeRow]]:
    out = {}
    for i, label in enumerate(TASK6_LABELS):
        pos_rows = [
            r
            for r in rows
            if r.task6[i] == 1 and task6_label_agreement(r.labels_task2, label) >= 1.0
        ]
        if len(pos_rows) < k_per_label:
            out[label] = select_prototypes_task6(rows, k_per_label)[label]
        else:
            out[label] = pos_rows[:k_per_label]
    return out


def build_description_prompt(lang: str, ocr: str) -> str:
    return (
        "You are analyzing a meme for sexism classification.\n"
        "Given the image and its embedded text, describe briefly and concisely the main situation or message conveyed. "
        "Focus on the social or gender implications. Avoid unnecessary visual details.\n"
        "Limit to one or two sentences.\n"
        f"Language: {lang}\n"
        f"OCR: {ocr}\n"
        "Describe this meme:"
    )


def build_text_only_description_prompt(lang: str, ocr: str) -> str:
    return (
        "You are analyzing meme OCR text for sexism classification.\n"
        "Write a concise semantic description of the message and social/gender implication.\n"
        "Do not invent visual details that are not present in OCR.\n"
        "Limit to one or two sentences.\n"
        f"Language: {lang}\n"
        f"OCR: {ocr}\n"
        "Description:"
    )


def build_vl_messages(image: Image.Image, user_prompt: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]


def generate_vl_description(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 96) -> str:
    messages = build_vl_messages(image, prompt)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def generate_text_llm(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_task4_answer(text: str) -> int:
    t = (text or "").strip().upper()
    lines = [x.strip() for x in t.splitlines() if x.strip()]

    # Prefer the first non-empty line to avoid false negatives when the model
    # outputs YES/NO first and then continues with extra text.
    if lines:
        first = lines[0]
        if re.fullmatch(r"YES", first):
            return 1
        if re.fullmatch(r"NO", first):
            return 0
        if re.search(r"\bYES\b", first):
            return 1
        if re.search(r"\bNO\b", first):
            return 0

    # Fallback: use the first YES/NO token found anywhere in output.
    m = re.search(r"\b(YES|NO)\b", t)
    if m:
        return 1 if m.group(1) == "YES" else 0
    return 0


def parse_task6_answer(text: str) -> List[int]:
    out = [0] * len(TASK6_LABELS)
    t = (text or "").strip()
    if not t:
        return out

    # Parse the first JSON object even if extra trailing text exists.
    json_candidate = t
    m = re.search(r"\{.*?\}", t, flags=re.S)
    if m:
        json_candidate = m.group(0)

    decoded_any = False
    try:
        obj = json.loads(json_candidate)
        labels = obj.get("labels", []) if isinstance(obj, dict) else []
        for it in labels:
            x = str(it).strip().upper()
            if x in TASK6_LABELS:
                out[TASK6_LABELS.index(x)] = 1
                decoded_any = True
                continue
            # Accept human-readable label strings from prompt examples.
            if x in {k.upper(): v for k, v in TASK6_HUMAN_TO_CANON.items()}:
                canon = {k.upper(): v for k, v in TASK6_HUMAN_TO_CANON.items()}[x]
                out[TASK6_LABELS.index(canon)] = 1
                decoded_any = True
        if decoded_any:
            return out
    except Exception:
        pass

    low = t.lower()
    if re.search(r"\bnone\b|\bno sexism\b|\bnon-sexist\b", low):
        return out

    for k, v in TASK6_HUMAN_TO_CANON.items():
        if re.search(rf"\b{re.escape(k)}\b", low):
            out[TASK6_LABELS.index(v)] = 1

    for i, lab in enumerate(TASK6_LABELS):
        if lab in t.upper():
            out[i] = 1

    return out


def build_task4_fewshot_prompt(desc: str, proto_descs: Dict[str, List[Tuple[str, str]]]) -> str:
    lines = []
    lines.append("You are a sexism classifier for memes.")
    lines.append("Task: predict if the meme is sexist.")
    lines.append("Output ONLY one token: YES or NO.")
    lines.append("Do not output any explanation or extra text.")
    lines.append("")
    lines.append("Few-shot examples:")
    for label in ("YES", "NO"):
        for sid, d in proto_descs.get(label, []):
            lines.append(f"- Description: {d}")
            lines.append(f"  Label: {label}")
    lines.append("")
    lines.append(f"Target description: {desc}")
    lines.append("Label:")
    return "\n".join(lines)


def build_task6_fewshot_prompt(desc: str, proto_descs: Dict[str, List[Tuple[str, str]]]) -> str:
    lines = []
    lines.append("You are a sexism-type classifier for memes.")
    lines.append("Task: predict sexism category labels from description.")
    lines.append("Allowed labels (use these exact canonical strings):")
    for lab in TASK6_LABELS:
        lines.append(f"- {lab}")
    lines.append("Output ONLY JSON: {\"labels\": [<label strings from allowed list>]}")
    lines.append("If no sexist category applies, output: {\"labels\": []}")
    lines.append("Do not output any explanation or extra text.")
    lines.append("")
    lines.append("Few-shot examples:")
    lines.append("- Description: neutral or non-sexist meme text")
    lines.append("  Output: {\"labels\": []}")
    for lab in TASK6_LABELS:
        for sid, d in proto_descs.get(lab, []):
            lines.append(f"- Description: {d}")
            lines.append(f"  Output: {{\"labels\": [\"{lab}\"]}}")
    lines.append("")
    lines.append(f"Target description: {desc}")
    lines.append("Output:")
    return "\n".join(lines)


def model_name_to_cache_folder(model_name: str) -> str:
    return "models--" + model_name.replace("/", "--")


def resolve_local_model_path(input_path: str, model_name: str) -> str:
    p = Path(input_path)

    if Path(model_name).exists() and (Path(model_name) / "config.json").exists():
        return str(Path(model_name))

    if p.exists() and (p / "config.json").exists():
        return str(p)

    snapshots_dir = p / model_name_to_cache_folder(model_name) / "snapshots"
    if snapshots_dir.exists():
        candidates = [x for x in snapshots_dir.iterdir() if x.is_dir() and (x / "config.json").exists()]
        if candidates:
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(candidates[0])

    return model_name


def load_or_generate_descriptions(
    rows: List[MemeRow],
    desc_cache_path: str,
    vl_model,
    vl_processor,
    max_new_tokens: int,
) -> Dict[str, str]:
    cache = {}
    if os.path.exists(desc_cache_path):
        with open(desc_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[str(obj["sample_id"])] = str(obj["description"])

    missing = [r for r in rows if r.sample_id not in cache]
    if not missing:
        return cache

    os.makedirs(os.path.dirname(desc_cache_path), exist_ok=True)
    with open(desc_cache_path, "a", encoding="utf-8") as fout:
        for r in tqdm(missing, desc="Generating descriptions"):
            try:
                image = Image.open(r.image_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (448, 448), color="gray")

            prompt = build_description_prompt(lang=r.lang, ocr=r.text)
            desc = generate_vl_description(vl_model, vl_processor, image, prompt, max_new_tokens=max_new_tokens)
            cache[r.sample_id] = desc
            fout.write(json.dumps({"sample_id": r.sample_id, "description": desc}, ensure_ascii=False) + "\n")

    return cache


def load_or_generate_text_only_descriptions(
    rows: List[MemeRow],
    desc_cache_path: str,
    text_model,
    text_tokenizer,
    max_new_tokens: int,
) -> Dict[str, str]:
    cache = {}
    if os.path.exists(desc_cache_path):
        with open(desc_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[str(obj["sample_id"])] = str(obj["description"])

    missing = [r for r in rows if r.sample_id not in cache]
    if not missing:
        return cache

    os.makedirs(os.path.dirname(desc_cache_path), exist_ok=True)
    with open(desc_cache_path, "a", encoding="utf-8") as fout:
        for r in tqdm(missing, desc="Generating text-only descriptions"):
            prompt = build_text_only_description_prompt(lang=r.lang, ocr=r.text)
            desc = generate_text_llm(text_model, text_tokenizer, prompt, max_new_tokens=max_new_tokens)
            cache[r.sample_id] = desc
            fout.write(json.dumps({"sample_id": r.sample_id, "description": desc}, ensure_ascii=False) + "\n")

    return cache


def evaluate_pipeline_b(
    test_rows: List[MemeRow],
    desc_cache: Dict[str, str],
    clf_model,
    clf_tokenizer,
    proto_task4_descs: Dict[str, List[Tuple[str, str]]],
    proto_task6_descs: Dict[str, List[Tuple[str, str]]],
    output_path: str,
    hierarchical_gate: bool = True,
    eval_mode: str = "both",
) -> Dict:
    y4_true, y4_pred = [], []
    y6_true, y6_pred = [], []
    y4_true_soft, y4_pred_soft = [], []
    y6_true_soft, y6_pred_soft = [], []
    recs = []

    for r in tqdm(test_rows, desc="Classifying"):
        desc = desc_cache.get(r.sample_id, "")

        p4 = build_task4_fewshot_prompt(desc=desc, proto_descs=proto_task4_descs)
        raw4 = generate_text_llm(clf_model, clf_tokenizer, p4, max_new_tokens=4)
        pred4 = parse_task4_answer(raw4)

        if hierarchical_gate and pred4 == 0:
            raw6 = "{\"labels\": []}"
            pred6 = [0] * len(TASK6_LABELS)
        else:
            p6 = build_task6_fewshot_prompt(desc=desc, proto_descs=proto_task6_descs)
            raw6 = generate_text_llm(clf_model, clf_tokenizer, p6, max_new_tokens=64)
            pred6 = parse_task6_answer(raw6)

        y4_true.append(r.task4)
        y4_pred.append(pred4)
        y6_true.append(r.task6)
        y6_pred.append(pred6)
        y4_true_soft.append(float(r.task4_soft))
        y4_pred_soft.append(float(pred4))
        y6_true_soft.append([float(x) for x in r.task6_soft])
        y6_pred_soft.append([float(x) for x in pred6])

        recs.append(
            {
                "sample_id": r.sample_id,
                "description": desc,
                "task4_true": r.task4,
                "task4_pred": pred4,
                "task4_raw": raw4,
                "task6_true": r.task6,
                "task6_pred": pred6,
                "task6_raw": raw6,
            }
        )

    y6_true_np = np.array(y6_true)
    y6_pred_np = np.array(y6_pred)

    metrics = {"num_samples": len(test_rows), "eval_mode": eval_mode}

    if eval_mode in ("hard-hard", "both"):
        metrics["hard_hard"] = {
            "task4": {
                "f1_binary": float(f1_score(y4_true, y4_pred, average="binary", zero_division=0)),
                "f1_macro": float(f1_score(y4_true, y4_pred, average="macro", zero_division=0)),
            },
            "task6": {
                "f1_macro": float(f1_score(y6_true_np, y6_pred_np, average="macro", zero_division=0)),
                "f1_micro": float(f1_score(y6_true_np, y6_pred_np, average="micro", zero_division=0)),
                "f1_samples": float(f1_score(y6_true_np, y6_pred_np, average="samples", zero_division=0)),
            },
        }

    if eval_mode in ("soft-soft", "both"):
        y4_true_soft_np = np.array(y4_true_soft, dtype=np.float32)
        y4_pred_soft_np = np.array(y4_pred_soft, dtype=np.float32)
        y6_true_soft_np = np.array(y6_true_soft, dtype=np.float32)
        y6_pred_soft_np = np.array(y6_pred_soft, dtype=np.float32)

        task4_brier = float(np.mean((y4_pred_soft_np - y4_true_soft_np) ** 2))
        task6_brier_per_label = np.mean((y6_pred_soft_np - y6_true_soft_np) ** 2, axis=0)
        task6_brier_macro = float(np.mean(task6_brier_per_label))
        task6_brier_micro = float(np.mean((y6_pred_soft_np - y6_true_soft_np) ** 2))

        metrics["soft_soft"] = {
            "task4": {
                "brier": task4_brier,
            },
            "task6": {
                "brier_macro": task6_brier_macro,
                "brier_micro": task6_brier_micro,
            },
        }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_path, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Pipeline B: Qwen-VL description + Mistral few-shot classifier")
    parser.add_argument("--data_dir", type=str, default="/root/QWen/EXIST 2024 Memes Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_split", action="store_true")

    parser.add_argument("--vl_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--classifier_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model_cache_dir", type=str, default="/root/autodl-tmp/hf-cache")

    parser.add_argument("--k_task4", type=int, default=3)
    parser.add_argument("--k_task6", type=int, default=2)
    parser.add_argument("--prototype_mode", type=str, default="ranked", choices=["ranked", "full-consensus"])

    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_test_rows", type=int, default=0)
    parser.add_argument("--max_desc_tokens", type=int, default=96)
    parser.add_argument("--eval_mode", type=str, default="both", choices=["hard-hard", "soft-soft", "both"])
    parser.add_argument("--disable_hierarchical_gate", action="store_true")
    parser.add_argument("--text_only_demo", action="store_true", help="Use OCR-only description generation with a text LLM")

    parser.add_argument("--output_dir", type=str, default="/root/QWen/outputs/pipeline_b")
    parser.add_argument("--desc_cache_path", type=str, default="/root/QWen/outputs/pipeline_b/descriptions.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits = create_data_splits(args.data_dir, seed=args.seed, force_split=args.force_split)
    train_ids = splits.get("train", [])
    test_ids = splits.get("test", [])

    train_rows = load_rows(args.data_dir, train_ids)
    test_rows = load_rows(args.data_dir, test_ids)

    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_test_rows > 0:
        test_rows = test_rows[: args.max_test_rows]

    print(f"[data] train/test rows: {len(train_rows)}/{len(test_rows)}")

    if args.prototype_mode == "full-consensus":
        proto_task4 = select_prototypes_task4_full_consensus(train_rows, k_per_class=args.k_task4)
        proto_task6 = select_prototypes_task6_full_consensus(train_rows, k_per_label=args.k_task6)
    else:
        proto_task4 = select_prototypes_task4(train_rows, k_per_class=args.k_task4)
        proto_task6 = select_prototypes_task6(train_rows, k_per_label=args.k_task6)

    all_proto_ids = set()
    for rows in proto_task4.values():
        for r in rows:
            all_proto_ids.add(r.sample_id)
    for rows in proto_task6.values():
        for r in rows:
            all_proto_ids.add(r.sample_id)

    row_by_id = {r.sample_id: r for r in train_rows + test_rows}
    proto_rows = [row_by_id[sid] for sid in all_proto_ids if sid in row_by_id]

    desc_rows = proto_rows + test_rows
    if args.text_only_demo:
        demo_model_path = resolve_local_model_path(args.model_cache_dir, args.classifier_model_name)
        print(f"[demo] text-only model path: {demo_model_path}")
        demo_tokenizer = AutoTokenizer.from_pretrained(demo_model_path, cache_dir=args.model_cache_dir)
        demo_model = AutoModelForCausalLM.from_pretrained(
            demo_model_path,
            cache_dir=args.model_cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        demo_model.eval()

        desc_cache = load_or_generate_text_only_descriptions(
            rows=desc_rows,
            desc_cache_path=args.desc_cache_path,
            text_model=demo_model,
            text_tokenizer=demo_tokenizer,
            max_new_tokens=args.max_desc_tokens,
        )

        del demo_model
        del demo_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        vl_model_path = resolve_local_model_path(args.model_cache_dir, args.vl_model_name)
        print(f"[vl] model path: {vl_model_path}")
        vl_processor = AutoProcessor.from_pretrained(vl_model_path, trust_remote_code=True, cache_dir=args.model_cache_dir)
        vl_model = AutoModelForImageTextToText.from_pretrained(
            vl_model_path,
            trust_remote_code=True,
            cache_dir=args.model_cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        vl_model.eval()

        desc_cache = load_or_generate_descriptions(
            rows=desc_rows,
            desc_cache_path=args.desc_cache_path,
            vl_model=vl_model,
            vl_processor=vl_processor,
            max_new_tokens=args.max_desc_tokens,
        )

        # Release VL resources before loading classifier model.
        del vl_model
        del vl_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    proto_task4_descs = {
        "YES": [(r.sample_id, desc_cache.get(r.sample_id, "")) for r in proto_task4["YES"]],
        "NO": [(r.sample_id, desc_cache.get(r.sample_id, "")) for r in proto_task4["NO"]],
    }
    proto_task6_descs = {
        label: [(r.sample_id, desc_cache.get(r.sample_id, "")) for r in rows]
        for label, rows in proto_task6.items()
    }

    clf_model_path = resolve_local_model_path(args.model_cache_dir, args.classifier_model_name)
    print(f"[clf] model path: {clf_model_path}")
    clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_path, cache_dir=args.model_cache_dir)
    clf_model = AutoModelForCausalLM.from_pretrained(
        clf_model_path,
        cache_dir=args.model_cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    clf_model.eval()

    metrics = evaluate_pipeline_b(
        test_rows=test_rows,
        desc_cache=desc_cache,
        clf_model=clf_model,
        clf_tokenizer=clf_tokenizer,
        proto_task4_descs=proto_task4_descs,
        proto_task6_descs=proto_task6_descs,
        output_path=args.output_dir,
        hierarchical_gate=not args.disable_hierarchical_gate,
        eval_mode=args.eval_mode,
    )

    print("\n==== Pipeline B done ====")
    if "hard_hard" in metrics:
        print(f"[hard-hard] Task4 F1(binary): {metrics['hard_hard']['task4']['f1_binary']:.4f}")
        print(f"[hard-hard] Task4 F1(macro):  {metrics['hard_hard']['task4']['f1_macro']:.4f}")
        print(f"[hard-hard] Task6 F1(macro):  {metrics['hard_hard']['task6']['f1_macro']:.4f}")
        print(f"[hard-hard] Task6 F1(micro):  {metrics['hard_hard']['task6']['f1_micro']:.4f}")
        print(f"[hard-hard] Task6 F1(samples):{metrics['hard_hard']['task6']['f1_samples']:.4f}")
    if "soft_soft" in metrics:
        print(f"[soft-soft] Task4 Brier:      {metrics['soft_soft']['task4']['brier']:.4f}")
        print(f"[soft-soft] Task6 Brier(macro): {metrics['soft_soft']['task6']['brier_macro']:.4f}")
        print(f"[soft-soft] Task6 Brier(micro): {metrics['soft_soft']['task6']['brier_micro']:.4f}")
    print(f"samples: {metrics['num_samples']}")
    print(f"metrics path: {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
