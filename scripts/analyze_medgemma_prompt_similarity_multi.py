import autorootcwd
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from difflib import SequenceMatcher
from tqdm import tqdm


# ✅ 목적에 맞춘 쿼리(구조/난이도/아티팩트/비혈관 혼입 신호)
# - "ONE line" 고정
# - 가능하면 동일한 포맷을 강제해서 샘플 간 차이를 더 잘 관측
DEFAULT_PROMPTS = [
    "Describe any abnormal or noteworthy findings visible in this angiogram image. "
    "Be concise and factual. Output ONE short sentence."
]

# -----------------------------
# Logits-feature design (A)
# -----------------------------
LEVELS = ["A", "B", "C", "D", "E", "F", "G", "H"]  # A=very_low ... H=max
LABELS = ["low_contrast", "motion", "overlap", "catheter", "clutter", "noise", "blur", "other"]
LABELS_FP = ["background", "catheter", "bone", "saturation", "texture", "artifact", "blur", "other"]
UNK_LABEL = "unknown"

# Rad-Explain style template (system + user in one prompt)
RAD_EXPLAIN_TEMPLATE = (
    "You are a public-facing clinician. "
    "A learning user has provided a sentence from a radiology report and is viewing the accompanying {IMAGE_TYPE} image. "
    "Your task is to explain the meaning of ONLY the provided sentence in simple, clear terms. "
    "Explain terminology and abbreviations. Keep it concise. "
    "Directly address the meaning of the sentence. Do not use introductory phrases like \"Okay\" or refer to the sentence itself or the report itself. "
    "Crucially, since the user is looking at their {IMAGE_TYPE} image, provide guidance on where to look on the image to understand your explanation, if applicable. "
    "Do not discuss any other part of the report or any sentences not explicitly provided by the user. "
    "Stick to facts in the text. Do not infer anything. "
    "=== For context, the full REPORT is: {FULL_REPORT_TEXT} "
    "=== Explain this sentence from the radiology report: \"{SELECTED_SENTENCE}\""
)


def build_rad_explain_system_prompt(report_text: str, image_type: str) -> str:
    base = (
        "You are a public-facing clinician. "
        "A learning user has provided a sentence from a radiology report and is viewing the accompanying {IMAGE_TYPE} image. "
        "Your task is to explain the meaning of ONLY the provided sentence in simple, clear terms. "
        "Explain terminology and abbreviations. Keep it concise. "
        "Directly address the meaning of the sentence. Do not use introductory phrases like \"Okay\" or refer to the sentence itself or the report itself. "
        "Crucially, since the user is looking at their {IMAGE_TYPE} image, provide guidance on where to look on the image to understand your explanation, if applicable. "
        "Do not discuss any other part of the report or any sentences not explicitly provided by the user. "
        "Stick to facts in the text. Do not infer anything. "
    )
    report = report_text.strip()
    if report:
        base += f"=== For context, the full REPORT is: {report} "
    return base.format(IMAGE_TYPE=image_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MedGemma-4B VLM prompt dispersion test.\n"
            "- Generates responses for multiple prompts\n"
            "- Computes text-embedding dispersion and text similarity\n"
            "- Saves responses and summary stats\n"
        )
    )
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with images.")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for images.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all).")

    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompt", action="append", help="Prompt string. Can be used multiple times.")
    prompt_group.add_argument("--prompts-file", type=str, help="Text file with one prompt per line.")
    parser.add_argument(
        "--use-rad-explain-prompts",
        action="store_true",
        help="Interpret prompts as report sentences and wrap with rad_explain-style template.",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Full report text file used for rad_explain-style prompts.",
    )
    parser.add_argument(
        "--image-type",
        type=str,
        default="radiology",
        help="Image type string for rad_explain template (e.g., radiology, CXR, angiogram).",
    )

    # ✅ 기본값을 MedGemma-4B IT로 설정 (HF 모델 카드 기준)
    parser.add_argument("--model-name", type=str, default="google/medgemma-4b-it", help="HF model id (MedGemma VLM).")

    # ✅ 게이트 모델 접근을 위한 토큰 옵션 (또는 HF_TOKEN env 사용)
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env).")

    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--device-map", type=str, default="auto", help="auto|none (use 'none' for single GPU)")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")

    parser.add_argument(
        "--analysis-mode",
        type=str,
        default="logits",
        choices=["logits", "text"],
        help="logits: candidate-token logits features (default). text: text-embedding dispersion.",
    )

    # generation controls (표현 다양성/분산 조절)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (use with --do-sample).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p (use with --do-sample).")

    parser.add_argument(
        "--feature-mode",
        type=str,
        default="logprob",
        choices=["logprob", "logit"],
        help="Use candidate log-probabilities (recommended) or raw logits as features.",
    )
    parser.add_argument(
        "--logit-temp",
        type=float,
        default=1.0,
        help="Temperature for candidate logits before logprob/logit extraction.",
    )
    parser.add_argument("--center", action="store_true", help="Center features per prompt (subtract mean feature).")
    parser.add_argument("--l2norm", action="store_true", help="L2-normalize features per sample (per prompt).")

    # embedding pool
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "last"], help="Embedding pool.")
    parser.add_argument("--output-dir", type=str, default="outputs/medgemma_prompt_dispersion")

    # sometimes needed for some HF repos
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_report_text(report_file: Optional[str]) -> str:
    if not report_file:
        return ""
    path = Path(report_file)
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt:
        raw = [p.strip() for p in args.prompt if p.strip()]
    elif args.prompts_file:
        prompts_path = Path(args.prompts_file)
        raw = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    else:
        raw = list(DEFAULT_PROMPTS)

    if not args.use_rad_explain_prompts:
        return raw

    if not raw:
        raise ValueError("rad_explain prompts require at least one sentence via --prompt or --prompts-file.")

    return raw


def safe_cosine_similarity_matrix(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(vecs, axis=1, keepdims=True)
    denom = np.clip(denom, eps, None)
    normed = vecs / denom
    return np.dot(normed, normed.T)


def text_similarity_matrix(texts: List[str]) -> np.ndarray:
    n = len(texts)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = SequenceMatcher(None, texts[i], texts[j]).ratio()
    return sim


def write_matrix_csv(path: Path, labels: List[str], mat: np.ndarray) -> None:
    lines = ["," + ",".join(labels)]
    for i, label in enumerate(labels):
        row = [label] + [f"{mat[i, j]:.6f}" for j in range(len(labels))]
        lines.append(",".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def _to_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    key = dtype_str.lower()
    if key == "float16":
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    if key == "float32":
        return torch.float32
    return "auto"


def _build_messages(prompt: str, image: Image.Image, system_prompt: Optional[str]) -> List[Dict]:
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    )
    return messages


def _apply_chat(
    processor,
    messages: List[Dict],
    add_generation_prompt: bool,
) -> Dict[str, torch.Tensor]:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _move_inputs(inputs: Dict[str, torch.Tensor], device: torch.device, dtype: Optional[torch.dtype]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if dtype is not None and v.is_floating_point():
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device=device)
    return out


def _load_model_and_processor(model_name: str, device: torch.device, device_map: Optional[str], dtype, hf_token: Optional[str], trust_remote_code: bool):
    from transformers import AutoProcessor

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except Exception:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    if device_map is not None and device_map != "none":
        model = model_cls.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        model_device = None
    else:
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
        model_device = device

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, processor, model_device


def _get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _tokenize_mm(
    processor,
    image: Image.Image,
    text: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    if dtype is None:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device, dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


@torch.inference_mode()
def _forward_logits(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs, return_dict=True)
    return out.logits


def _safe_single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Tokenizer produced empty ids for candidate '{s}'")
    return ids[0]


def _extract_answer_value_positions(answer: str, prompt_idx: int) -> Dict[str, int]:
    ans = answer.strip()

    if prompt_idx == 0:
        m = re.fullmatch(r"continuity=([A-H]);breaks=([A-H]);cause=([a-z_]+)", ans)
        if not m:
            return {}
        cpos = ans.find("continuity=") + len("continuity=")
        bpos = ans.find("breaks=") + len("breaks=")
        pos = ans.find("cause=") + len("cause=")
        return {"continuity": cpos, "breaks": bpos, "cause": pos}

    if prompt_idx == 1:
        m = re.fullmatch(r"nonvessel=([A-H]);fp_risk=([A-H]);source=([a-z_]+)", ans)
        if not m:
            return {}
        npos = ans.find("nonvessel=") + len("nonvessel=")
        fpos = ans.find("fp_risk=") + len("fp_risk=")
        spos = ans.find("source=") + len("source=")
        return {"nonvessel": npos, "fp_risk": fpos, "source": spos}

    if prompt_idx == 2:
        m = re.fullmatch(r"thin=([A-H]);periphery=([A-H]);visibility=([A-H])", ans)
        if not m:
            return {}
        tpos = ans.find("thin=") + len("thin=")
        ppos = ans.find("periphery=") + len("periphery=")
        vpos = ans.find("visibility=") + len("visibility=")
        return {"thin": tpos, "periphery": ppos, "visibility": vpos}

    return {}


def _find_token_index_by_charpos(tokenizer, text: str, char_pos: int) -> Optional[int]:
    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except TypeError:
        return None

    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        return None

    offsets = offsets[0].tolist()
    for t_idx, (s, e) in enumerate(offsets):
        if s <= char_pos < e:
            return t_idx
    return None


def _candidate_feature_for_field(
    logits_step: torch.Tensor,
    cand_token_ids: List[int],
    mode: str,
    temp: float,
) -> np.ndarray:
    cand_logits = logits_step[cand_token_ids].float()
    if temp is None or temp <= 0:
        temp = 1.0
    cand_logits = cand_logits / float(temp)
    if mode == "logit":
        return cand_logits.detach().cpu().numpy()
    logp = torch.log_softmax(cand_logits, dim=0)
    return logp.detach().cpu().numpy()


@torch.inference_mode()
def extract_logits_feature(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    answer: str,
    prompt_idx: int,
    feature_mode: str = "logprob",
    logit_temp: float = 1.0,
    system_prompt: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, str]]:
    tok = _get_tokenizer(processor)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Build tokenized prefix/full via chat template (MedGemma-style)
    prefix_msgs = _build_messages(prompt, image, system_prompt)
    prefix_inputs = _apply_chat(processor, prefix_msgs, add_generation_prompt=True)
    prefix_inputs = _move_inputs(prefix_inputs, device=device, dtype=dtype)

    full_msgs = _build_messages(prompt, image, system_prompt)
    full_msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    full_inputs = _apply_chat(processor, full_msgs, add_generation_prompt=False)
    full_inputs = _move_inputs(full_inputs, device=device, dtype=dtype)

    prefix_ids = prefix_inputs.get("input_ids")
    full_ids = full_inputs.get("input_ids")
    if prefix_ids is None or full_ids is None:
        raise RuntimeError("Processor did not return input_ids for logits feature extraction.")

    prefix_len = int(prefix_ids.shape[1])
    full_len = int(full_ids.shape[1])

    logits = _forward_logits(model, full_inputs)

    positions = _extract_answer_value_positions(answer, prompt_idx)
    debug: Dict[str, str] = {}

    if prompt_idx == 0:
        label_cands = list(LABELS) + [UNK_LABEL]
        label_ids = [_safe_single_token_id(tok, s) for s in label_cands]
        level_ids = [_safe_single_token_id(tok, s) for s in LEVELS]
        cand_ids_map = {
            "continuity": level_ids,
            "breaks": level_ids,
            "cause": label_ids,
        }
    elif prompt_idx == 1:
        label_cands = list(LABELS_FP) + [UNK_LABEL]
        label_ids = [_safe_single_token_id(tok, s) for s in label_cands]
        level_ids = [_safe_single_token_id(tok, s) for s in LEVELS]
        cand_ids_map = {
            "nonvessel": level_ids,
            "fp_risk": level_ids,
            "source": label_ids,
        }
    else:
        level_ids = [_safe_single_token_id(tok, s) for s in LEVELS]
        cand_ids_map = {
            "thin": level_ids,
            "periphery": level_ids,
            "visibility": level_ids,
        }

    if not positions:
        debug["parse_error"] = "answer_format_mismatch"
        if prompt_idx in (0, 1):
            label_len = len(LABELS) + 1
            K = 2 * len(LEVELS) + label_len
        else:
            K = 3 * len(LEVELS)
        probs = np.ones((K,), dtype=np.float32) / float(K)
        feat = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)
        return feat, debug

    feat_parts: List[np.ndarray] = []
    for field, char_pos in positions.items():
        t_idx_ans = _find_token_index_by_charpos(tok, answer, char_pos)
        if t_idx_ans is None:
            debug[f"{field}_align_error"] = "no_offsets_mapping_or_charpos_not_found"
            K = len(cand_ids_map[field])
            probs = np.ones((K,), dtype=np.float32) / float(K)
            feat_parts.append(np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32))
            continue
        t_idx_full = prefix_len + t_idx_ans
        step = t_idx_full - 1
        if step < 0 or step >= full_len:
            debug[f"{field}_align_error"] = f"step_out_of_range(step={step}, full_len={full_len})"
            K = len(cand_ids_map[field])
            probs = np.ones((K,), dtype=np.float32) / float(K)
            feat_parts.append(np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32))
            continue
        logits_step = logits[0, step, :]
        cand_ids = cand_ids_map.get(field, [])
        if not cand_ids:
            debug[f"{field}_cand_error"] = "empty_candidates"
            K = 1
            probs = np.ones((K,), dtype=np.float32)
            feat_parts.append(np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32))
            continue
        f = _candidate_feature_for_field(logits_step, cand_ids, mode=feature_mode, temp=logit_temp)
        feat_parts.append(f)
        debug[f"{field}_token_index_full"] = str(t_idx_full)
        debug[f"{field}_token_index_answer"] = str(t_idx_ans)

    feat = np.concatenate(feat_parts, axis=0).astype(np.float32)
    debug["prefix_len"] = str(prefix_len)
    debug["full_len"] = str(full_len)
    debug["feature_dim"] = str(int(feat.shape[0]))
    return feat, debug


def postprocess_features(
    feats_by_prompt: List[np.ndarray],
    center: bool,
    l2norm: bool,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for X in feats_by_prompt:
        Y = X.astype(np.float32, copy=True)
        if center:
            mu = Y.mean(axis=0, keepdims=True)
            Y = Y - mu
        if l2norm:
            denom = np.linalg.norm(Y, axis=1, keepdims=True)
            denom = np.clip(denom, 1e-12, None)
            Y = Y / denom
        out.append(Y)
    return out


def generate_text(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str] = None,
) -> str:
    messages = _build_messages(prompt, image, system_prompt)
    inputs = _apply_chat(processor, messages, add_generation_prompt=True)

    # best-effort device placement
    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_device = torch.device("cpu")
        model_dtype = None

    inputs = _move_inputs(inputs, device=model_device, dtype=model_dtype)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

    output_ids = model.generate(**inputs, **gen_kwargs)

    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, output_ids)]
    else:
        gen_ids = output_ids

    tok = _get_tokenizer(processor)
    text_out = tok.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text_out.replace("\n", " ").strip()


def embed_text(model, processor, text: str, pool: str = "mean") -> torch.Tensor:
    tok = _get_tokenizer(processor)
    tokens = tok(text, return_tensors="pt", padding=True, truncation=True)

    # Try to find the language model component
    text_model = getattr(model, "language_model", None)
    if text_model is None:
        text_model = getattr(model, "text_model", None)
    if text_model is None:
        text_model = getattr(model, "model", model)

    try:
        text_device = next(text_model.parameters()).device
    except StopIteration:
        text_device = next(model.parameters()).device

    tokens = {k: v.to(text_device) for k, v in tokens.items() if isinstance(v, torch.Tensor)}

    try:
        outputs = text_model(**tokens, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
        mask = tokens.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
        else:
            mask = mask.to(hidden.device)
        mask = mask.unsqueeze(-1)

        if pool == "last":
            idx = (mask.squeeze(-1).sum(dim=1) - 1).clamp(min=0).long()
            eos_id = getattr(tok, "eos_token_id", None)
            if eos_id is not None and "input_ids" in tokens:
                input_ids = tokens["input_ids"]
                eos_positions = (input_ids == eos_id).long()
                has_eos = eos_positions.sum(dim=1) > 0
                if has_eos.any():
                    last_eos = eos_positions.cumsum(dim=1).eq(eos_positions.sum(dim=1, keepdim=True))
                    last_eos_idx = last_eos.float().argmax(dim=1)
                    idx = torch.where(has_eos, last_eos_idx, idx)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        else:
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        return pooled
    except Exception:
        # Fallback: average input embeddings
        emb = text_model.get_input_embeddings()(tokens["input_ids"])
        if pool == "last":
            pooled = emb[:, -1, :]
        else:
            pooled = emb.mean(dim=1)
        return pooled


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prompts = load_prompts(args)
    report_text = _load_report_text(args.report_file) if args.use_rad_explain_prompts else ""
    rad_system_prompt = (
        build_rad_explain_system_prompt(report_text, args.image_type)
        if args.use_rad_explain_prompts
        else None
    )
    if args.use_rad_explain_prompts:
        args.do_sample = False
        args.temperature = 0.0
        args.top_p = 1.0

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob(args.pattern))
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir} with pattern {args.pattern}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_sample").mkdir(parents=True, exist_ok=True)
    if args.use_rad_explain_prompts:
        wrapped = []
        for sent in prompts:
            wrapped.append(
                RAD_EXPLAIN_TEMPLATE.format(
                    IMAGE_TYPE=args.image_type,
                    FULL_REPORT_TEXT=report_text,
                    SELECTED_SENTENCE=sent,
                )
            )
        (output_dir / "prompts.txt").write_text("\n\n".join(wrapped), encoding="utf-8")
    else:
        (output_dir / "prompts.txt").write_text("\n".join(prompts), encoding="utf-8")

    device_map = None if args.device_map == "none" else args.device_map
    dtype = _to_dtype(args.dtype)

    model, processor, _ = _load_model_and_processor(
        args.model_name, device, device_map, dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    labels = [f"p{i}" for i in range(len(prompts))]
    sample_labels = [p.stem for p in image_paths]

    responses_path = output_dir / "responses.jsonl"
    all_texts_by_prompt: List[List[str]] = [[] for _ in range(len(prompts))]

    if args.analysis_mode == "text":
        all_embs_by_prompt: List[List[np.ndarray]] = [[] for _ in range(len(prompts))]
        with responses_path.open("w", encoding="utf-8") as f_out:
            for img_path in tqdm(image_paths, desc="Images"):
                image = Image.open(img_path).convert("RGB")

                texts: List[str] = []
                for prompt_idx, prompt in enumerate(prompts):
                    text = generate_text(
                        model, processor, image, prompt,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        system_prompt=rad_system_prompt,
                    )
                    texts.append(text)
                    all_texts_by_prompt[prompt_idx].append(text)

                    e_text = embed_text(model, processor, text, pool=args.pool).float().cpu().numpy().squeeze()
                    all_embs_by_prompt[prompt_idx].append(e_text)

                    f_out.write(
                        json.dumps(
                            {
                                "image": img_path.name,
                                "prompt_idx": prompt_idx,
                                "prompt": prompt,
                                "system": rad_system_prompt,
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                txt_sim = text_similarity_matrix(texts)
                base = output_dir / "per_sample" / img_path.stem
                write_matrix_csv(base.with_suffix(".text_sim.csv"), labels, txt_sim)

                resp_lines = []
                for prompt_idx, prompt in enumerate(prompts):
                    resp_lines.append(f"[p{prompt_idx}] {prompt}")
                    resp_lines.append(texts[prompt_idx])
                    resp_lines.append("")
                (base.with_suffix(".responses.txt")).write_text("\n".join(resp_lines), encoding="utf-8")

        summary = []
        for pidx in range(len(prompts)):
            X = np.stack(all_embs_by_prompt[pidx], axis=0).astype(np.float32)
            sim = safe_cosine_similarity_matrix(X)
            write_matrix_csv(output_dir / f"prompt_{pidx}_sample_text_cosine.csv", sample_labels, sim)

            n = sim.shape[0]
            off = sim[~np.eye(n, dtype=bool)]
            l2 = np.linalg.norm(X, axis=1)
            summary.append(
                {
                    "prompt_idx": pidx,
                    "prompt_label": labels[pidx],
                    "mean_offdiag_cosine": float(off.mean()) if off.size else float("nan"),
                    "std_offdiag_cosine": float(off.std()) if off.size else float("nan"),
                    "min_offdiag_cosine": float(off.min()) if off.size else float("nan"),
                    "max_offdiag_cosine": float(off.max()) if off.size else float("nan"),
                    "l2_mean": float(l2.mean()),
                    "l2_std": float(l2.std()),
                    "l2_min": float(l2.min()),
                    "l2_max": float(l2.max()),
                }
            )

        (output_dir / "summary_text_embed.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[OK] Model: {args.model_name}")
        print(f"[OK] Saved responses to: {responses_path}")
        print(f"[OK] Saved per-sample matrices to: {output_dir / 'per_sample'}")
        print(f"[OK] Saved summary to: {output_dir / 'summary_text_embed.json'}")
        return

    all_feats_by_prompt: List[List[np.ndarray]] = [[] for _ in range(len(prompts))]
    with responses_path.open("w", encoding="utf-8") as f_out:
        for img_path in tqdm(image_paths, desc="Images"):
            image = Image.open(img_path).convert("RGB")

            texts: List[str] = []
            feats: List[np.ndarray] = []
            for prompt_idx, prompt in enumerate(prompts):
                text = generate_text(
                    model, processor, image, prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    system_prompt=rad_system_prompt,
                )
                texts.append(text)
                all_texts_by_prompt[prompt_idx].append(text)

                feat, _dbg = extract_logits_feature(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    answer=text,
                    prompt_idx=prompt_idx,
                    feature_mode=args.feature_mode,
                    logit_temp=args.logit_temp,
                    system_prompt=rad_system_prompt,
                )
                feats.append(feat)
                all_feats_by_prompt[prompt_idx].append(feat)

                f_out.write(
                    json.dumps(
                        {
                            "image": img_path.name,
                            "prompt_idx": prompt_idx,
                            "prompt": prompt,
                            "system": rad_system_prompt,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            max_dim = max(f.shape[0] for f in feats)
            feats_pad = np.stack(
                [np.pad(f, (0, max_dim - f.shape[0])) for f in feats],
                axis=0,
            )

            cos_sim = safe_cosine_similarity_matrix(feats_pad)
            txt_sim = text_similarity_matrix(texts)

            base = output_dir / "per_sample" / img_path.stem
            write_matrix_csv(base.with_suffix(".feat_cosine.csv"), labels, cos_sim)
            write_matrix_csv(base.with_suffix(".text_sim.csv"), labels, txt_sim)

            resp_lines = []
            for prompt_idx, prompt in enumerate(prompts):
                resp_lines.append(f"[p{prompt_idx}] {prompt}")
                resp_lines.append(texts[prompt_idx])
                resp_lines.append("")
            (base.with_suffix(".responses.txt")).write_text("\n".join(resp_lines), encoding="utf-8")

    feats_np_by_prompt: List[np.ndarray] = []
    feat_dims: List[int] = []
    for pidx in range(len(prompts)):
        X = np.stack(all_feats_by_prompt[pidx], axis=0).astype(np.float32)
        feats_np_by_prompt.append(X)
        feat_dims.append(int(X.shape[1]))

    feats_np_by_prompt = postprocess_features(feats_np_by_prompt, center=args.center, l2norm=args.l2norm)

    summary = []
    for pidx in range(len(prompts)):
        X = feats_np_by_prompt[pidx]
        sim = safe_cosine_similarity_matrix(X)
        write_matrix_csv(output_dir / f"prompt_{pidx}_sample_feat_cosine.csv", sample_labels, sim)

        n = sim.shape[0]
        off = sim[~np.eye(n, dtype=bool)]
        l2 = np.linalg.norm(X, axis=1)
        summary.append(
            {
                "prompt_idx": pidx,
                "prompt_label": labels[pidx],
                "feature_dim": int(X.shape[1]),
                "mean_offdiag_cosine": float(off.mean()) if off.size else float("nan"),
                "std_offdiag_cosine": float(off.std()) if off.size else float("nan"),
                "min_offdiag_cosine": float(off.min()) if off.size else float("nan"),
                "max_offdiag_cosine": float(off.max()) if off.size else float("nan"),
                "l2_mean": float(l2.mean()),
                "l2_std": float(l2.std()),
                "l2_min": float(l2.min()),
                "l2_max": float(l2.max()),
            }
        )

    (output_dir / "summary_logits_feature.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Model: {args.model_name}")
    print(f"[OK] Saved responses to: {responses_path}")
    print(f"[OK] Saved per-sample matrices to: {output_dir / 'per_sample'}")
    print(f"[OK] Saved summary to: {output_dir / 'summary_logits_feature.json'}")


if __name__ == "__main__":
    main()
