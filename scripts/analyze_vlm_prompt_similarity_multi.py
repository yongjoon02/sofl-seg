import autorootcwd
import argparse
import json
import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from difflib import SequenceMatcher
from tqdm import tqdm


# Import VLM modules without triggering package __init__ (avoids registry circular imports)
_ROOT = Path(__file__).resolve().parents[1]
_VLM_PATH = _ROOT / "src" / "archs" / "components" / "vlm_conditioner.py"
_spec = importlib.util.spec_from_file_location("vlm_conditioner", _VLM_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load VLM module from {_VLM_PATH}")
_vlm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vlm_module)
QwenVLMTextGenerator = _vlm_module.QwenVLMTextGenerator
TextToConditionVector = _vlm_module.TextToConditionVector


# -----------------------------
# Logits-feature design (A)
# -----------------------------
# Use single-token labels to avoid multi-token splitting issues.
LEVELS = ["A", "B", "C", "D", "E", "F", "G", "H"]  # A=very_low ... H=max
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
UNK_LABEL = "unknown"  # used only for p2 fallback and invalid parsing

DEFAULT_PROMPTS = [
    # (p0) Two 5-way categorical ratings
    (
        "Output EXACTLY one line:\n"
        "contrast=LEVEL;fragmentation=LEVEL;thinness=LEVEL\n"
        "Rules:\n"
        "- LEVEL MUST be exactly one of: A,B,C,D,E,F,G,H.\n"
        "- A=very_low, B=low, C=mid, D=mid_high, E=high, F=very_high, G=extreme, H=max.\n"
        "- Decide based ONLY on the angiogram image.\n"
        "- Output MUST contain NO spaces and NO extra words.\n"
        "If you violate the format, output again correctly."
    ),
    # (p1) Two 5-way categorical ratings
    (
        "Output EXACTLY one line:\n"
        "blur=LEVEL;motion=LEVEL;defocus=LEVEL\n"
        "Rules:\n"
        "- LEVEL MUST be exactly one of: A,B,C,D,E,F,G,H.\n"
        "- A=very_low, B=low, C=mid, D=mid_high, E=high, F=very_high, G=extreme, H=max.\n"
        "- Decide based ONLY on the angiogram image.\n"
        "- Output MUST contain NO spaces and NO extra words.\n"
        "If you violate the format, output again correctly."
    ),
    # (p2) 4-way dominant label
    (
        "Output EXACTLY one line:\n"
        "dominant=LABEL;secondary=LABEL;confidence=LEVEL\n"
        "Rules:\n"
        "- LABEL MUST be exactly one of: A,B,C,D,E,F,G,H.\n"
        "- Map labels to issues: A=contrast, B=fragmentation, C=clutter, D=blur, "
        "E=low_snr, F=illumination, G=saturation, H=motion.\n"
        "- LEVEL MUST be exactly one of: A,B,C,D,E,F,G,H (A=very_low ... H=max).\n"
        "- Choose the MOST dominant factor affecting vessel visibility for this image.\n"
        "- Output MUST contain NO spaces and NO extra words.\n"
        "If you violate the format, output again correctly."
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Design-A logits feature dispersion test for Qwen2.5-VL.\n"
            "Instead of hidden-state mean pooling, extract candidate-token logits/logprobs at answer-value positions.\n"
            "\n"
            "Fixes applied vs old script:\n"
            "- NEVER returns zero-vector fallback for p2; uses an explicit 'unknown' bucket.\n"
            "- Adds optional postprocess: center + L2 normalize (recommended for dispersion tests).\n"
            "- Aggregations ignore padded dims via masks, and handle all-zero rows safely.\n"
        )
    )
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with images.")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for images.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all).")

    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompt", action="append", help="Prompt string. Can be used multiple times.")
    prompt_group.add_argument("--prompts-file", type=str, help="Text file with one prompt per line.")

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--device-map", type=str, default="auto", help="auto|none (use 'none' for single GPU)")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")

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
    parser.add_argument(
        "--center-l2",
        action="store_true",
        help="Center features per prompt and L2-normalize per sample before cosine.",
    )
    parser.add_argument(
        "--corr-gamma-beta",
        action="store_true",
        help="Compute correlation between text cond_vec norms and gamma/beta stats from ckpt heads.",
    )

    # Postprocess (recommended for similarity/dispersion metrics)
    parser.add_argument("--center", action="store_true", help="Center features per prompt (subtract mean feature).")
    parser.add_argument("--l2norm", action="store_true", help="L2-normalize features per sample (per prompt).")

    # p2 robustness knobs
    parser.add_argument(
        "--p2-include-unknown",
        action="store_true",
        help="Include an explicit 'unknown' bucket for p2 feature (default: ON).",
    )
    parser.set_defaults(p2_include_unknown=True)

    parser.add_argument(
        "--p2-fallback",
        type=str,
        default="unknown",
        choices=["unknown", "uniform", "prior"],
        help=(
            "Fallback strategy when p2 parsing/alignment fails.\n"
            "- unknown: put all mass on UNK bucket (requires --p2-include-unknown).\n"
            "- uniform: uniform over LABELS (and UNK if included).\n"
            "- prior: empirical prior over p2 labels computed from parsed outputs; if unavailable, falls back to uniform."
        ),
    )

    parser.add_argument("--output-dir", type=str, default="outputs/vlm_logits_feature_dispersion")

    # You asked to hard-set this default.
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/home/yongjun/sofl-seg/experiments/medsegdiff_flow/xca/medsegdiff_flow_xca_20260206_194231/checkpoints/best.ckpt",
        help="(Kept for continuity) Existing FiLM checkpoint path. NOTE: logits-feature conditioning requires retraining to match ckpt heads.",
    )

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt:
        return [p.strip() for p in args.prompt if p.strip()]
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        return [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    return list(DEFAULT_PROMPTS)


def safe_cosine_similarity_matrix(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cosine similarity with safe handling of all-zero vectors.
    """
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


def _get_tokenizer(vlm: "QwenVLMTextGenerator"):
    vlm._load_model()
    proc = vlm._processor
    tok = getattr(proc, "tokenizer", proc)
    return tok


def _build_chat_text_fallback(prompt: str, answer: Optional[str] = None) -> str:
    # Fallback if processor.apply_chat_template is not available.
    if answer is None:
        return f"USER:\n{prompt}\nASSISTANT:\n"
    return f"USER:\n{prompt}\nASSISTANT:\n{answer}"


def _try_build_chat_text(vlm: "QwenVLMTextGenerator", prompt: str, answer: Optional[str]) -> str:
    """
    Try to build a Qwen-style chat template including image placeholder.
    If unavailable, fallback to a simple textual wrapper.
    """
    proc = vlm._processor
    apply_chat = getattr(proc, "apply_chat_template", None)
    if apply_chat is None:
        return _build_chat_text_fallback(prompt, answer)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is None:
        try:
            return apply_chat(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return apply_chat(messages, tokenize=False)
    else:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]} )
        try:
            return apply_chat(messages, tokenize=False, add_generation_prompt=False)
        except TypeError:
            return apply_chat(messages, tokenize=False)


def _tokenize_mm(
    vlm: "QwenVLMTextGenerator",
    image: Image.Image,
    text: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    proc = vlm._processor
    inputs = proc(text=[text], images=[image], return_tensors="pt", padding=True)
    model = vlm._model
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device
    inputs = {k: v.to(model_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return inputs


@torch.inference_mode()
def _forward_logits(vlm: "QwenVLMTextGenerator", inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Returns logits: (B, T, V)
    """
    model = vlm._model
    out = model(**inputs, return_dict=True)
    return out.logits


def _safe_single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Tokenizer produced empty ids for candidate '{s}'")
    if len(ids) != 1:
        # Keep candidates simple; we still proceed with first token.
        return ids[0]
    return ids[0]


def _extract_answer_value_positions(answer: str, prompt_idx: int) -> Dict[str, int]:
    """
    Parse the answer string and return char positions (index in string) of the VALUE token(s).
    """
    ans = answer.strip()

    if prompt_idx == 0:
        m = re.fullmatch(r"contrast=([A-H]);fragmentation=([A-H]);thinness=([A-H])", ans)
        if not m:
            return {}
        cpos = ans.find("contrast=") + len("contrast=")
        fpos = ans.find("fragmentation=") + len("fragmentation=")
        tpos = ans.find("thinness=") + len("thinness=")
        return {"contrast": cpos, "fragmentation": fpos, "thinness": tpos}

    if prompt_idx == 1:
        m = re.fullmatch(r"blur=([A-H]);motion=([A-H]);defocus=([A-H])", ans)
        if not m:
            return {}
        bpos = ans.find("blur=") + len("blur=")
        mpos = ans.find("motion=") + len("motion=")
        dpos = ans.find("defocus=") + len("defocus=")
        return {"blur": bpos, "motion": mpos, "defocus": dpos}

    if prompt_idx == 2:
        m = re.fullmatch(r"dominant=([A-H]);secondary=([A-H]);confidence=([A-H])", ans)
        if not m:
            return {}
        dpos = ans.find("dominant=") + len("dominant=")
        spos = ans.find("secondary=") + len("secondary=")
        cpos = ans.find("confidence=") + len("confidence=")
        return {"dominant": dpos, "secondary": spos, "confidence": cpos}

    return {}


def _find_token_index_by_charpos(tokenizer, text: str, char_pos: int) -> Optional[int]:
    """
    Tokenize `text` with offsets and return token index whose span covers char_pos.
    Requires a fast tokenizer that supports return_offsets_mapping.
    """
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
    logits_step: torch.Tensor,  # (V,)
    cand_token_ids: List[int],
    mode: str,
    temp: float,
) -> np.ndarray:
    """
    Produce a feature vector for one field: either logprob or raw logit over candidates.
    """
    cand_logits = logits_step[cand_token_ids].float()  # (K,)
    if temp is None or temp <= 0:
        temp = 1.0
    cand_logits = cand_logits / float(temp)
    if mode == "logit":
        return cand_logits.detach().cpu().numpy()
    logp = torch.log_softmax(cand_logits, dim=0)
    return logp.detach().cpu().numpy()


def _p2_fallback_feature(
    feature_mode: str,
    include_unknown: bool,
    fallback: str,
    prior_probs: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Returns (feat, debug) for p2 when parsing/alignment fails.
    feat is NEVER all-zero.
    """
    debug: Dict[str, str] = {"p2_fallback": fallback}

    # p2 now has 3 fields: dominant (LABELS), secondary (LABELS), confidence (LEVELS)
    K_dom = len(LABELS) + (1 if include_unknown else 0)
    K_sec = len(LABELS) + (1 if include_unknown else 0)
    K_conf = len(LEVELS)
    K = K_dom + K_sec + K_conf

    if fallback == "prior" and prior_probs is not None and prior_probs.shape[0] == K:
        probs = prior_probs.copy()
        debug["p2_fallback_used"] = "prior"
    else:
        if fallback == "prior":
            debug["p2_fallback_used"] = "uniform(prior_unavailable)"
        else:
            debug["p2_fallback_used"] = fallback

        if fallback == "unknown" and include_unknown:
            probs = np.zeros((K,), dtype=np.float32)
            # put UNK for dominant and secondary if enabled, otherwise uniform
            probs[len(LABELS)] = 1.0  # dominant UNK
            probs[len(LABELS) + (1 if include_unknown else 0) + len(LABELS)] = 1.0  # secondary UNK
        else:
            probs = np.ones((K,), dtype=np.float32) / float(K)

    if feature_mode == "logit":
        # For logit mode, return logits that correspond to the categorical distribution.
        # Use log-prob as "pseudo-logit" (up to constant), which keeps scale sane.
        feat = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)
    else:
        feat = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)  # already logprob-like

    return feat, debug


def _load_text_to_cond_from_ckpt(state: Dict[str, torch.Tensor]) -> Optional[torch.nn.Module]:
    # Find a prefix that contains text_to_cond params
    key_candidates = [k for k in state.keys() if "text_to_cond.mlp.0.weight" in k]
    if not key_candidates:
        return None
    key0 = key_candidates[0]
    prefix = key0.replace("text_to_cond.mlp.0.weight", "text_to_cond")

    w0 = state.get(f"{prefix}.mlp.0.weight")
    w2 = state.get(f"{prefix}.mlp.2.weight")
    if w0 is None or w2 is None:
        return None
    hidden_dim = int(w0.shape[0])
    embed_dim = int(w0.shape[1])
    cond_dim = int(w2.shape[0])

    t2c = TextToConditionVector(embed_dim=embed_dim, cond_dim=cond_dim, hidden_dim=hidden_dim)
    sub = {k.replace(f"{prefix}.", ""): v for k, v in state.items() if k.startswith(prefix)}
    t2c.load_state_dict(sub, strict=True)
    t2c.eval()
    return t2c


def _load_film_heads_from_ckpt(state: Dict[str, torch.Tensor]) -> List[torch.nn.Module]:
    head_keys = [k for k in state.keys() if k.startswith("vlm_film_heads.")]
    if not head_keys:
        return []
    head_indices = sorted({int(k.split(".")[1]) for k in head_keys})
    heads: List[torch.nn.Module] = []
    for idx in head_indices:
        w0 = state.get(f"vlm_film_heads.{idx}.mlp.0.weight")
        w2 = state.get(f"vlm_film_heads.{idx}.mlp.2.weight")
        if w0 is None or w2 is None:
            continue
        hidden_dim = int(w0.shape[0])
        cond_dim = int(w0.shape[1])
        channels = int(w2.shape[0] // 2)
        head = _vlm_module.AdaptiveFiLMHead(
            cond_dim=cond_dim,
            channels=channels,
            hidden_dim=hidden_dim,
            gamma_scale=0.1,
            beta_scale=0.1,
            use_layernorm=True,
        )
        head_state = {
            k.replace(f"vlm_film_heads.{idx}.", ""): v
            for k, v in state.items()
            if k.startswith(f"vlm_film_heads.{idx}.")
        }
        head.load_state_dict(head_state, strict=True)
        head.eval()
        heads.append(head)
    return heads


@torch.inference_mode()
def extract_logits_feature(
    vlm: "QwenVLMTextGenerator",
    image: Image.Image,
    prompt: str,
    answer: str,
    prompt_idx: int,
    feature_mode: str = "logprob",
    logit_temp: float = 1.0,
    p2_include_unknown: bool = True,
    p2_fallback: str = "unknown",
    p2_prior_probs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Teacher-forcing:
    - Build prefix (image+prompt with assistant generation prompt) and full (prefix + answer)
    - Forward full to get logits
    - Find answer-token index for each field-value position, then take logits at previous step
    - Return concatenated candidate logits/logprobs feature.

    Key fix:
    - For p2, parsing/alignment failure NEVER yields zero-vector.
      Instead returns a valid categorical fallback over (LABELS [+ UNK]).
    """
    tok = _get_tokenizer(vlm)

    prefix_text = _try_build_chat_text(vlm, prompt, answer=None)
    full_text = _try_build_chat_text(vlm, prompt, answer=answer)

    device = vlm.device if hasattr(vlm, "device") else torch.device("cpu")
    prefix_inputs = _tokenize_mm(vlm, image, prefix_text, device=device)
    full_inputs = _tokenize_mm(vlm, image, full_text, device=device)

    prefix_ids = prefix_inputs.get("input_ids")
    full_ids = full_inputs.get("input_ids")
    if prefix_ids is None or full_ids is None:
        raise RuntimeError("Processor did not return input_ids. Check Qwen2.5-VL processor outputs.")

    prefix_len = int(prefix_ids.shape[1])
    full_len = int(full_ids.shape[1])

    logits = _forward_logits(vlm, full_inputs)  # (1, T, V)

    positions = _extract_answer_value_positions(answer, prompt_idx)
    debug: Dict[str, str] = {}

    # Candidate sets per field
    cand_ids_map: Dict[str, List[int]] = {}
    if prompt_idx in (0, 1):
        cand_ids_map = {field: [_safe_single_token_id(tok, s) for s in LEVELS] for field in positions.keys()} if positions else {}
    else:
        # p2: dominant/secondary -> LABELS (+UNK), confidence -> LEVELS
        label_cands = list(LABELS) + ([UNK_LABEL] if p2_include_unknown else [])
        label_ids = [_safe_single_token_id(tok, s) for s in label_cands]
        level_ids = [_safe_single_token_id(tok, s) for s in LEVELS]
        cand_ids_map = {
            "dominant": label_ids,
            "secondary": label_ids,
            "confidence": level_ids,
        }

    # If parsing failed, only p2 can fallback robustly
    if not positions:
        debug["parse_error"] = "answer_format_mismatch"
        if prompt_idx == 2:
            feat, dbg2 = _p2_fallback_feature(
                feature_mode=feature_mode,
                include_unknown=p2_include_unknown,
                fallback=p2_fallback,
                prior_probs=p2_prior_probs,
            )
            debug.update(dbg2)
            debug["prefix_len"] = str(prefix_len)
            debug["full_len"] = str(full_len)
            debug["feature_dim"] = str(int(feat.shape[0]))
            return feat, debug
        else:
            # For p0/p1, return a valid-length uniform feature (avoid shape mismatch)
            # p0/p1 each have 3 fields, each with len(LEVELS) candidates.
            K = 3 * len(LEVELS)
            probs = np.ones((K,), dtype=np.float32) / float(K)
            feat = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32) if feature_mode != "logit" else np.log(
                np.clip(probs, 1e-12, 1.0)
            ).astype(np.float32)
            debug["non_p2_uniform_fallback"] = "1"
            debug["prefix_len"] = str(prefix_len)
            debug["full_len"] = str(full_len)
            debug["feature_dim"] = str(int(feat.shape[0]))
            return feat, debug

    # Tokenize answer alone with offsets to map charpos -> token index (within answer)
    feat_parts: List[np.ndarray] = []

    for field, char_pos in positions.items():
        t_idx_ans = _find_token_index_by_charpos(tok, answer, char_pos)
        if t_idx_ans is None:
            debug[f"{field}_align_error"] = "no_offsets_mapping_or_charpos_not_found"
            if prompt_idx == 2:
                feat, dbg2 = _p2_fallback_feature(
                    feature_mode=feature_mode,
                    include_unknown=p2_include_unknown,
                    fallback=p2_fallback,
                    prior_probs=p2_prior_probs,
                )
                debug.update(dbg2)
                debug["prefix_len"] = str(prefix_len)
                debug["full_len"] = str(full_len)
                debug["feature_dim"] = str(int(feat.shape[0]))
                return feat, debug
            else:
                # p0/p1 uniform fallback
                K = len(cand_ids)
                probs = np.ones((K,), dtype=np.float32) / float(K)
                feat = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32) if feature_mode != "logit" else np.log(
                    np.clip(probs, 1e-12, 1.0)
                ).astype(np.float32)
                debug["non_p2_uniform_fallback"] = "1"
                debug["prefix_len"] = str(prefix_len)
                debug["full_len"] = str(full_len)
                debug["feature_dim"] = str(int(feat.shape[0]))
                return feat, debug

        t_idx_full = prefix_len + t_idx_ans
        step = t_idx_full - 1
        if step < 0 or step >= full_len:
            debug[f"{field}_align_error"] = f"step_out_of_range(step={step}, full_len={full_len})"
            if prompt_idx == 2:
                feat, dbg2 = _p2_fallback_feature(
                    feature_mode=feature_mode,
                    include_unknown=p2_include_unknown,
                    fallback=p2_fallback,
                    prior_probs=p2_prior_probs,
                )
                debug.update(dbg2)
                debug["prefix_len"] = str(prefix_len)
                debug["full_len"] = str(full_len)
                debug["feature_dim"] = str(int(feat.shape[0]))
                return feat, debug
            else:
                K = len(cand_ids)
                probs = np.ones((K,), dtype=np.float32) / float(K)
                feat = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32) if feature_mode != "logit" else np.log(
                    np.clip(probs, 1e-12, 1.0)
                ).astype(np.float32)
                debug["non_p2_uniform_fallback"] = "1"
                debug["prefix_len"] = str(prefix_len)
                debug["full_len"] = str(full_len)
                debug["feature_dim"] = str(int(feat.shape[0]))
                return feat, debug

        logits_step = logits[0, step, :]  # (V,)
        cand_ids = cand_ids_map.get(field, [])
        if not cand_ids:
            debug[f"{field}_cand_error"] = "empty_candidates"
            return _p2_fallback_feature(
                feature_mode=feature_mode,
                include_unknown=p2_include_unknown,
                fallback=p2_fallback,
                prior_probs=p2_prior_probs,
            )
        f = _candidate_feature_for_field(logits_step, cand_ids, mode=feature_mode, temp=logit_temp)
        feat_parts.append(f)

        debug[f"{field}_token_index_full"] = str(t_idx_full)
        debug[f"{field}_token_index_answer"] = str(t_idx_ans)

    feat = np.concatenate(feat_parts, axis=0).astype(np.float32)
    debug["prefix_len"] = str(prefix_len)
    debug["full_len"] = str(full_len)
    debug["feature_dim"] = str(int(feat.shape[0]))
    return feat, debug


def pad_to_max_dim(vec: np.ndarray, max_dim: int) -> np.ndarray:
    if vec.shape[0] == max_dim:
        return vec
    out = np.zeros((max_dim,), dtype=vec.dtype)
    out[: vec.shape[0]] = vec
    return out


def postprocess_features(
    feats_by_prompt: List[np.ndarray],
    center: bool,
    l2norm: bool,
) -> List[np.ndarray]:
    """
    feats_by_prompt: list of shape (N, d_p) arrays (variable d across prompts)
    Applies per-prompt centering (subtract mean over samples) and per-sample L2 norm.
    """
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _contrast_snr_proxy(gray: np.ndarray) -> Tuple[float, float]:
    p5, p95 = np.percentile(gray, [5, 95])
    contrast_range = float(p95 - p5)
    std = float(gray.std()) + 1e-6
    snr_proxy = float(gray.mean() / std)
    return contrast_range, snr_proxy


def _blur_score(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    out = np.zeros_like(gray, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            patch = pad[i:i + 3, j:j + 3]
            out[i, j] = float((patch * k).sum())
    return float(out.var())


def main() -> None:
    args = parse_args()
    if getattr(args, "center_l2", False):
        args.center = True
        args.l2norm = True
    device = resolve_device(args.device)

    prompts = load_prompts(args)
    if len(prompts) < 2:
        raise ValueError("Provide at least two prompts for similarity analysis.")

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob(args.pattern))
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir} with pattern {args.pattern}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_sample").mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.txt").write_text("\n".join(prompts), encoding="utf-8")

    device_map = None if args.device_map == "none" else args.device_map

    vlm = QwenVLMTextGenerator(
        model_name=args.model_name,
        device=device,
        dtype=args.dtype,
        device_map=device_map,
    )
    vlm._load_model()

    labels = [f"p{i}" for i in range(len(prompts))]
    sample_labels = [p.stem for p in image_paths]

    # Variable-dim features per prompt
    all_feats_by_prompt: List[List[np.ndarray]] = [[] for _ in range(len(prompts))]
    all_texts_by_prompt: List[List[str]] = [[] for _ in range(len(prompts))]

    responses_path = output_dir / "responses.jsonl"
    feat_debug_path = output_dir / "feature_debug.jsonl"

    # For p2 prior fallback, we collect empirical label frequencies from parsed p2 texts (best-effort).
    p2_label_counts = {k: 0 for k in LABELS}
    p2_total_parsed = 0

    with responses_path.open("w", encoding="utf-8") as f_out, feat_debug_path.open("w", encoding="utf-8") as f_dbg:
        for img_path in tqdm(image_paths, desc="Images"):
            image = Image.open(img_path).convert("RGB")
            texts: List[str] = []
            feats: List[np.ndarray] = []

            for prompt_idx, prompt in enumerate(prompts):
                # 1) generate short answer
                text = vlm.generate_text(image, prompt).strip()
                texts.append(text)
                all_texts_by_prompt[prompt_idx].append(text)

                # Track p2 empirical prior from parsed outputs
                if prompt_idx == 2:
                    m = re.fullmatch(r"dominant=([A-H]);secondary=([A-H]);confidence=([A-H])", text.strip())
                    if m:
                        p2_label_counts[m.group(1)] += 1
                        p2_total_parsed += 1

                # Build p2 prior probs if requested and available
                p2_prior_probs = None
                if args.p2_fallback == "prior":
                    K = len(LABELS) + (1 if args.p2_include_unknown else 0)
                    if p2_total_parsed > 0:
                        probs = np.zeros((K,), dtype=np.float32)
                        for i, lab in enumerate(LABELS):
                            probs[i] = p2_label_counts[lab] / float(p2_total_parsed)
                        if args.p2_include_unknown:
                            probs[-1] = 0.0
                        # renorm safely
                        s = probs.sum()
                        if s > 0:
                            probs = probs / s
                        else:
                            probs[:] = 1.0 / float(K)
                        p2_prior_probs = probs

                # 2) teacher-forcing logits feature extraction (Design A with fixes)
                feat, dbg = extract_logits_feature(
                    vlm=vlm,
                    image=image,
                    prompt=prompt,
                    answer=text,
                    prompt_idx=prompt_idx,
                    feature_mode=args.feature_mode,
                    logit_temp=args.logit_temp,
                    p2_include_unknown=args.p2_include_unknown,
                    p2_fallback=args.p2_fallback,
                    p2_prior_probs=p2_prior_probs,
                )

                feats.append(feat)
                all_feats_by_prompt[prompt_idx].append(feat)

                # log records
                f_out.write(
                    json.dumps(
                        {
                            "image": img_path.name,
                            "prompt_idx": prompt_idx,
                            "prompt": prompt,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_dbg.write(
                    json.dumps(
                        {
                            "image": img_path.name,
                            "prompt_idx": prompt_idx,
                            "text": text,
                            "debug": dbg,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Per-sample prompt-prompt cosine: pad to max dim for this sample
            max_dim = max(f.shape[0] for f in feats)
            feats_pad = np.stack([pad_to_max_dim(f, max_dim) for f in feats], axis=0)  # (P, max_dim)

            # Optional postprocess per-sample (center across prompts doesn't make sense; we keep raw here)
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

    # Convert to arrays per prompt and apply optional postprocessing
    feats_np_by_prompt: List[np.ndarray] = []
    feat_dims: List[int] = []
    for pidx in range(len(prompts)):
        X = np.stack(all_feats_by_prompt[pidx], axis=0).astype(np.float32)  # (N, d_p)
        feats_np_by_prompt.append(X)
        feat_dims.append(int(X.shape[1]))

    feats_np_by_prompt = postprocess_features(feats_np_by_prompt, center=args.center, l2norm=args.l2norm)

    # Save feature dims and settings
    (output_dir / "feature_dims.json").write_text(json.dumps(feat_dims, indent=2), encoding="utf-8")
    (output_dir / "postprocess.json").write_text(
        json.dumps({"center": bool(args.center), "l2norm": bool(args.l2norm)}, indent=2),
        encoding="utf-8",
    )

    # Padded: (N, P, Dmax_all)
    Dmax_all = int(max(feat_dims)) if feat_dims else 0
    feats_padded = np.zeros((len(image_paths), len(prompts), Dmax_all), dtype=np.float32)
    feats_mask = np.zeros((len(image_paths), len(prompts), Dmax_all), dtype=np.float32)

    for pidx in range(len(prompts)):
        X = feats_np_by_prompt[pidx]  # (N, d)
        d = X.shape[1]
        feats_padded[:, pidx, :d] = X
        feats_mask[:, pidx, :d] = 1.0

    np.save(output_dir / "logits_features_padded.npy", feats_padded)
    np.save(output_dir / "logits_features_mask.npy", feats_mask)

    # Aggregate prompt-prompt cosine similarity across samples (on padded features)
    # (We still compute cosine on full padded vector; safe_cosine handles norms.
    #  Since each prompt has its own d, the extra zeros are consistent across samples.)
    agg = np.zeros((len(prompts), len(prompts)), dtype=np.float32)
    for sidx in range(feats_padded.shape[0]):
        agg += safe_cosine_similarity_matrix(feats_padded[sidx])
    agg /= float(feats_padded.shape[0])
    write_matrix_csv(output_dir / "prompt_feat_cosine_mean.csv", labels, agg)

    # Per-prompt: sample-sample cosine similarity (feature dispersion)
    for pidx in range(len(prompts)):
        d = feat_dims[pidx]
        vecs = feats_padded[:, pidx, :d]  # (N, d)
        sim = safe_cosine_similarity_matrix(vecs)
        write_matrix_csv(output_dir / f"prompt_{pidx}_sample_feat_cosine.csv", sample_labels, sim)

    # Summary stats per prompt: off-diagonal cosine distribution + L2 norm stats
    summary = []
    for pidx in range(len(prompts)):
        d = feat_dims[pidx]
        vecs = feats_padded[:, pidx, :d]
        sim = safe_cosine_similarity_matrix(vecs)
        n = sim.shape[0]
        off = sim[~np.eye(n, dtype=bool)]

        l2 = np.linalg.norm(vecs, axis=1)
        summary.append(
            {
                "prompt_idx": pidx,
                "prompt_label": labels[pidx],
                "feature_dim": int(d),
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

    # Feature ↔ quality metrics correlation (p0/p1 fields vs contrast/blur proxies)
    contrast_range = []
    snr_proxy = []
    blur_var = []
    for img_path in image_paths:
        im = Image.open(img_path).convert("L")
        gray = np.asarray(im, dtype=np.float32) / 255.0
        c_range, snr = _contrast_snr_proxy(gray)
        contrast_range.append(c_range)
        snr_proxy.append(snr)
        blur_var.append(_blur_score(gray))
    contrast_range = np.array(contrast_range, dtype=np.float32)
    snr_proxy = np.array(snr_proxy, dtype=np.float32)
    blur_var = np.array(blur_var, dtype=np.float32)

    def _expected_level(field_vec: np.ndarray) -> np.ndarray:
        if field_vec.shape[1] == 0:
            return np.zeros((field_vec.shape[0],), dtype=np.float32)
        ex = np.exp(field_vec - field_vec.max(axis=1, keepdims=True))
        probs = ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)
        levels = np.arange(1, field_vec.shape[1] + 1, dtype=np.float32)
        return (probs * levels[None, :]).sum(axis=1)

    p0_raw = np.stack(all_feats_by_prompt[0], axis=0) if len(all_feats_by_prompt) > 0 else None
    p1_raw = np.stack(all_feats_by_prompt[1], axis=0) if len(all_feats_by_prompt) > 1 else None
    corr_out = []
    if p0_raw is not None and p0_raw.shape[1] >= 24:
        p0_contrast = _expected_level(p0_raw[:, 0:8])
        corr_out.append(
            {
                "prompt": "p0",
                "field": "contrast",
                "corr_pearson_range": _pearson(p0_contrast, contrast_range),
                "corr_spearman_range": _spearman(p0_contrast, contrast_range),
                "corr_pearson_snr": _pearson(p0_contrast, snr_proxy),
                "corr_spearman_snr": _spearman(p0_contrast, snr_proxy),
            }
        )
    if p1_raw is not None and p1_raw.shape[1] >= 24:
        p1_blur = _expected_level(p1_raw[:, 0:8])
        corr_out.append(
            {
                "prompt": "p1",
                "field": "blur",
                "corr_pearson_blurvar": _pearson(p1_blur, blur_var),
                "corr_spearman_blurvar": _spearman(p1_blur, blur_var),
            }
        )
    (output_dir / "feature_quality_corr.json").write_text(
        json.dumps(corr_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Optional: correlation between text-cond norms and gamma/beta stats (from ckpt heads)
    if args.corr_gamma_beta and args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        t2c = _load_text_to_cond_from_ckpt(state)
        heads = _load_film_heads_from_ckpt(state)

        if t2c is None or not heads:
            print("[WARN] text_to_cond or vlm_film_heads not found in ckpt; skip corr.")
        else:
            corr_out = []
            # compute per-prompt cond_vec norms from generated texts
            for pidx in range(len(prompts)):
                texts = all_texts_by_prompt[pidx]
                cond_vecs = []
                for text in texts:
                    e_text = embed_text_tokens_only(vlm, text, pool="mean").float().cpu()
                    with torch.no_grad():
                        cond = t2c(e_text).detach().cpu().float().squeeze(0)
                    cond_vecs.append(cond.numpy())
                cond_vecs = np.stack(cond_vecs, axis=0)  # (N, cond_dim)
                cond_norm = np.linalg.norm(cond_vecs, axis=1)

                for hidx, head in enumerate(heads):
                    with torch.no_grad():
                        gamma, beta = head(torch.from_numpy(cond_vecs).float())
                        gamma = gamma.detach().cpu().float()
                        beta = beta.detach().cpu().float()
                    gamma_dev = (gamma - 1.0).abs().mean(dim=(1, 2, 3)).numpy()
                    beta_norm = beta.view(beta.size(0), -1).norm(dim=1).numpy()

                    # Pearson correlations
                    def _corr(a, b):
                        if a.std() < 1e-8 or b.std() < 1e-8:
                            return float("nan")
                        return float(np.corrcoef(a, b)[0, 1])

                    corr_out.append(
                        {
                            "prompt_idx": pidx,
                            "head_idx": hidx,
                            "cond_norm_vs_gamma_dev": _corr(cond_norm, gamma_dev),
                            "cond_norm_vs_beta_norm": _corr(cond_norm, beta_norm),
                        }
                    )
            (output_dir / "cond_gamma_beta_corr.json").write_text(
                json.dumps(corr_out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[OK] Saved corr to: {output_dir / 'cond_gamma_beta_corr.json'}")

    # ckpt-path note (you asked to set it; for now we just log the path and warn)
    (output_dir / "ckpt_note.txt").write_text(
        "NOTE: --ckpt-path is set for continuity, but logits-feature conditioning is a different conditioner.\n"
        "To get meaningful gamma/beta from this logits-feature, you need to retrain (or at least learn a new conditioner MLP) with these features.\n"
        f"ckpt_path={args.ckpt_path}\n",
        encoding="utf-8",
    )

    # Save p2 prior stats
    (output_dir / "p2_prior_stats.json").write_text(
        json.dumps(
            {
                "p2_total_parsed": int(p2_total_parsed),
                "p2_label_counts": p2_label_counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Saved responses to: {responses_path}")
    print(f"[OK] Saved feature debug to: {feat_debug_path}")
    print(f"[OK] Saved per-sample matrices to: {output_dir / 'per_sample'}")
    print(f"[OK] Saved padded logits-feature tensor to: {output_dir / 'logits_features_padded.npy'}")
    print(f"[OK] Saved summary to: {output_dir / 'summary_logits_feature.json'}")
    print(f"[OK] p2 prior stats: {output_dir / 'p2_prior_stats.json'}")
    print(f"[WARN] ckpt-path is recorded but not applied to logits-feature FiLM without retraining: {args.ckpt_path}")


if __name__ == "__main__":
    main()
