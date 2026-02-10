import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List

# Import VLM modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.archs.components.vlm_conditioner import QwenVLMTextGenerator, TextToConditionVector

# --- Configurable ---
IMAGE_DIR = "data/OCTA500_3M/images"  # 이미지 폴더 경로
PROMPT = (
    "Return ONE short sentence describing factors that affect segmentation of the target in this image "
    "(contrast, occlusion, clutter, target size, artifacts). Output ONLY the sentence, one line. "
    "No lists, no JSON, no markdown."
)
COND_DIM = 256
HIDDEN_DIM = 256
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = "auto"

# --- Utility ---
def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    normed = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.dot(normed, normed.T)

# --- Main ---
def main():
    image_paths = sorted(list(Path(IMAGE_DIR).glob("*.png")))
    print(f"Found {len(image_paths)} images.")

    vlm = QwenVLMTextGenerator(model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE)
    vlm._load_model()
    embed_dim = vlm.get_text_embed_dim()
    text_to_cond = TextToConditionVector(embed_dim=embed_dim, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM)
    text_to_cond.to(DEVICE)

    texts: List[str] = []
    cond_vecs: List[np.ndarray] = []

    for img_path in tqdm(image_paths):
        image = Image.open(img_path).convert("RGB")
        # 1. 텍스트 응답
        text = vlm.generate_text(image, PROMPT)
        texts.append(text)
        # 2. 텍스트 임베딩
        e_text = vlm.embed_text(text).to(DEVICE)
        # 3. conditioning vector
        cond_vec = text_to_cond(e_text).detach().cpu().numpy().squeeze()
        cond_vecs.append(cond_vec)
        print(f"{img_path.name}: {text}")

    cond_vecs_np = np.stack(cond_vecs, axis=0)
    sim_matrix = cosine_similarity_matrix(cond_vecs_np)

    print("\nConditioning Vector Cosine Similarity Matrix:")
    print(sim_matrix)

    # 텍스트 응답 유사도 (간단히 문자열 유사도)
    from difflib import SequenceMatcher
    def text_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
    text_sim_matrix = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            text_sim_matrix[i, j] = text_similarity(texts[i], texts[j])
    print("\nText Response Similarity Matrix:")
    print(text_sim_matrix)

if __name__ == "__main__":
    main()
