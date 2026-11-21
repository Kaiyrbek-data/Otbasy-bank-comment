"""Dense FAQ retrieval with the trained BGE-M3 residual projection."""

import json
import os
from pathlib import Path
from threading import Lock

os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


class ResidualProjection(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.scale = dimension ** 0.5
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        update = self.up(F.gelu(self.down(x * self.scale))) / self.scale
        return F.normalize(x + update, dim=1)


class FAQRetriever:
    def __init__(self, root=ROOT, threads=2):
        root = Path(root)
        self.config = json.loads((root / "model.json").read_text(encoding="utf-8"))
        self.docs = json.loads((root / "assets" / "documents.json").read_text(encoding="utf-8"))
        self.index = np.load(root / "assets" / "index.npy", mmap_mode="r", allow_pickle=False)
        dim, rows, rank = (self.config[k] for k in ("dimension", "rows", "rank"))
        if self.index.shape != (rows, dim) or self.index.dtype != np.float32 or len(self.docs) != rows:
            raise ValueError("Размер индекса не совпадает с настройками пакета.")
        self.projection = ResidualProjection(dim, rank).eval()
        with np.load(root / "assets" / "adapter.npz", allow_pickle=False) as weights:
            self.projection.load_state_dict({key: torch.from_numpy(weights[key]) for key in weights.files})
        torch.set_num_threads(threads)
        supported_engines = torch.backends.quantized.supported_engines
        if "fbgemm" in supported_engines:
            torch.backends.quantized.engine = "fbgemm"
        elif "qnnpack" in supported_engines:
            torch.backends.quantized.engine = "qnnpack"
        else:
            raise RuntimeError(
                "PyTorch не предоставляет поддерживаемый CPU-бэкенд квантования "
                "(ожидались FBGEMM или QNNPACK)."
            )
        self.encoder = SentenceTransformer(self.config["model"], revision=self.config["revision"], device="cpu")
        self.encoder.max_seq_length = self.config["max_tokens"]
        torch.ao.quantization.quantize_dynamic(self.encoder, {nn.Linear}, dtype=torch.qint8, inplace=True)
        self._lock = Lock()

    @torch.inference_mode()
    def search(self, question, top_k=3, dedupe=True):
        question = question.strip()
        if not question:
            raise ValueError("Введите вопрос.")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k должен быть от 1 до 10.")
        with self._lock:
            embedding = self.encoder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
            vector = self.projection(torch.from_numpy(embedding.astype(np.float32, copy=False))).numpy()[0]
        scores = self.index @ vector
        ranked = np.argsort(-scores, kind="stable")
        if dedupe:
            selected = []
            seen_answers = set()
            for index in ranked:
                answer = self.docs[index]["answer"]
                if answer in seen_answers:
                    continue
                seen_answers.add(answer)
                selected.append(index)
                if len(selected) == top_k:
                    break
        else:
            selected = ranked[:top_k]

        return [
            {
                **self.docs[index],
                "rank": rank,
                "cosine": float(np.clip(scores[index], -1.0, 1.0)),
            }
            for rank, index in enumerate(selected, start=1)
        ]
