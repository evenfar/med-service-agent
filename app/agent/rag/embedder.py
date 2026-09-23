"""向量化（第5期）：真实 embedding API + 离线确定性假实现。

HashEmbedder（离线/测试）：字符 bigram + 英文词 → MD5 稳定散列到固定维度。
要点：必须用 MD5 这类跨进程稳定的散列（Python 内置 hash 每个进程随机盐化，
索引文件换个进程就作废 —— 这是一个真实的踩坑点，写进了 docs 第5期）。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from abc import ABC, abstractmethod

from app.agent.tracer import Tracer
from app.config.settings import Settings
from app.llm.client import with_retry


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, settings: Settings, tracer: Tracer | None = None,
                 inner_client=None):
        self._s = settings
        self._tracer = tracer
        self._batch = 32
        if inner_client is not None:
            self._inner = inner_client
        else:
            import openai
            self._inner = openai.OpenAI(api_key=settings.openai_api_key,
                                        base_url=settings.openai_base_url)

    @property
    def model(self) -> str:
        return self._s.embedding_model

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i:i + self._batch]
            t0 = time.perf_counter()
            resp = with_retry(
                lambda b=batch: self._inner.embeddings.create(model=self.model, input=b))
            if self._tracer:
                self._tracer.log_llm(
                    purpose="embedding", model=self.model, prompt_tokens=sum(len(t) for t in batch) // 2,
                    completion_tokens=0, latency_ms=(time.perf_counter() - t0) * 1000)
            out.extend(d.embedding for d in resp.data)
        return out


def _tokens(text: str) -> list[str]:
    toks: list[str] = []
    for seg in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if seg.isascii():
            toks.append(seg)
        else:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1)) or toks.append(seg)
    return toks


class HashEmbedder(BaseEmbedder):
    """确定性假向量：语义无关但稳定可复现，离线测试/演示用。"""

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def model(self) -> str:
        return f"hash-{self._dim}"

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embedder(settings: Settings, tracer: Tracer | None = None) -> BaseEmbedder:
    from app.config.settings import is_offline
    if is_offline(settings):
        return HashEmbedder()
    return OpenAIEmbedder(settings, tracer)
