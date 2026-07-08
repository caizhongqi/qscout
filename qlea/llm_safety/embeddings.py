"""Prompt embeddings for offline LLM safety-boundary experiments."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def embed_prompts(
    prompts: list[str],
    *,
    n_components: int = 12,
    seed: int = 7,
) -> np.ndarray:
    """Return dense normalized TF-IDF/SVD prompt features."""
    max_features = max(256, min(4096, len(prompts) * 16))
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )
    sparse = tfidf.fit_transform(prompts)
    dim = max(2, min(n_components, sparse.shape[0] - 1, sparse.shape[1] - 1))
    if dim < 2:
        return sparse.toarray().astype(np.float32)
    pipe = make_pipeline(
        TruncatedSVD(n_components=dim, random_state=seed),
        Normalizer(copy=False),
        StandardScaler(),
    )
    return pipe.fit_transform(sparse).astype(np.float32)
