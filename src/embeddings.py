from openai import OpenAI
import os
from typing import List
import numpy as np

class EmbeddingGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_embedding(self, text: str) -> List[float]:
        """テキストの埋め込みベクトルを生成"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        # 最初の256次元を取得し、L2正規化を適用
        embedding = response.data[0].embedding[:256]
        normalized_embedding = self._normalize_l2(embedding)
        return normalized_embedding

    def _normalize_l2(self, x: List[float]) -> List[float]:
        """ベクトルのL2正規化を行う"""
        x = np.array(x)
        norm = np.linalg.norm(x)
        if norm == 0:
            return x.tolist()
        return (x / norm).tolist()