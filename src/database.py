from supabase import create_client
import os
from typing import List, Dict, Optional
import numpy as np
import logging

class SupabaseManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        self.client = create_client(url, key)

    def search_similar_summaries(self, query_embedding: List[float], threshold: float = 0.4, limit: int = 5) -> List[Dict]:
        """類似度の高いサマリーを検索
        
        Args:
            query_embedding: 検索クエリの埋め込みベクトル
            threshold: 類似度の閾値（0.4がデフォルト）
            limit: 返却する結果の最大数
        
        Returns:
            類似度が閾値を超えるガイドライン情報のリスト。
            各要素には id, title, summary, adopted_date, similarity が含まれる
        """
        response = self.client.rpc(
            'match_guidelines',
            {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }
        ).execute()
        
        # レスポンスデータをそのまま返す（similarityは既に含まれている）
        return response.data if response.data else []

    def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        guideline_matches: List[Dict],  # match_guidelinesの結果を受け取る
        threshold: float = 0.0, 
        context_size: int = 1
    ) -> List[Dict]:
        """類似チャンクを検索"""
        try:
            # ガイドラインマッチの結果を適切な形式に変換
            guideline_matches_table = [
                {'id': match['id'], 'similarity': match['similarity']}
                for match in guideline_matches
            ]
            
            response = self.client.rpc(
                'match_chunks_with_context',
                {
                    'query_embedding': query_embedding,
                    'guideline_matches': guideline_matches_table,  # 新しいパラメータ
                    'chunk_similarity_threshold': threshold,
                    'context_size': context_size
                }
            ).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logging.error(f"Error in search_similar_chunks: {str(e)}")
            return []

    def get_context_chunks(self, chunk_id: str, query_embedding: List[float]) -> List[Dict]:
        """チャンクの前後のコンテキストを取得（類似度付き）"""
        response = self.client.rpc(
            'get_context_chunks',
            {
                'target_chunk_id': chunk_id,
                'query_embedding': query_embedding
            }
        ).execute()
        
        return response.data if response.data else []

    def get_guideline_metadata(self, guideline_id: str, similarity: float = None) -> Optional[Dict]:
        """ガイドラインのメタデータを取得"""
        response = self.client.table('guidelines')\
            .select('title, version, adopted_date, document_type, summary')\
            .eq('id', guideline_id)\
            .single()\
            .execute()
            
        if response.data:
            # similarityが提供された場合、それをメタデータに追加
            metadata = response.data
            if similarity is not None:
                metadata['similarity'] = similarity
            return metadata
        return None 