from openai import OpenAI
from typing import List, Dict
import logging

class LLMManager:
    def __init__(self):
        self.client = OpenAI()

    def generate_response(self, 
                         query: str, 
                         relevant_chunks: List[Dict],
                         metadata: List[Dict]) -> str:
        """LLMを使用して回答を生成"""
        context = self._build_context(relevant_chunks, metadata)
        
        # コンテキストの詳細をログに記録
        logging.info("=== LLM Context Details ===")
        logging.info(f"Query: {query}")
        logging.info(f"Number of relevant chunks: {len(relevant_chunks)}")
        for i, chunk in enumerate(relevant_chunks, 1):
            logging.info(f"\nChunk {i}:")
            logging.info(f"Similarity: {chunk['similarity']:.3f}")
            logging.info(f"Content: {chunk['content'][:200]}...")  # 長すぎる場合は省略
        
        logging.info("\nMetadata:")
        for meta in metadata:
            logging.info(f"Title: {meta['title']}")
            logging.info(f"Version: {meta.get('version', 'N/A')}")
        
        logging.info("\nFull Context:")
        logging.info(context)
        logging.info("========================")
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": """
                You are a GDPR expert assistant. Provide accurate and helpful responses based on the provided guidelines.
                Answer in the same language as the user's query (English or Japanese).
                
                When answering:
                - When citing from guidelines, indicate the source immediately after the citation like:
                  "[From Guidelines on XXX: citation text]"
                - When referencing GDPR articles, cite them inline like:
                  "[GDPR Art. XX: relevant text]"
                - Prioritize information from more recent guidelines when there are conflicts
                - If providing information beyond what is covered in the guidelines, explicitly state this
                - Keep the same language as the query (English or Japanese) for the entire response
                """},
                {"role": "user", "content": f"""
                Question: {query}

                Relevant Context:
                {context}

                Please provide a comprehensive answer, citing sources inline when referencing guidelines or GDPR articles.
                Prioritize information from more recent guidelines when available.
                """}
            ]
        )
        
        return response.choices[0].message.content

    def _build_context(self, chunks: List[Dict], metadata: List[Dict]) -> str:
        """回答生成用のコンテキストを構築"""
        context = "Guidelines Information:\n"
        
        # メタデータの追加（Executive Summaryを除外し、採択日を追加）
        for meta in metadata:
            context += f"\nGuideline: {meta['title']}\n"
            context += f"Version: {meta['version']}\n"
            context += f"Adopted Date: {meta.get('adopted_date', 'N/A')}\n"
        
        # チャンクの追加
        context += "\nRelevant Sections:\n"
        for chunk in chunks:
            context += f"\n{chunk['content']}\n"
            
        return context 