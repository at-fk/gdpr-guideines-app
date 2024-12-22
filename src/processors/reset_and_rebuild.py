import os
from pathlib import Path
from typing import Dict
import PyPDF2
from datetime import datetime
from tqdm import tqdm
import time
import logging
from .guideline_processor import GuidelineProcessor
from openai import OpenAI
import asyncio
import json
import pytz
import re
import uuid

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rebuild.log'),
        logging.StreamHandler()
    ]
)

class GuidelineRebuilder:
    def __init__(self, 
                 supabase_client,
                 pdf_directory: str = "/path/to/pdfs"):
        self.supabase = supabase_client
        self.pdf_directory = Path(pdf_directory)
        self.processor = GuidelineProcessor(supabase_client)
        self.timezone = pytz.timezone('America/Los_Angeles')

    def get_all_pdf_files(self) -> list:
        """PDFファイルを収集"""
        return list(self.pdf_directory.rglob('*.pdf'))

    def process_single_pdf(self, pdf_path: str, guideline_id: int) -> bool:
        """単一のPDFを処理"""
        try:
            # メタデータ抽出
            metadata = self.processor.extract_metadata(pdf_path)
            if not metadata:
                metadata = self._create_default_metadata(pdf_path)

            # サマリー生成
            summary = self.processor.generate_summary(pdf_path)
            if summary:
                metadata['summary'] = summary

            # テキスト抽出
            text = self.processor.extract_text(pdf_path)
            if not text:
                return False

            # ガイドラインデータの作成
            guideline_data = {
                'id': guideline_id,
                **metadata,
                'content': text,
                'pdf_path': str(pdf_path),
                'embedding_status': 'pending'
            }

            # ガイドライン保存
            guideline = self.processor.save_guideline(guideline_data)
            if not guideline:
                return False

            # サマリー埋め込み保存
            if summary:
                embedding = self.processor.get_embedding(summary)
                if embedding:
                    self.processor.save_summary_embedding(guideline['id'], embedding)

            # チャンク処理
            chunks = self.processor.create_chunks(text)
            self.processor.save_chunks(guideline['id'], chunks)

            return True

        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return False

    def rebuild_from_pdfs(self) -> bool:
        """全PDFの再構築を実行"""
        try:
            pdf_files = self.get_all_pdf_files()
            logging.info(f"Found {len(pdf_files)} PDF files")

            for pdf_path in tqdm(pdf_files):
                logging.info(f"\nProcessing: {pdf_path}")
                # UUIDまたはハッシュ値を使用してユニークなIDを生成
                guideline_id = str(uuid.uuid4())
                if not self.process_single_pdf(str(pdf_path), guideline_id):
                    logging.warning(f"Failed to process {pdf_path}")
                time.sleep(1)  # レート制限考慮

            logging.info("\nRebuild process completed!")
            return True

        except Exception as e:
            logging.error(f"Error in rebuild process: {str(e)}")
            return False

    def _create_default_metadata(self, pdf_path: str) -> Dict:
        """デフォルトのメタデータを作成"""
        return {
            'title': Path(pdf_path).stem.replace('_', ' ').title(),
            'version': '1.0',
            'adopted_date': datetime.now(self.timezone).strftime('%Y-%m-%d'),
            'document_type': 'Guidelines',
            'summary': ''
        }