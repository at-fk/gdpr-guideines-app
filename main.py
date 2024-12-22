# main.py
import os
from dotenv import load_dotenv
from supabase import create_client
from src.processors.reset_and_rebuild import GuidelineRebuilder
import logging
from pathlib import Path

# ロギング設定
def setup_logging():
    log_dir = Path(os.getenv("LOG_DIRECTORY", "logs"))
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'update.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """メイン処理"""
    try:
        setup_logging()
        logging.info("Starting database update process")
        
        # 環境変数の読み込み
        load_dotenv()
        
        # Supabaseクライアントの初期化
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
        
        # PDFディレクトリのパスを環境変数から取得
        pdf_directory = Path(os.getenv("PDF_DIRECTORY", "data/guidelines"))
        pdf_directory.mkdir(parents=True, exist_ok=True)
        
        # リビルダーの初期化と実行
        rebuilder = GuidelineRebuilder(
            supabase_client=supabase,
            pdf_directory=pdf_directory
        )
        
        # テーブルのリセットと再構築
        success = rebuilder.rebuild_from_pdfs()
        
        if success:
            logging.info("Database update completed successfully")
        else:
            logging.error("Database update completed with errors")
            
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
