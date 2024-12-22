import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
import time

def generate_and_store_embeddings():
    # 1. 初期設定
    load_dotenv()
    
    # 2. Supabaseクライアントの初期化
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_ANON_KEY")
    )
    
    # 3. OpenAI Embeddingsモデルの初期化
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=True,
        max_retries=3,
    )
    
    # 4. すべてのガイドラインを取得
    response = supabase.table('guidelines')\
        .select('id', 'summary')\
        .execute()
    
    guidelines = response.data
    print(f"Found {len(guidelines)} guidelines to process")
    
    # 5. 各ガイドラインに対してエンベディングを生成
    for guideline in guidelines:
        try:
            if guideline['summary']:
                print(f"Processing guideline {guideline['id']}")
                
                # サマリー全体を1つのエンベディングとして処理
                embedding = embeddings.embed_query(guideline['summary'])
                
                # 生成したエンベディングをSupabaseに保存
                result = supabase.table('guidelines')\
                    .update({'summary_embedding': embedding})\
                    .eq('id', guideline['id'])\
                    .execute()
                
                print(f"Updated embedding for guideline {guideline['id']}")
                
                # APIレート制限に配慮して待機
                time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing guideline {guideline['id']}: {str(e)}")
            continue
    
    print("Embedding generation complete")

if __name__ == "__main__":
    generate_and_store_embeddings()