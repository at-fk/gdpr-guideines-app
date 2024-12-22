import streamlit as st
from dotenv import load_dotenv
import os
from src.database import SupabaseManager
from src.embeddings import EmbeddingGenerator
from src.llm import LLMManager
from typing import List, Dict
import logging
from pathlib import Path
import json
import csv
from io import StringIO

# 環境変数の読み込み
load_dotenv()

# ロギング設定
log_dir = Path(os.getenv("LOG_DIRECTORY", "logs"))
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler()
    ]
)

# セッション状態の初期化
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_managers():
    """マネージャークラスの初期化"""
    return (
        SupabaseManager(),
        EmbeddingGenerator(),
        LLMManager()
    )

def search_and_generate_response(
    query: str,
    db_manager: SupabaseManager,
    embedding_generator: EmbeddingGenerator,
    llm_manager: LLMManager,
    summary_threshold: float,
    chunk_threshold: float
) -> tuple[str, List[Dict], List[Dict]]:
    """検索と回答生成を実行"""
    try:
        # クエリの埋め込みを生成
        query_embedding = embedding_generator.get_embedding(query)

        # サマリー検索
        similar_summaries = db_manager.search_similar_summaries(
            query_embedding,
            threshold=summary_threshold
        )

        # チャンク検索（サマリー検索の結果を利用）
        similar_chunks = db_manager.search_similar_chunks(
            query_embedding,
            guideline_matches=similar_summaries,
            threshold=chunk_threshold
        )
        
        # 閾値を超えるチャンクのみをフィルタリング
        filtered_chunks = [chunk for chunk in similar_chunks if chunk['similarity'] >= chunk_threshold]

        # メタデータを取得
        metadata = []
        guideline_titles = {}
        for summary in similar_summaries:
            meta = db_manager.get_guideline_metadata(
                summary['id'],
                similarity=summary['similarity']
            )
            if meta:
                metadata.append(meta)
                guideline_titles[summary['id']] = meta['title']

        # フィルタリングされたチャンクを使用して回答を生成
        response = llm_manager.generate_response(query, filtered_chunks, metadata)

        return response, metadata, filtered_chunks, guideline_titles

    except Exception as e:
        logging.error(f"Error in search_and_generate_response: {str(e)}")
        raise e

def main():
    st.title("GDPR Guidelines Search")
    
    # マネージャーの初期化
    db_manager, embedding_generator, llm_manager = initialize_managers()
    
    # サイドバーの設定
    with st.sidebar:
        st.header("Search Settings")
        
        # 類似度の閾値設定
        summary_threshold = st.slider(
            "Summary Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum similarity score for guideline summaries"
        )
        
        chunk_threshold = st.slider(
            "Content Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum similarity score for content chunks"
        )
        
        # 設定を適用するボタン
        st.markdown("---")
        update_settings = st.button(
            "📝 Update Search Configuration",
            help="Apply these threshold settings to your next search"
        )

    # メインエリアの検索インターフェース
    query = st.text_input("🔍 Enter your question about GDPR guidelines")
    search_button = st.button("🔎 Search", disabled=not query)

    # 検索実行の条件
    should_search = (
        search_button and 
        query and 
        (update_settings or 'last_settings' in st.session_state)
    )

    # 設定が新された場合、セッショ��状態に保存
    if update_settings:
        st.session_state.last_settings = {
            'summary_threshold': summary_threshold,
            'chunk_threshold': chunk_threshold
        }
        st.success("Search configuration updated! You can now perform your search.")

    # 検索実行
    if should_search:
        try:
            with st.spinner('Searching and generating response...'):
                # セッション状態から最後の設定を使用
                settings = st.session_state.last_settings
                response, metadata, context_chunks, guideline_titles = search_and_generate_response(
                    query,
                    db_manager,
                    embedding_generator,
                    llm_manager,
                    settings['summary_threshold'],
                    settings['chunk_threshold']
                )

                # チャット履歴に追加
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response,
                    "metadata": metadata,
                    "chunks": context_chunks,
                    "guideline_titles": guideline_titles
                })

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return

    # チャット履歴の表示（逆順）
    for chat_index, item in enumerate(reversed(st.session_state.chat_history)):
        st.write("---")
        
        # エクスポートボタンを右寄せで配置
        col1, col2 = st.columns([6, 1])
        with col1:
            st.write("🤔 **Question:**")
        with col2:
            # このチャットをCSVとしてエクスポート
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['Type', 'Content'])
            writer.writerow(['Question', item["query"]])
            writer.writerow(['Answer', item["response"]])
            writer.writerow([''])  # 空行
            
            # 参照ガイドライン情報
            writer.writerow(['Referenced Guidelines:'])
            for summary in item["metadata"]:
                writer.writerow([
                    'Guideline',
                    f'{summary["title"]} (Version: {summary["version"]}, Adopted: {summary["adopted_date"]})'
                ])
            
            # 関連セクション
            writer.writerow([''])
            writer.writerow(['Relevant Sections:'])
            sorted_chunks = sorted(item["chunks"], key=lambda x: x['similarity'], reverse=True)
            for chunk in sorted_chunks[:5]:  # 上位5件
                if chunk['similarity'] >= chunk_threshold:
                    writer.writerow([
                        f'Similarity: {chunk["similarity"]:.2f}',
                        chunk['content'].replace('\n', ' ')  # 改行を空白に置換
                    ])
            
            csv_data = output.getvalue()
            st.download_button(
                label="📥",
                data=csv_data,
                file_name=f"chat_export_{chat_index}.csv",
                mime="text/csv",
                help="Export this Q&A as CSV"
            )

        st.write(item["query"])
        st.write("🤖 **Answer:**")
        st.write(item["response"])

        # 参考資料の表示
        st.write("📚 **Referenced Guidelines**")
        for summary in item["metadata"]:
            with st.expander(f"**{summary['title']}** (Similarity: {summary['similarity']:.2f})"):
                st.write(f"- Version: {summary['version']}")
                st.write(f"- Adopted: {summary['adopted_date']}")
                st.write(f"- Type: {summary['document_type']}")
                st.write("#### Summary")
                st.write(summary['summary'])

        # 関連チャンク（類似度でソート済み）
        st.write("📄 **Relevant Sections**")
        sorted_chunks = sorted(item["chunks"], key=lambda x: x['similarity'], reverse=True)
        guideline_titles = item.get("guideline_titles", {})

        # 上位5個のチャンクのみを表示
        for i, chunk in enumerate(sorted_chunks[:5], 1):
            if chunk['similarity'] < chunk_threshold:
                continue
            
            guideline_title = guideline_titles.get(chunk['guideline_id'], "Unknown Guideline")
            
            # メインコンテンツをランク付き���表示
            st.markdown(f"""
            <div style='
                border-left: 3px solid #e6e6e6;
                padding-left: 15px;
                margin-bottom: 5px;
            '>
                <div style='color: #666; margin-bottom: 10px;'>
                    <strong>#{i}</strong> | <strong>From: {guideline_title}</strong> (Similarity: {chunk['similarity']:.2f})
                </div>
                <p style='font-size: 1em; margin: 0 0 5px 0;'>{chunk['content']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # コンテキストボタンをメインコンテンツに視覚的に関連付ける
            st.markdown("""
            <div style='
                margin-left: 18px;
                margin-bottom: 20px;
                border-left: 3px solid #e6e6e6;
            '>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 Show Full Context"):
                context_chunks = chunk.get('context_chunks', [])
                if not context_chunks:
                    st.write("No additional context available")
                    continue
                
                # コンテキストチャンクを順序付けて処理
                sorted_contexts = sorted(context_chunks, key=lambda x: x['chunk_id'])
                combined_text = []
                
                for i, ctx in enumerate(sorted_contexts):
                    current_text = ctx['content']
                    
                    if ctx['chunk_id'] == chunk['id']:
                        # メインコンテンツはそのまま表示
                        combined_text.append(f"<p style='font-size: 1em; margin: 10px 0;'>{current_text}</p>")
                    else:
                        # 前のチャンクとのオーバーラップをチェック
                        if i > 0:
                            prev_text = sorted_contexts[i-1]['content']
                            overlap = find_overlap(prev_text, current_text)
                            if overlap:
                                current_text = current_text[len(overlap):]
                        
                        # 次のチャンクとのオーバーラップをチェック
                        if i < len(sorted_contexts) - 1:
                            next_text = sorted_contexts[i+1]['content']
                            overlap = find_overlap(current_text, next_text)
                            if overlap:
                                current_text = current_text[:-len(overlap)]
                        
                        combined_text.append(f"<p style='font-size: 1em; margin: 10px 0;'>{current_text}</p>")
                
                st.markdown("".join(combined_text), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def find_overlap(text1: str, text2: str, min_length: int = 50) -> str:
    """2つのテキスト間のオーバーラップを検出"""
    # text1の末尾とtext2の先頭で最長の共通部分を探す
    for length in range(min(len(text1), len(text2)), min_length - 1, -1):
        if text1[-length:] == text2[:length]:
            return text2[:length]
    return ""

if __name__ == "__main__":
    main()