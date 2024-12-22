import streamlit as st
from dotenv import load_dotenv
import os
from src.database import SupabaseManager
from src.embeddings import EmbeddingGenerator
from src.llm import LLMManager
from typing import List, Dict
import logging
from pathlib import Path

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

# 言語設定用の辞書
TRANSLATIONS = {
    "en": {
        "title": "GDPR Guidelines Search",
        "search_settings": "Search Settings",
        "summary_threshold_label": "Summary Similarity Threshold",
        "summary_threshold_help": "Minimum similarity score for guideline summaries",
        "chunk_threshold_label": "Content Similarity Threshold",
        "chunk_threshold_help": "Minimum similarity score for content chunks",
        "recent_searches": "Recent Searches",
        "search_placeholder": "Enter your question about GDPR guidelines",
        "search_button": "Search",
        "searching": "Searching and generating response...",
        "question_label": "Question:",
        "answer_label": "Answer:",
        "referenced_guidelines": "Referenced Guidelines",
        "relevant_sections": "Relevant Sections",
        "version": "Version",
        "adopted": "Adopted",
        "type": "Type",
        "summary": "Summary",
        "show_context": "Show Full Context",
        "no_context": "No additional context available",
        "error": "An error occurred"
    },
    "ja": {
        "title": "GDPR ガイドライン検索",
        "search_settings": "検索設定",
        "summary_threshold_label": "サマリー類似度閾値",
        "summary_threshold_help": "ガイドラインサマリーの最小類似度スコア",
        "chunk_threshold_label": "コンテンツ類似度閾値",
        "chunk_threshold_help": "コンテンツチャンクの最小類似度スコア",
        "recent_searches": "最近の検索",
        "search_placeholder": "GDPRガイドラインについての質問を入力してください",
        "search_button": "検索",
        "searching": "検索中・回答生成��...",
        "question_label": "質問:",
        "answer_label": "回答:",
        "referenced_guidelines": "参照ガイドライン",
        "relevant_sections": "関連セクション",
        "version": "バージョン",
        "adopted": "採択日",
        "type": "種類",
        "summary": "概要",
        "show_context": "全文脈を表示",
        "no_context": "追加のコンテキストはありません",
        "error": "エラーが発生しました"
    }
}

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
) -> tuple[str, List[Dict], List[Dict], Dict]:
    """検索と回答生成を実行"""
    try:
        # クエリの埋め込みを生成
        query_embedding = embedding_generator.get_embedding(query)

        # サマリー検索
        similar_summaries = db_manager.search_similar_summaries(
            query_embedding,
            threshold=summary_threshold
        )

        # ��ャンク検索（サマリー検索の結果を利用）
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
    # 言語選択（セッション状態の初期化）
    if 'language' not in st.session_state:
        st.session_state.language = "en"
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # 現在の言語の翻訳を取得
    t = TRANSLATIONS[st.session_state.language]
    
    st.title(t["title"])
    
    # マネージャーの初期化
    db_manager, embedding_generator, llm_manager = initialize_managers()
    
    # サイドバーの設定
    with st.sidebar:
        # 言語選択
        st.selectbox(
            "Language / 言語",
            ["English", "日本語"],
            index=0 if st.session_state.language == "en" else 1,
            key="language_selector",
            on_change=lambda: setattr(
                st.session_state, 
                'language', 
                "en" if st.session_state.language_selector == "English" else "ja"
            )
        )
        
        st.header(t["search_settings"])
        
        summary_threshold = st.slider(
            t["summary_threshold_label"],
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help=t["summary_threshold_help"]
        )
        
        chunk_threshold = st.slider(
            t["chunk_threshold_label"],
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help=t["chunk_threshold_help"]
        )
        
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown(f"**{t['recent_searches']}:**")
            for q in reversed(st.session_state.query_history):
                st.markdown(f"- {q}")

    query = st.text_area(
        "🔍 " + t["search_placeholder"],
        key="search_query",
        height=50
    )
    search_button = st.button("🔎 " + t["search_button"], disabled=not query)

    if search_button and query:
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
            if len(st.session_state.query_history) > 5:
                st.session_state.query_history.pop(0)
        
        try:
            with st.spinner(t["searching"]):
                response, metadata, context_chunks, guideline_titles = search_and_generate_response(
                    query,
                    db_manager,
                    embedding_generator,
                    llm_manager,
                    summary_threshold,
                    chunk_threshold
                )

                st.write("---")
                st.write(f"🤔 **{t['question_label']}**")
                st.write(query)
                st.write(f"🤖 **{t['answer_label']}**")
                st.write(response)

                st.write(f"📚 **{t['referenced_guidelines']}**")
                for summary in metadata:
                    with st.expander(f"**{summary['title']}** (Similarity: {summary['similarity']:.2f})"):
                        st.write(f"- {t['version']}: {summary['version']}")
                        st.write(f"- {t['adopted']}: {summary['adopted_date']}")
                        st.write(f"- {t['type']}: {summary['document_type']}")
                        st.write(f"#### {t['summary']}")
                        st.write(summary['summary'])

                st.write(f"📄 **{t['relevant_sections']}**")
                sorted_chunks = sorted(context_chunks, key=lambda x: x['similarity'], reverse=True)

                # 上位5個のチャンクのみを表示
                for i, chunk in enumerate(sorted_chunks[:5], 1):
                    if chunk['similarity'] < chunk_threshold:
                        continue
                    
                    guideline_title = guideline_titles.get(chunk['guideline_id'], "Unknown Guideline")
                    
                    # メインコンテンツをランク付き表示
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
                    
                    # コンテキストボンをメインコンテンツに視覚的に関連付ける
                    st.markdown("""
                    <div style='
                        margin-left: 18px;
                        margin-bottom: 20px;
                        border-left: 3px solid #e6e6e6;
                    '>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"📖 {t['show_context']}"):
                        context_chunks = chunk.get('context_chunks', [])
                        if not context_chunks:
                            st.write(t["no_context"])
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
                                # のチャンクとのオーバーラップをチェック
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

        except Exception as e:
            st.error(f"{t['error']}: {str(e)}")
            return

def find_overlap(text1: str, text2: str, min_length: int = 50) -> str:
    """2つのテキスト間のオーバーラップを検出"""
    # text1の末尾とtext2の先頭で最長の共通部分を探す
    for length in range(min(len(text1), len(text2)), min_length - 1, -1):
        if text1[-length:] == text2[:length]:
            return text2[:length]
    return ""

if __name__ == "__main__":
    main()