# app.py

import streamlit as st
from rag_utils import (
    extract_text,
    detect_language,
    detect_doc_type,
    detect_country,
    chunk_legal_document_semantic,
    embed_text,
    retrieve_top_similar_chunks,
    check_legal_compliance,
    clean_section_title
)
import tempfile
import os
from typing import Dict, List, Optional, Union
import json
import re

# Configure page
st.set_page_config(
    page_title="⚖️ Legal Compliance Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS with improved RTL support
st.markdown("""
<style>
    .rtl-text {
        text-align: right;
        direction: rtl;
        font-family: 'Arial', sans-serif;
        white-space: pre-wrap;
    }
    .ltr-text {
        text-align: left;
        direction: ltr;
        font-family: 'Arial', sans-serif;
        white-space: pre-wrap;
    }
    .compliance-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .compliant {
        background-color: #e6f7e6;
        border-color: #2e7d32;
    }
    .partially-compliant {
        background-color: #fff8e1;
        border-color: #ff8f00;
    }
    .non-compliant {
        background-color: #ffebee;
        border-color: #c62828;
    }
    .warning {
        background-color: #fff3e0;
        border-color: #ffa000;
    }
    .error {
        background-color: #ffebee;
        border-color: #d32f2f;
    }
    .reference-item {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f5f5f5;
        border-radius: 0.25rem;
    }
    .reference-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .section-content {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SessionState:
    def __init__(self):
        self.chunks = []
        self.full_text = ""
        self.full_report = ""
        self.lang = "en"
        self.country_code = "XX"
        self.doc_type = "Unknown"
        self.filename = ""
        self.analysis_results = {}
        self.last_error = ""

    def reset(self):
        self.__init__()

if "state" not in st.session_state:
    st.session_state.state = SessionState()

def save_uploaded_file(uploaded_file) -> Optional[str]:
    try:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
            st.session_state.state.last_error = f"Unsupported file extension: {file_ext}"
            return None
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.session_state.state.last_error = f"File save error: {str(e)}"
        return None

def process_document(uploaded_file):
    st.session_state.state.reset()
    with st.spinner("🔄 Saving uploaded file..."):
        tmp_path = save_uploaded_file(uploaded_file)
        if not tmp_path:
            return False

    with st.spinner("📄 Extracting document text..."):
        try:
            st.session_state.state.full_text = extract_text(tmp_path)
            st.session_state.state.filename = uploaded_file.name
        except Exception as e:
            st.session_state.state.last_error = str(e)
            return False
        finally:
            try: os.unlink(tmp_path)
            except: pass

    with st.spinner("🔍 Analyzing document properties..."):
        try:
            st.session_state.state.lang = detect_language(st.session_state.state.full_text)
            st.session_state.state.country_code = detect_country(st.session_state.state.full_text)
            st.session_state.state.doc_type = detect_doc_type(st.session_state.state.full_text)
        except Exception as e:
            st.session_state.state.last_error = f"Analysis error: {str(e)}"
            return False

    with st.spinner("✂️ Segmenting document..."):
        try:
            st.session_state.state.chunks = chunk_legal_document_semantic(st.session_state.state.full_text)
            if not st.session_state.state.chunks:
                st.session_state.state.last_error = "Document could not be segmented. Please check if the file contains real legal content."
                return False
        except Exception as e:
            st.session_state.state.last_error = f"Chunking error: {str(e)}"
            return False

    return True

def display_compliance_result(result: Dict[str, Union[str, List[str]]], lang: str = "en"):
    status = result.get("status", "Non-Compliant")
    status_config = {
        "Compliant": {"icon": "✅", "class": "compliant"},
        "Partially Compliant": {"icon": "⚠️", "class": "partially-compliant"},
        "Non-Compliant": {"icon": "❌", "class": "non-compliant"},
        "Error": {"icon": "❌", "class": "error"},
    }
    config = status_config.get(status, status_config["Non-Compliant"])
    text_class = "rtl-text" if lang == "ar" else "ltr-text"
    analysis_text = re.sub(r'<[^>]+>', '', result.get("analysis", ""))

    st.markdown(f"""
    <div class="compliance-result {config['class']}">
        <h4>{config['icon']} {status}</h4>
        <div class="{text_class}">{analysis_text}</div>
    </div>
    """, unsafe_allow_html=True)

    if result.get("issues"):
        issues_title = "المشاكل الرئيسية" if lang == "ar" else "Key Issues"
        st.markdown(f"**{issues_title}:**")
        for issue in result["issues"]:
            st.markdown(f"- {issue}")

    if result.get("recommendations"):
        rec_title = "التوصيات" if lang == "ar" else "Recommendations"
        st.markdown(f"**{rec_title}:**")
        for rec in result["recommendations"]:
            st.markdown(f"- {rec}")

# Main UI
st.title("⚖️ Legal Document Compliance Analyzer")
st.markdown("Upload legal documents to check compliance with relevant laws and regulations.")

uploaded_file = st.file_uploader("Upload legal document", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file and st.button("Analyze Document", type="primary"):
    if process_document(uploaded_file):
        st.success("✅ Document processed successfully!")

        st.markdown(f"""
        **File:** {st.session_state.state.filename}  
        **Language:** {"Arabic" if st.session_state.state.lang == "ar" else "English"}  
        **Jurisdiction:** {st.session_state.state.country_code}  
        **Document Type:** {st.session_state.state.doc_type}  
        **Sections:** {len(st.session_state.state.chunks)}  
        """)

        st.download_button(
            "📥 Download Extracted Chunks (JSON)",
            data=json.dumps(st.session_state.state.chunks, ensure_ascii=False, indent=2),
            file_name="chunks.json",
            mime="application/json"
        )
    else:
        st.error(f"❌ Processing failed: {st.session_state.state.last_error}")

if st.session_state.state.chunks:
    st.subheader("📑 Document Sections")

    selected_index = st.selectbox(
        "Select a section to analyze:",
        options=[f"{i+1}. {clean_section_title(c['header'])}" for i, c in enumerate(st.session_state.state.chunks)],
        index=0
    ).split(".")[0]
    selected_index = int(selected_index) - 1
    selected_chunk = st.session_state.state.chunks[selected_index]

    with st.expander(f"🔍 View Section: {clean_section_title(selected_chunk['header'])}", expanded=True):
        st.markdown(f"""
        <div class="section-content {'rtl-text' if st.session_state.state.lang == 'ar' else 'ltr-text'}">
            {selected_chunk["content"]}
        </div>
        """, unsafe_allow_html=True)

        if st.button("Check Compliance for This Section", key=f"check_{selected_index}"):
            with st.spinner("⚖️ Analyzing compliance..."):
                embedding = embed_text(selected_chunk["content"])
                similar_chunks = retrieve_top_similar_chunks(
                    embedding,
                    st.session_state.state.country_code,
                    st.session_state.state.doc_type
                )
                result = check_legal_compliance(
                    selected_chunk["content"],
                    similar_chunks,
                    st.session_state.state.lang
                )
                st.session_state.state.analysis_results[selected_index] = {
                    "result": result,
                    "similar_chunks": similar_chunks
                }
            st.subheader("📋 Compliance Analysis")
            display_compliance_result(result, st.session_state.state.lang)

            with st.expander("🔎 View Retrieved Legal References"):
                for chunk in similar_chunks:
                    st.markdown(f"""
                    <div class="reference-item">
                        <div class="reference-title">{chunk['document_name']} ({chunk.get('document_type', 'N/A')})</div>
                        <div class="reference-excerpt {'rtl-text' if st.session_state.state.lang == 'ar' else 'ltr-text'}">
                            {chunk['chunk_text'][:500]}{'...' if len(chunk['chunk_text']) > 500 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("Check legal documents for compliance with local regulations.")
