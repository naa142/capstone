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
    .issues-section, .recommendations-section {
        margin-top: 0.5rem;
    }
    .compliance-result ul {
        padding-left: 1.5rem;
        margin: 0.5rem 0;
    }
    .compliance-result h4, .compliance-result h5 {
        margin: 0.25rem 0;
    }
    .file-success {
        color: #388e3c;
        font-weight: bold;
    }
    .file-error {
        color: #d32f2f;
        font-weight: bold;
    }
    .progress-container {
        margin-bottom: 1rem;
    }
    .reference-item {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f5f5f5;
        border-radius: 0.25rem;
    }
    .section-content {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .reference-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .reference-excerpt {
        margin-top: 0.25rem;
        font-size: 0.9em;
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
    """Save uploaded file with proper extension handling."""
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
    """Document processing pipeline."""
    st.session_state.state.reset()

    # Step 1: Save the file
    with st.spinner("🔄 Saving uploaded file..."):
        tmp_path = save_uploaded_file(uploaded_file)
        if not tmp_path:
            return False

    # Step 2: Extract text
    with st.spinner("📄 Extracting document text..."):
        try:
            st.session_state.state.full_text = extract_text(tmp_path)
            st.session_state.state.filename = uploaded_file.name
        except Exception as e:
            st.session_state.state.last_error = str(e)
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    # Step 3: Analyze metadata
    with st.spinner("🔍 Analyzing document properties..."):
        try:
            st.session_state.state.lang = detect_language(st.session_state.state.full_text)
            st.session_state.state.country_code = detect_country(st.session_state.state.full_text)
            st.session_state.state.doc_type = detect_doc_type(st.session_state.state.full_text)
        except Exception as e:
            st.session_state.state.last_error = f"Analysis error: {str(e)}"
            return False

    # Step 4: Chunk document (semantic)
    with st.spinner("✂️ Segmenting document..."):
        try:
            st.session_state.state.chunks = chunk_legal_document_semantic(st.session_state.state.full_text)
            if not st.session_state.state.chunks:
                st.session_state.state.last_error = (
                    "Document could not be segmented. Please check if the file contains real legal content."
                )
                return False
            # Optional: Warn if all chunks look like headers or are very short (could be a TOC)
            short_chunks = [
                c for c in st.session_state.state.chunks
                if len(c.get("content", "").strip()) < 60
            ]
            if len(short_chunks) == len(st.session_state.state.chunks):
                st.warning(
                    "⚠️ Warning: Document appears to contain only very short sections (possibly a table of contents or index). "
                    "You can still analyze, but results may not be meaningful."
                )
        except Exception as e:
            st.session_state.state.last_error = f"Chunking error: {str(e)}"
            return False

    return True

def display_compliance_result(result: Dict[str, Union[str, List[str]]], lang: str = "en"):
    """Enhanced display with references"""
    status = result.get("status", "Non-Compliant")
    status_config = {
        "Compliant": {"icon": "✅", "class": "compliant"},
        "Partially Compliant": {"icon": "⚠️", "class": "partially-compliant"},
        "Non-Compliant": {"icon": "❌", "class": "non-compliant"},
        "Error": {"icon": "❌", "class": "error"},
        "⚠️ Section too large": {"icon": "⚠️", "class": "warning"},
        "⚠️ Content too large": {"icon": "⚠️", "class": "warning"}
    }
    config = status_config.get(status, status_config["Non-Compliant"])
    text_dir = "rtl" if lang == "ar" else "ltr"
    text_align = "right" if lang == "ar" else "left"
    text_class = "rtl-text" if lang == "ar" else "ltr-text"
    analysis_text = re.sub(r'<[^>]+>', '', result.get("analysis", ""))

    st.markdown(f"""
    <div class="compliance-result {config['class']}" style="text-align:{text_align}; direction:{text_dir};">
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

    if result.get("references"):
        ref_title = "المراجع القانونية" if lang == "ar" else "Legal References"
        with st.expander(f"🔗 {ref_title}"):
            for ref in result["references"]:
                st.markdown(f"""
                <div class="reference-item">
                    <div class="reference-title">{ref['document']} ({ref['type']})</div>
                    <div class="reference-excerpt {'rtl-text' if lang == 'ar' else 'ltr-text'}">
                    {ref['excerpt']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Main UI
st.title("⚖️ Legal Document Compliance Analyzer")
st.markdown("""
Upload legal documents to check compliance with relevant laws and regulations. 
Supports contracts, agreements, policies, and other legal documents in English and Arabic.
""")

uploaded_file = st.file_uploader(
    "Upload legal document (PDF, DOCX, or image)",
    type=["pdf", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="file_uploader"
)

if uploaded_file and st.button("Analyze Document", type="primary"):
    if process_document(uploaded_file):
        st.success("✅ Document processed successfully!")

        # Display document metadata
        lang_display = "Arabic" if st.session_state.state.lang == "ar" else "English"
        doc_type_display = st.session_state.state.doc_type

        st.markdown(f"""
        <div class="file-success">
            <strong>File:</strong> {st.session_state.state.filename}<br>
            <strong>Type:</strong> {doc_type_display}<br>
            <strong>Language:</strong> {lang_display}<br>
            <strong>Jurisdiction:</strong> {st.session_state.state.country_code}<br>
            <strong>Sections:</strong> {len(st.session_state.state.chunks)}
        </div>
        """, unsafe_allow_html=True)
        
        # Download extracted chunks button (key 1)
        st.download_button(
            label="📥 Download Extracted Chunks (JSON)",
            data=json.dumps(st.session_state.state.chunks, ensure_ascii=False, indent=2),
            file_name=f"chunks_{st.session_state.state.filename.split('.')[0]}.json",
            mime="application/json",
            key="download_chunks_1"
        )
    else:
        st.error(f"❌ Processing failed: {st.session_state.state.last_error}")

# Display document sections if available
if st.session_state.state.chunks:
    st.divider()
    st.subheader("📑 Document Sections Analysis")

    # Section selector with cleaned titles
    selected_section = st.selectbox(
        "Select a section to analyze:",
        options=[f"{i+1}. {clean_section_title(chunk['header'])}" for i, chunk in enumerate(st.session_state.state.chunks)],
        index=0,
        key="section_selector"
    )
    selected_index = int(selected_section.split(".")[0]) - 1
    selected_chunk = st.session_state.state.chunks[selected_index]

    # Special highlight for legal reference sections
    if "النصوص القانونية" in clean_section_title(selected_chunk.get("header", "")):
        st.info("This section contains legal references, usually quoted directly from the law.")

    # Display selected section with cleaned title
    with st.expander(f"🔍 View Section: {clean_section_title(selected_chunk['header'])}", expanded=True):
        st.markdown(f"""
        <div class="section-content {'rtl-text' if st.session_state.state.lang == 'ar' else 'ltr-text'}">
            {selected_chunk["content"]}
        </div>
        """, unsafe_allow_html=True)

        # Compliance check button
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
                    "similar_chunks": similar_chunks[:3]
                }
            st.subheader("📋 Compliance Analysis")
            display_compliance_result(result, st.session_state.state.lang)

    # Full document analysis
    st.divider()
    if st.button("Analyze Entire Document", type="primary"):
        full_report = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, chunk in enumerate(st.session_state.state.chunks):
            progress = (i + 1) / len(st.session_state.state.chunks)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing section {i+1} of {len(st.session_state.state.chunks)}...")

            try:
                embedding = embed_text(chunk["content"])
                similar_chunks = retrieve_top_similar_chunks(
                    embedding,
                    st.session_state.state.country_code,
                    st.session_state.state.doc_type
                )
                result = check_legal_compliance(
                    chunk["content"],
                    similar_chunks,
                    st.session_state.state.lang
                )
                full_report.append({
                    "section": clean_section_title(chunk["header"]),
                    "content": chunk["content"],
                    "analysis": result,
                    "similar_references": [
                        {
                            "document": c["document_name"],
                            "type": c.get("document_type", "N/A"),
                            "excerpt": c["chunk_text"][:500] + "..." if len(c["chunk_text"]) > 500 else c["chunk_text"]
                        }
                        for c in similar_chunks[:3]
                    ]
                })
            except Exception as e:
                full_report.append({
                    "section": clean_section_title(chunk["header"]),
                    "error": str(e),
                    "analysis": {
                        "status": "Error",
                        "analysis": str(e),
                        "issues": [],
                        "recommendations": [],
                        "references": []
                    }
                })

        progress_bar.empty()
        status_text.empty()
        st.success("✅ Full analysis completed!")

        # Display summary
        st.subheader("📊 Summary of Findings")
        compliant_count = sum(1 for item in full_report 
                          if isinstance(item, dict) and 
                          item.get("analysis", {}).get("status") == "Compliant")
        partial_count = sum(1 for item in full_report 
                         if isinstance(item, dict) and 
                         item.get("analysis", {}).get("status") == "Partially Compliant")
        non_compliant_count = sum(1 for item in full_report 
                               if isinstance(item, dict) and 
                               item.get("analysis", {}).get("status") == "Non-Compliant")

        # Save JSON for download
        st.session_state.state.full_report = json.dumps(full_report, ensure_ascii=False, indent=2)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Compliant Sections", compliant_count)
        with col2:
            st.metric("Partially Compliant", partial_count)
        with col3:
            st.metric("Non-Compliant", non_compliant_count)

        # Display detailed results
        for i, item in enumerate(full_report):
            if not isinstance(item, dict):
                continue
            with st.expander(f"{i+1}. {clean_section_title(item['section'])}"):
                st.markdown(f"""
                <div class="section-content {'rtl-text' if st.session_state.state.lang == 'ar' else 'ltr-text'}">
                    {item.get("content", "No content available")}
                </div>
                """, unsafe_allow_html=True)
                if "error" in item:
                    st.error(f"Analysis failed: {item['error']}")
                else:
                    display_compliance_result(item["analysis"], st.session_state.state.lang)

        # Download full report button (key 2)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=st.session_state.state.full_report,
            file_name=f"compliance_report_{st.session_state.state.filename.split('.')[0]}.json",
            mime="application/json",
            key="download_report_2"
        )

        # Download extracted chunks (key 3, always available after analysis)
        st.download_button(
            label="📥 Download Extracted Chunks (JSON)",
            data=json.dumps(st.session_state.state.chunks, ensure_ascii=False, indent=2),
            file_name=f"chunks_{st.session_state.state.filename.split('.')[0]}.json",
            mime="application/json",
            key="download_chunks_3"
        )

# Sidebar with info
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=LegalAI", width=150)
    st.markdown("### About This Tool")
    st.markdown("""
    This tool analyzes legal documents for:
    - Regulatory compliance
    - Contractual best practices
    - Potential legal risks
    - Jurisdiction-specific requirements
    """)
    if st.session_state.state.chunks:
        st.markdown("### Document Stats")
        st.metric("Total Sections", len(st.session_state.state.chunks))
        analyzed = len(st.session_state.state.analysis_results)
        st.metric("Sections Analyzed", f"{analyzed}/{len(st.session_state.state.chunks)}")
