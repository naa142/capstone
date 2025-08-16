import streamlit as st
import tempfile
import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    st.error(f"OpenAI API initialization error: {e}")
    client = None

from rag_utils import (
    extract_text,
    detect_language,
    detect_country,
    detect_doc_type,
    chunk_legal_document_semantic,
    embed_text,
    retrieve_top_similar_chunks,
    clean_section_title,
    cohere_client,
    MAX_SECTION_LENGTH,
    get_db_connection,
    DB_EMBEDDING_DIM,
    parse_compliance_response,
    extract_article_number,
    MAX_REFERENCE_LENGTH,
    DocumentMetadata,
    format_retrieved_chunk,
    check_legal_compliance
)


# Streamlit UI Configuration
st.set_page_config(
    page_title="⚖️ Legal Document Compliance Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Debug mode
DEBUG_MODE = st.sidebar.checkbox("Enable debug mode", False)

# API Status Checks
if cohere_client is None:
    st.warning("⚠️ Cohere API not available - using fallback embeddings")

if client is None:
    st.error("❌ OpenAI API not available - compliance analysis will be limited")

# Database validation
if DEBUG_MODE:
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT vector_dims(metadata_node_embedding) FROM document_part_new_embedding LIMIT 1;")
                db_dim = cur.fetchone()[0]
                st.sidebar.markdown("### Database Info")
                st.sidebar.write(f"📐 Database embedding dimension: {db_dim}")
                st.sidebar.write(f"📐 Expected dimension: {DB_EMBEDDING_DIM}")
                
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(DISTINCT(metadata->>'country')) as countries,
                        COUNT(DISTINCT(metadata->>'doc_type')) as doc_types
                    FROM document_new
                """)
                counts = cur.fetchone()
                st.sidebar.write(f"📊 Documents: {counts[0]}")
                st.sidebar.write(f"🌍 Countries: {counts[1]}")
                st.sidebar.write(f"📝 Document types: {counts[2]}")
    except Exception as e:
        st.sidebar.error(f"Database check failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

st.title("⚖️ Legal Document Compliance Analyzer")
st.write("Upload legal documents to check compliance with relevant laws and regulations")

# File uploader
uploaded_file = st.file_uploader("📄 Upload legal document", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("🔍 Analyzing document..."):
            start_time = time.time()
            
            # Extract and analyze document
            raw_text = extract_text(tmp_path)
            language = detect_language(raw_text)
            country = detect_country(raw_text)
            doc_type = detect_doc_type(raw_text)

            if DEBUG_MODE:
                st.sidebar.markdown("### Document Info")
                st.sidebar.metric("Text Length", f"{len(raw_text):,} chars")
                st.sidebar.write(f"Language: {language}")
                st.sidebar.write(f"Country: {country}")
                st.sidebar.write(f"Type: {doc_type}")

            st.success("✅ Document loaded successfully")
            st.markdown(f"**Language**: `{language}` | **Country**: `{country}` | **Type**: `{doc_type}`")

            # Add metadata filters
            st.sidebar.markdown("### 🔍 Metadata Filters")
            topic_filter = st.sidebar.text_input("Filter by topic (optional)")
            authority_filter = st.sidebar.text_input("Filter by authority (optional)")

            # Chunk the document
            chunks = chunk_legal_document_semantic(raw_text)

            if not chunks:
                st.error("❌ No content found after chunking.")
            else:
                st.markdown("### 📑 Document Sections")
                for idx, chunk in enumerate(chunks):
                    with st.expander(f"📄 Section {idx + 1}: {clean_section_title(chunk['header'])}"):
                        st.text_area("Content", chunk["content"], height=250, key=f"content_{idx}", label_visibility="collapsed")

                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🔍 View Retrieved Chunks", key=f"preview_btn_{idx}"):
                                with st.spinner("Finding similar legal references..."):
                                    try:
                                        embedding = embed_text(chunk["content"])
                                        
                                        if DEBUG_MODE:
                                            st.sidebar.markdown("### Embedding Debug")
                                            st.sidebar.write(f"Dimensions: {len(embedding)}")
                                            st.sidebar.write(f"Sample values: {embedding[:5]}...")

                                        similar_chunks = retrieve_top_similar_chunks(
                                            embedding,
                                            country_code=country,
                                            doc_type=doc_type,
                                            topic=topic_filter if topic_filter else None,
                                            authority=authority_filter if authority_filter else None,
                                            similarity_threshold=0.15,
                                            debug=DEBUG_MODE
                                        )

                                        if similar_chunks:
                                            st.markdown(f"#### Found {len(similar_chunks)} similar legal provisions:")
                                            for i, ref in enumerate(similar_chunks):
                                                formatted = format_retrieved_chunk(ref)
                                                with st.expander(f"{i+1}. {formatted['display']['law_title']} (Similarity: {ref['similarity']:.2f})"):
                                                    col_meta, col_content = st.columns([1, 3])
                                                    with col_meta:
                                                        st.markdown("**Metadata:**")
                                                        st.markdown(f"- Authority: {formatted['display']['authority']}")
                                                        if formatted['display'].get('country'):
                                                            st.markdown(f"- Country: {formatted['display']['country']}")
                                                        if formatted['display'].get('doc_type'):
                                                            st.markdown(f"- Type: {formatted['display']['doc_type']}")
                                                    with col_content:
                                                        st.markdown("**Content:**")
                                                        st.markdown(f"**Relevant Article:** {formatted['display']['article']}")
                                                        st.text(formatted['formatted_content'])
                                        else:
                                            st.warning("No similar legal provisions found.")
                                            
                                    except Exception as e:
                                        st.error(f"Error retrieving chunks: {str(e)}")
                                        if DEBUG_MODE:
                                            st.exception(e)

                        with col2:
                            if st.button("Check Compliance", key=f"btn_{idx}"):
                                with st.spinner("⚖️ Analyzing compliance..."):
                                    try:
                                        embedding = embed_text(chunk["content"])
                                        similar_chunks = retrieve_top_similar_chunks(
                                            embedding,
                                            country_code=country,
                                            doc_type=doc_type,
                                            topic=topic_filter if topic_filter else None,
                                            authority=authority_filter if authority_filter else None,
                                            similarity_threshold=0.15
                                        )
                                        
                                        result = check_legal_compliance(
                                            chunk["content"],
                                            similar_chunks,
                                            language
                                        )
                                        
                                        # Display results
                                        status_color = {
                                            "Compliant": "green",
                                            "Partially Compliant": "orange",
                                            "Non-Compliant": "red",
                                            "Error": "gray"
                                        }.get(result["status"], "blue")
                                        
                                        st.markdown(f"### Compliance Status: :{status_color}[{result['status']}]")
                                        
                                        if result['references']:
                                            with st.expander("📜 Referenced Legal Articles"):
                                                for ref in result['references']:
                                                    st.code(ref)
                                        
                                        with st.expander("📝 Full Analysis"):
                                            st.markdown(result['analysis'])
                                        
                                        if result['issues']:
                                            st.warning("#### 🚨 Potential Issues")
                                            for issue in result['issues']:
                                                st.write(f"- {issue}")
                                        
                                        if result['recommendations']:
                                            st.info("#### 💡 Recommendations")
                                            for rec in result['recommendations']:
                                                st.write(f"- {rec}")
                                                
                                        if result['relevant_authorities']:
                                            st.markdown("#### 🏛️ Relevant Authorities")
                                            for auth in result['relevant_authorities']:
                                                st.write(f"- {auth}")
                                                
                                        if result['related_topics']:
                                            st.markdown("#### 🏷️ Related Topics")
                                            for topic in result['related_topics']:
                                                st.write(f"- {topic}")
                                                
                                    except Exception as e:
                                        st.error(f"Compliance analysis failed: {str(e)}")
                                        if DEBUG_MODE:
                                            st.exception(e)

    except Exception as e:
        st.error(f"Document processing failed: {e}")
        if DEBUG_MODE:
            st.exception(e)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass