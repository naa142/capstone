import os
import re
import fitz
import mammoth
from PIL import Image
import pytesseract
import numpy as np
import psycopg2
import json
import time
import hashlib
from typing import List, Dict, Optional, Union, Tuple
from dotenv import load_dotenv
from pathlib import Path
from pdf2image import convert_from_path
from langdetect import detect
import cohere
import psycopg2.extras
import traceback
from pydantic import BaseModel
from datetime import datetime
from openai import OpenAI
from psycopg2 import pool, errors as pg_errors
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize API clients
try:
    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY")) if os.getenv("COHERE_API_KEY") else None
    logger.info("Cohere client initialized successfully")
except Exception as e:
    logger.error(f"Cohere API initialization error: {e}")
    cohere_client = None

try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"OpenAI API initialization error: {e}")
    openai_client = None

# Constants
MAX_SECTION_LENGTH = 3000
MAX_REFERENCE_LENGTH = 1000
MAX_REFERENCES = 50
MAX_TOKENS_SAFE = 7000
MAX_WORDS_PER_CHUNK = 300
MIN_WORDS_PER_CHUNK = 50
ARABIC_HEADER_THRESHOLD = 0.7
COHERE_EMBEDDING_DIM = 1024
DB_EMBEDDING_DIM = 2000
TARGET_EMBEDDING_DIM = DB_EMBEDDING_DIM
MIN_SIMILARITY_THRESHOLD = 0.15
FALLBACK_SIMILARITY_THRESHOLD = 0.05
COHERE_EMBEDDING_MODEL = "embed-v4.0"
MAX_QUERY_TIME_MS = 30000
MAX_CONNECTION_RETRIES = 3
CONNECTION_RETRY_DELAY = 1

# Connection pool
connection_pool = None

def init_connection_pool():
    global connection_pool
    if not connection_pool:
        try:
            # Get all values from environment variables
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")

            # Ensure all variables are present
            if not all([db_host, db_port, db_name, db_user, db_password]):
                raise ValueError("❌ Missing one or more required DB environment variables in .env")

            connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=db_host,
                port=int(db_port),
                dbname=db_name,
                user=db_user,
                password=db_password,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
                options=f"-c statement_timeout={MAX_QUERY_TIME_MS}"
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

def get_db_connection(max_retries: int = MAX_CONNECTION_RETRIES) -> Optional[psycopg2.extensions.connection]:
    init_connection_pool()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            conn = connection_pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                if cur.fetchone()[0] != 1:
                    raise psycopg2.OperationalError("Connection test failed")
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            last_error = e
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(CONNECTION_RETRY_DELAY * (attempt + 1))
    
    logger.error(f"Failed to get database connection after {max_retries} attempts")
    if last_error:
        raise last_error
    return None

def return_db_connection(conn):
    if conn:
        try:
            if not conn.closed:
                conn.rollback()
            connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")

class DocumentMetadata(BaseModel):
    law: Optional[str] = None
    lawTitle: Optional[str] = None
    date: Optional[List[str]] = None
    authority: Optional[str] = None
    requester: Optional[str] = None
    topic: Optional[str] = None
    opponents: Optional[str] = None
    result: Optional[str] = None
    pathMetadata: Optional[List[str]] = None
    fileNameMetadata: Optional[Dict[str, str]] = None
    country: Optional[str] = None
    doc_type: Optional[str] = None
    language: Optional[str] = None
    signatories: Optional[List[str]] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".docx":
            return extract_text_from_docx(path)
        elif ext == ".pdf":
            return extract_text_from_pdf(path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            return extract_text_from_image(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Error extracting text from {path}: {e}")
        raise ValueError(f"Failed to extract text: {str(e)}")

def extract_text_from_docx(path: str) -> str:
    try:
        with open(path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file)
            return result.value
    except Exception as e:
        logger.error(f"Error extracting from DOCX: {e}")
        raise

def extract_text_from_pdf(path: str) -> str:
    try:
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) > 100:
                return text
        
        images = convert_from_path(path)
        return "\n".join(pytesseract.image_to_string(img, lang='ara+eng') for img in images)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise

def extract_text_from_image(path: str) -> str:
    try:
        return pytesseract.image_to_string(Image.open(path), lang='ara+eng')
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        raise

def detect_language(text: str) -> str:
    try:
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = max(len(re.sub(r'\s', '', text)), 1)
        if arabic_chars / total_chars > 0.3:
            return "ar"
        return detect(text) or "en"
    except Exception as e:
        logger.warning(f"Language detection failed, defaulting to English: {e}")
        return "en"

def detect_country(text: str) -> str:
    country_patterns = {
        "LB": r"(?:لبنان|الجمهورية اللبنانية|حكومة لبنان)",
        "SA": r"(?:المملكة العربية السعودية|السعودية|حكومة السعودية)",
        "AE": r"(?:دولة الإمارات|الإمارات العربية المتحدة|أبو ظبي|دبي)",
        "EG": r"(?:مصر|جمهورية مصر العربية|حكومة مصر)",
        "IQ": r"(?:العراق|جمهورية العراق|حكومة العراق)",
        "JO": r"(?:الأردن|المملكة الأردنية الهاشمية|حكومة الأردن)"
    }
    
    for code, pattern in country_patterns.items():
        if re.search(pattern, text[:2000], re.IGNORECASE):
            return code
    return "XX"

def detect_doc_type(text: str) -> str:
    text = text.lower()[:2000]
    
    doc_categories = {
        "Contract": {
            "Lease Agreement": ["إيجار", "عقد إيجار", "استئجار", "lease"],
            "Employment Agreement": ["عمل", "وظيفة", "توظيف", "عقد عمل", "employment"],
            "Service Agreement": ["خدمات", "مقاول", "service agreement"],
            "NDA": ["سرية", "عدم إفشاء", "nda", "non-disclosure"]
        },
        "Legislation": {
            "Law": ["قانون", "تشريع", "law no", "مرسوم"],
            "Regulation": ["لائحة", "نظام", "regulation", "تعليمات"],
            "Amendment": ["تعديل", "تعديلات", "amendment"]
        },
        "Court Document": {
            "Decision": ["حكم", "قرار", "ruling", "judgment"],
            "Appeal": ["استئناف", "appeal"],
            "Complaint": ["شكوى", "دعوى", "complaint", "claim"]
        },
        "Corporate": {
            "Bylaws": ["نظام أساسي", "bylaws"],
            "Resolution": ["قرار مجلس", "resolution"],
            "Policy": ["سياسة", "policy"]
        }
    }
    
    for category, subcategories in doc_categories.items():
        for subcategory, keywords in subcategories.items():
            for kw in keywords:
                if kw in text:
                    return f"{category} - {subcategory}"
    
    general_types = {
        "Contract": ["عقد", "اتفاق", "اتفاقية", "تعاقد", "contract"],
        "Terms": ["شروط", "أحكام", "terms and conditions"],
        "Notice": ["إشعار", "إنذار", "إبلاغ", "notice"]
    }
    
    for doc_type, keywords in general_types.items():
        for kw in keywords:
            if kw in text:
                return doc_type
    
    return "Unknown"

def extract_comprehensive_metadata(text: str) -> Dict[str, Union[str, List[str]]]:
    metadata = {
        'authority': extract_authority(text),
        'law_title': extract_law_title(text),
        'article': extract_article_number(text),
        'doc_type': detect_doc_type(text),
        'country': detect_country(text),
        'signatories': extract_signatories(text),
        'effective_date': extract_date(text, 'effective'),
        'expiration_date': extract_date(text, 'expiration'),
        'language': detect_language(text),
        'key_terms': extract_key_terms(text),
        'related_articles': find_related_articles(text)
    }
    return {k: v for k, v in metadata.items() if v is not None}

def extract_authority(text: str) -> Optional[str]:
    authority_patterns = [
        r'(?:صادر عن|باسم|من قبل|الجهة المختصة|السلطة المختصة)[:\s]*(.*?)(?:\n|\.|;|$)',
        r'(?:بموجب (?:قرار|تعميم|كتاب) (?:رقم)?.*? (?:الصادر عن|من) (.*?)(?:\s|$))',
        r'(?:تخضع (?:لأحكام|لرقابة) (.*?)(?:\s|$))',
        r'(?:الموقع (?:من|بواسطة) (.*?)(?:\s|$))'
    ]
    
    for pattern in authority_patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE)
        if match:
            authority = clean_entity_name(match.group(1))
            if authority and len(authority) > 3:
                return authority
    
    known_authorities = [
        "وزارة العدل", "المحكمة العليا", "هيئة التشريع",
        "وزارة التجارة", "البلدية", "الجهة المختصة",
        "مجلس النواب", "رئاسة الوزراء"
    ]
    
    for auth in known_authorities:
        if auth in text[:2000]:
            return auth
    
    return None

def extract_law_title(text: str) -> Optional[str]:
    title_patterns = [
        r'(?:عنوان|اسم) (?:القانون|اللوائح|النظام|العقد|الاتفاقية)[:\s]*(.*?)(?:\n|$)',
        r'(?:بموجب (?:قانون|نظام|مرسوم) (.*?)(?:\s|$))',
        r'(?:المسمى (?:ب|فيما يلي) (.*?)(?:\s|$))',
        r'(?:ينشر (?:قانون|نظام) (.*?)(?:\s|$))'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE)
        if match:
            title = clean_text(match.group(1))
            if len(title) > 5:
                return title
    return None

def extract_signatories(text: str) -> List[str]:
    signatory_patterns = [
        r'(?:بين[\s]*(.*?)[\s]و[\s]*(.*?)(?:\n|$))',
        r'(?:يوقع هذا (?:الوثيقة|العقد)[\s]*(?:من قبل|بين)[\s]*(.*?)(?:\n|$))',
        r'(?:طرف أول[\s]*(.*?)[\s]و[\s]*طرف ثاني[\s]*(.*?)(?:\n|$))',
        r'(?:المتعاقدون[\s]*(.*?)(?:\n|$))'
    ]
    
    signatories = []
    for pattern in signatory_patterns:
        matches = re.finditer(pattern, text[:5000], re.IGNORECASE)
        for match in matches:
            for group in match.groups():
                if group:
                    signatory = clean_entity_name(group)
                    if signatory and signatory not in signatories:
                        signatories.append(signatory)
    return signatories if signatories else None

def extract_date(text: str, date_type: str) -> Optional[str]:
    date_patterns = {
        'effective': [
            r'(?:تاريخ التنفيذ|نافذ المفعول|تاريخ البدء)[:\s]*(.*?)(?:\n|$)',
            r'(?:يبدأ العمل بهذا (?:القانون|العقد) في (.*?)(?:\s|$))'
        ],
        'expiration': [
            r'(?:تاريخ الانتهاء|تنتهي المدة|تاريخ الانقضاء)[:\s]*(.*?)(?:\n|$)',
            r'(?:ينتهي هذا (?:القانون|العقد) في (.*?)(?:\s|$))'
        ]
    }
    
    for pattern in date_patterns[date_type]:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            date_str = standardize_date(match.group(1))
            if date_str:
                return date_str
    return None

def standardize_date(date_str: str) -> Optional[str]:
    try:
        date_str = re.sub(r'[\u0660-\u0669]', lambda x: str(ord(x.group(0)) - ord('\u0660')), date_str)
        
        patterns = [
            r'(?P<day>\d{1,2})[/\-](?P<month>\d{1,2})[/\-](?P<year>\d{2,4})',
            r'(?P<year>\d{4})[/\-](?P<month>\d{1,2})[/\-](?P<day>\d{1,2})',
            r'(?P<day>\d{1,2})\s+(?P<month>\S+)\s+(?P<year>\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                parts = match.groupdict()
                if 'month' in parts and len(parts['month']) > 2:
                    month_map = {
                        'يناير': 1, 'فبراير': 2, 'مارس': 3, 'أبريل': 4,
                        'مايو': 5, 'يونيو': 6, 'يوليو': 7, 'أغسطس': 8,
                        'سبتمبر': 9, 'أكتوبر': 10, 'نوفمبر': 11, 'ديسمبر': 12
                    }
                    parts['month'] = month_map.get(parts['month'], parts['month'])
                
                year = int(parts['year'])
                if year < 100:
                    year += 2000 if year < 50 else 1900
                
                return f"{year}-{int(parts['month']):02d}-{int(parts['day']):02d}"
        return None
    except:
        return None

def clean_entity_name(name: str) -> str:
    name = re.sub(r'[\n\r\t]', ' ', name)
    name = re.sub(r'[،؛:.]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    prefixes = [
        'السيد', 'الاستاذ', 'المكرم', 'المحترم', 
        'المادة', 'المعدة', 'رقم', 'شركة', 'مؤسسة'
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
    
    return name if len(name) > 3 else None

def clean_text(text: str) -> str:
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[،؛:.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_key_terms(text: str) -> List[str]:
    terms = set()
    
    arabic_terms = re.findall(r'\b(?:المادة|البند|الفقرة|عقد|اتفاقية|شرط|التزام|مسؤولية|مخالفة|عقوبة)\b', text)
    terms.update(arabic_terms)
    
    english_terms = re.findall(r'\b(?:article|clause|paragraph|contract|agreement|obligation|liability|violation|penalty)\b', text, re.IGNORECASE)
    terms.update(english_terms)
    
    entities = re.findall(r'(?:[A-Z][a-z\u0600-\u06FF]+(?:\s+[A-Z][a-z\u0600-\u06FF]+)+)', text)
    terms.update(entities)
    
    return sorted(terms, key=len, reverse=True)[:15]

def find_related_articles(text: str) -> List[str]:
    references = set()
    
    arabic_refs = re.findall(r'(?:المادة|البند|الفقرة)\s*[\d\u0660-\u0669]+(?:\s*إلى\s*[\d\u0660-\u0669]+)?', text)
    references.update(arabic_refs)
    
    english_refs = re.findall(r'(?:Article|Clause|Section)\s*\d+(?:\s*to\s*\d+)?', text, re.IGNORECASE)
    references.update(english_refs)
    
    return sorted(references)

def extract_article_number(text: str) -> str:
    arabic_patterns = [
        r'المادة\s*([\d\u0660-\u0669]+)',
        r'البند\s*([\d\u0660-\u0669]+)',
        r'الفقرة\s*([\d\u0660-\u0669]+)'
    ]
    
    for pattern in arabic_patterns:
        match = re.search(pattern, text)
        if match:
            article_num = match.group(1)
            article_num = re.sub(r'[\u0660-\u0669]', lambda x: str(ord(x.group(0)) - ord('\u0660')), article_num)
            return f"المادة {article_num}" if article_num.isdigit() else "N/A"
    
    english_patterns = [
        r'Article\s*(\d+)',
        r'Clause\s*(\d+)',
        r'Section\s*(\d+)'
    ]
    
    for pattern in english_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"Article {match.group(1)}"
    
    return "N/A"

def chunk_legal_document_semantic(text: str) -> List[Dict]:
    try:
        text = text.replace('\r\n', '\n').replace('\u2028', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        chunks = []
        current_chunk = []
        current_header = "Document Preamble"
        current_level = 0
        
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#'):
                if current_chunk:
                    chunks.append({
                        "header": clean_section_title(current_header),
                        "level": current_level,
                        "content": "\n".join(current_chunk).strip()
                    })
                    current_chunk = []
                current_level = line.count('#')
                current_header = line.split(maxsplit=1)[1] if ' ' in line else line[1:]
                i += 1
                continue
                
            arabic_section_match = re.match(
                r'^(?:الجزء|الباب|الفصل|القسم|المادة|البند|الفقرة)\s*[\d\u0660-\u0669]*[\.\:]?\s*(.*)',
                line
            )
            if arabic_section_match and is_arabic_header(line):
                if current_chunk:
                    chunks.append({
                        "header": clean_section_title(current_header),
                        "level": current_level,
                        "content": "\n".join(current_chunk).strip()
                    })
                    current_chunk = []
                current_header = line.strip()
                current_level = 2 if "الجزء" in line or "الباب" in line else 3
                i += 1
                continue
                
            if i + 1 < len(lines) and re.match(r'^[-=]{3,}$', lines[i+1].strip()):
                if current_chunk:
                    chunks.append({
                        "header": clean_section_title(current_header),
                        "level": current_level,
                        "content": "\n".join(current_chunk).strip()
                    })
                    current_chunk = []
                current_header = line.strip()
                current_level = 1 if '=' in lines[i+1] else 2
                i += 2
                continue
                
            if line:
                current_chunk.append(line)
            i += 1
            
        if current_chunk:
            chunks.append({
                "header": clean_section_title(current_header),
                "level": current_level,
                "content": "\n".join(current_chunk).strip()
            })
            
        return chunks
        
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        return [{
            "header": "Error",
            "level": 1,
            "content": f"Chunking failed: {str(e)}"
        }]

def clean_section_title(title: str) -> str:
    if not title:
        return ""
    
    title = re.sub(r'<[^>]+>', '', title)
    title = re.sub(r'^[\d\.\s]+', '', title)
    title = re.sub(r'[_\-=]{4,}', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', title))
    total_chars = max(len(re.sub(r'\s', '', title)), 1)
    
    if arabic_chars / total_chars > ARABIC_HEADER_THRESHOLD:
        title = re.sub(r'^[\s\d\u0660-\u0669]+[\-\.\)\s]*', '', title)
        title = re.sub(r'[ـَُِّٰٓ]+', '', title)
        title = re.sub(r'[،؛؟]', ' ', title)
    
    return title

def is_arabic_header(line: str) -> bool:
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', line))
    total_chars = max(len(re.sub(r'\s', '', line)), 1)
    return (arabic_chars / total_chars) > ARABIC_HEADER_THRESHOLD

def embed_text(text: str) -> List[float]:
    if not text.strip():
        return [0.0] * TARGET_EMBEDDING_DIM

    clean_text = re.sub(r'\s+', ' ', text.strip())[:5000]

    if not cohere_client:
        raise RuntimeError("Cohere client is not initialized. Cannot generate embeddings.")

    try:
        res = cohere_client.embed(
            texts=[clean_text],
            model=COHERE_EMBEDDING_MODEL,
            input_type="search_document"
        )
        
        embedding = np.array(res.embeddings[0])
        embedding = embedding / np.linalg.norm(embedding)
        
        if len(embedding) != COHERE_EMBEDDING_DIM:
            embedding = embedding[:COHERE_EMBEDDING_DIM]
            if len(embedding) < COHERE_EMBEDDING_DIM:
                embedding = np.pad(embedding, (0, COHERE_EMBEDDING_DIM - len(embedding)))
        
        padded_embedding = np.zeros(TARGET_EMBEDDING_DIM)
        repeat_factor = int(np.ceil(TARGET_EMBEDDING_DIM / COHERE_EMBEDDING_DIM))
        for i in range(repeat_factor):
            start = i * COHERE_EMBEDDING_DIM
            end = start + COHERE_EMBEDDING_DIM
            if end > TARGET_EMBEDDING_DIM:
                end = TARGET_EMBEDDING_DIM
            padded_embedding[start:end] = embedding[:end-start]
        
        return list(padded_embedding)
        
    except Exception as e:
        raise RuntimeError(f"Cohere embedding failed: {e}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

def retrieve_top_similar_chunks(
    embedding: List[float],
    country_code: str,
    doc_type: str = None,
    topic: str = None,
    authority: str = None,
    top_k: int = 5,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    priority: int = 1,  # NEW: Added priority parameter (default to 1)
    debug: bool = False,
    max_retries: int = 2
) -> List[Dict]:
    """
    ANN-first retrieval with two indexed strategies + a guaranteed fallback.
    - Strategy A: Fast ANN (no threshold), global
    - Strategy B: Fast ANN (no threshold), country-filtered
    - Strategy C (fallback): Small candidate set then ANN, guaranteed to return quickly
    - NEW: All strategies now filter by priority (default priority=1)

    Returns up to top_k items, re-ranked by similarity and filtered by threshold.
    """
    # 1) Ensure correct embedding length
    if len(embedding) != TARGET_EMBEDDING_DIM:
        logger.warning(f"Received embedding with {len(embedding)} dimensions, expected {TARGET_EMBEDDING_DIM}")
        embedding = _normalize_embedding_dimensions(embedding)

    # Prepare vector parameter once
    embedding_array = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return []

        # --- STRATEGY A: ANN global (fast) -----------------------------------
        ann_query_global = """
        SELECT 
            dpn.id,
            dpn.content,
            dpn.header,
            dn.name,
            dn.metadata->>'law' AS law,
            COALESCE(NULLIF(dn.metadata->>'authority', ''), 'Unknown Authority') AS authority,
            COALESCE(NULLIF(dn.metadata->>'lawTitle', ''), dn.name) AS law_title,
            dn.metadata->>'topic' AS topic,
            dn.metadata->>'result' AS result,
            dn.metadata->>'country' AS country,
            dn.metadata->>'doc_type' AS doc_type,
            dn.metadata AS full_metadata,
            (dpe.metadata_node_embedding <=> %s::vector) AS dist
        FROM document_part_new dpn
        JOIN document_part_new_embedding dpe ON dpn.id = dpe.document_part_id
        JOIN document_new dn ON dpn.document_id = dn.id
        WHERE dn.priority = %s  -- NEW: Priority filter
        ORDER BY dpe.metadata_node_embedding <=> %s::vector ASC
        LIMIT %s
        """

        # --- STRATEGY B: ANN country-filtered (fast) --------------------------
        ann_query_country = """
        WITH country_docs AS (
            SELECT id FROM document_new 
            WHERE metadata->>'country' = %s
            AND priority = %s  -- NEW: Priority filter
        )
        SELECT 
            dpn.id,
            dpn.content,
            dpn.header,
            dn.name,
            dn.metadata->>'law' AS law,
            COALESCE(NULLIF(dn.metadata->>'authority', ''), 'Unknown Authority') AS authority,
            COALESCE(NULLIF(dn.metadata->>'lawTitle', ''), dn.name) AS law_title,
            dn.metadata->>'topic' AS topic,
            dn.metadata->>'result' AS result,
            dn.metadata->>'country' AS country,
            dn.metadata->>'doc_type' AS doc_type,
            dn.metadata AS full_metadata,
            (dpe.metadata_node_embedding <=> %s::vector) AS dist
        FROM document_part_new dpn
        JOIN document_part_new_embedding dpe ON dpn.id = dpe.document_part_id
        JOIN document_new dn ON dpn.document_id = dn.id
        JOIN country_docs cd ON dn.id = cd.id
        ORDER BY dpe.metadata_node_embedding <=> %s::vector ASC
        LIMIT %s
        """

        # --- STRATEGY C: Guaranteed fallback ----------------------------------
        fallback_query = """
        SELECT 
            dpn.id,
            dpn.content,
            dpn.header,
            dn.name,
            dn.metadata->>'law' AS law,
            COALESCE(NULLIF(dn.metadata->>'authority', ''), 'Unknown Authority') AS authority,
            COALESCE(NULLIF(dn.metadata->>'lawTitle', ''), dn.name) AS law_title,
            dn.metadata->>'topic' AS topic,
            dn.metadata->>'result' AS result,
            dn.metadata->>'country' AS country,
            dn.metadata->>'doc_type' AS doc_type,
            dn.metadata AS full_metadata,
            (dpe.metadata_node_embedding <=> %s::vector) AS dist
        FROM document_part_new dpn
        JOIN document_part_new_embedding dpe ON dpn.id = dpe.document_part_id
        JOIN document_new dn ON dpn.document_id = dn.id
        WHERE dn.priority = %s  -- NEW: Priority filter
        ORDER BY dpe.metadata_node_embedding <=> %s::vector ASC
        LIMIT %s
        """

        # how many raw candidates to fetch before re-ranking and thresholding
        raw_limit_global = max(20, top_k * 5)
        raw_limit_country = max(20, top_k * 4)
        raw_limit_fallback = max(30, top_k * 6)

        strategies = [
            {
                "name": "ANN (global)",
                "query": ann_query_global,
                "params": [embedding_array, priority, embedding_array, raw_limit_global],  # Added priority
            },
            {
                "name": "ANN (country-filtered)",
                "query": ann_query_country,
                "params": [country_code, priority, embedding_array, embedding_array, raw_limit_country],  # Added priority
            },
            {
                "name": "Fallback (guaranteed)",
                "query": fallback_query,
                "params": [embedding_array, priority, embedding_array, raw_limit_fallback],  # Added priority
            },
        ]

        # [Rest of the function remains exactly the same...]
        results = []
        for strategy in strategies:
            if results:
                break  # we already have something good from a prior strategy

            for attempt in range(max_retries):
                try:
                    if debug:
                        logger.info(f"Trying strategy: {strategy['name']} (attempt {attempt + 1})")

                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(f"SET LOCAL statement_timeout = {MAX_QUERY_TIME_MS}")
                        t0 = time.time()
                        cur.execute(strategy["query"], strategy["params"])
                        batch = cur.fetchall()
                        t_ms = (time.time() - t0) * 1000
                        if debug:
                            logger.info(f"{strategy['name']} returned {len(batch)} rows in {t_ms:.1f} ms")

                    if batch:
                        results = batch
                        break

                except pg_errors.QueryCanceled:
                    logger.warning(f"Query timeout on attempt {attempt + 1} for strategy {strategy['name']}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        conn = reset_connection(conn)
                    continue

                except pg_errors.InFailedSqlTransaction:
                    logger.warning("Transaction failed, resetting connection")
                    conn = reset_connection(conn)
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    continue

                except Exception as e:
                    logger.error(f"Error in strategy {strategy['name']}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        conn = reset_connection(conn)
                    continue

        if not results:
            if debug:
                logger.info("No rows returned from all strategies.")
            return []

        # Post-process: convert to similarity, re-rank, dedupe, threshold, and trim
        formatted: List[Dict] = []
        for row in results:
            try:
                dist = float(row["dist"])
                similarity = 1.0 - dist

                if isinstance(row.get("full_metadata"), dict):
                    metadata = row["full_metadata"]
                else:
                    try:
                        metadata = json.loads(row["full_metadata"]) if row.get("full_metadata") else {}
                    except (TypeError, json.JSONDecodeError):
                        metadata = {}

                item = {
                    "id": row["id"],
                    "content": row["content"],
                    "header": row["header"],
                    "name": row.get("name", "Untitled Document"),
                    "similarity": similarity,
                    "law": metadata.get("law", row.get("law", "Unknown Law")),
                    "law_title": metadata.get("lawTitle", row.get("law_title", "")) or row.get("name", ""),
                    "authority": metadata.get("authority", row.get("authority", "Unknown Authority")),
                    "topic": metadata.get("topic", row.get("topic")),
                    "result": metadata.get("result", row.get("result")),
                    "country": metadata.get("country", row.get("country", country_code)),
                    "doc_type": metadata.get("doc_type", row.get("doc_type", doc_type)),
                    "article": extract_article_number(row["content"]),
                    "metadata": metadata,
                }
                formatted.append(item)
            except Exception as e:
                logger.error(f"Error formatting row: {e}")

        formatted.sort(key=lambda x: x["similarity"], reverse=True)

        seen_keys = set()
        deduped = []
        for item in formatted:
            key = (item.get("law_title") or "", item.get("article") or "")
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(item)

        filtered = [x for x in deduped if x["similarity"] >= similarity_threshold]

        if not filtered and deduped:
            relaxed = max(FALLBACK_SIMILARITY_THRESHOLD, similarity_threshold / 2)
            logger.warning(f"No results above threshold {similarity_threshold:.2f}; relaxing to {relaxed:.2f}")
            filtered = [x for x in deduped if x["similarity"] >= relaxed]

        return filtered[:top_k]

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        traceback.print_exc()
        return []
    finally:
        if conn:
            return_db_connection(conn)


def _normalize_embedding_dimensions(embedding: List[float]) -> List[float]:
    if len(embedding) > TARGET_EMBEDDING_DIM:
        return embedding[:TARGET_EMBEDDING_DIM]
    else:
        embedding = np.array(embedding)
        embedding = np.tile(embedding, int(np.ceil(TARGET_EMBEDDING_DIM / len(embedding))))[:TARGET_EMBEDDING_DIM]
        return list(embedding)

def parse_compliance_response(response_text: str, lang: str) -> Dict[str, Union[str, List[str]]]:
    result = {
        "status": "Unknown",
        "analysis": response_text,
        "issues": [],
        "recommendations": [],
        "references": [],
        "relevant_authorities": [],
        "related_topics": []
    }
    
    if not response_text:
        return result
        
    try:
        response_text = re.sub(r'<[^>]+>', '', response_text)
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        
        status_patterns = {
            "Compliant": [
                r"Compliance Status:\s*(Fully )?Compliant",
                r"حالة التوافق:\s*(متوافق تمامًا|متوافق)",
                r"(?:is|are) (fully )?compliant",
                r"لا يوجد مخالفات قانونية",
                r"مطابق(?:ة)? ل(?:جميع)? المتطلبات القانونية"
            ],
            "Partially Compliant": [
                r"Compliance Status:\s*Partially Compliant",
                r"حالة التوافق:\s*متوافق جزئيًا",
                r"partial(?:ly)? compliant",
                r"بعض المخالفات القانونية",
                r"يتوافق جزئيًا مع"
            ],
            "Non-Compliant": [
                r"Compliance Status:\s*Non-Compliant",
                r"حالة التوافق:\s*غير متوافق",
                r"(?:is|are) not compliant",
                r"يخالف(?:ة)? المتطلبات القانونية",
                r"غير مطابق(?:ة)? ل"
            ]
        }
        
        for status, patterns in status_patterns.items():
            if any(re.search(pattern, response_text, re.IGNORECASE) for pattern in patterns):
                result["status"] = status
                break
                
        if result["status"] == "Unknown":
            if lang == "ar":
                if re.search(r"(?:يخالف|غير مطابق|مخالفة)", response_text):
                    result["status"] = "Non-Compliant"
                elif re.search(r"(?:جزئيًا|بعض)", response_text):
                    result["status"] = "Partially Compliant"
                elif re.search(r"(?:متوافق|مطابق)", response_text):
                    result["status"] = "Compliant"
            else:
                if re.search(r"(?:violates|non-compliant|does not comply)", response_text, re.IGNORECASE):
                    result["status"] = "Non-Compliant"
                elif re.search(r"(?:partial|some)", response_text, re.IGNORECASE):
                    result["status"] = "Partially Compliant"
                elif re.search(r"(?:compliant|complies)", response_text, re.IGNORECASE):
                    result["status"] = "Compliant"
        
        if lang == "ar":
            ref_matches = re.finditer(r'(?:المادة|البند|الفقرة|مادة|بند|فقرة)\s*[\d\u0660-\u0669]+(?:\s*إلى\s*[\d\u0660-\u0669]+)?', response_text)
            auth_matches = re.finditer(r'(?:الجهة|الهيئة|المحكمة|السلطة|وزارة|هيئة)[:\s]*(.*?)(?:\n|$)', response_text)
            topic_matches = re.finditer(r'(?:الموضوع|المجال|المحور)[:\s]*(.*?)(?:\n|$)', response_text)
        else:
            ref_matches = re.finditer(r'(?:Article|Clause|Section)\s*\d+(?:\s*to\s*\d+)?', response_text, re.IGNORECASE)
            auth_matches = re.finditer(r'(?:Authority|Court|Body|Ministry)[:\s]*(.*?)(?:\n|$)', response_text, re.IGNORECASE)
            topic_matches = re.finditer(r'(?:Topic|Field|Subject)[:\s]*(.*?)(?:\n|$)', response_text, re.IGNORECASE)
            
        result["references"] = list(set([m.group(0) for m in ref_matches]))
        result["relevant_authorities"] = list(set([m.group(1).strip() for m in auth_matches if m.group(1)]))
        result["related_topics"] = list(set([m.group(1).strip() for m in topic_matches if m.group(1)]))
        
        if lang == "ar":
            issues_pattern = r'(?:المشاكل|المخالفات|المسائل)[:\s]*([\s\S]+?)(?=التوصيات|المراجع|$)'
            rec_pattern = r'(?:التوصيات|المقترحات)[:\s]*([\s\S]+?)(?=المراجع|$)'
        else:
            issues_pattern = r'(?:Issues|Problems|Concerns)[:\s]*([\s\S]+?)(?=Recommendations|References|$)'
            rec_pattern = r'(?:Recommendations|Suggestions)[:\s]*([\s\S]+?)(?=References|$)'
            
        if issues_match := re.search(issues_pattern, response_text, re.IGNORECASE):
            issues_text = issues_match.group(1).strip()
            result["issues"] = [i.strip() for i in re.split(r'[\n•*-]+', issues_text) if i.strip()]
            
        if rec_match := re.search(rec_pattern, response_text, re.IGNORECASE):
            rec_text = rec_match.group(1).strip()
            result["recommendations"] = [r.strip() for r in re.split(r'[\n•*-]+', rec_text) if r.strip()]
            
        return result
        
    except Exception as e:
        logger.error(f"Compliance parsing error: {e}")
        return result

def check_legal_compliance(
    section_text: str,
    similar_chunks: List[Dict],
    lang: str
) -> Dict:
    if not section_text.strip():
        return {
            "status": "Empty",
            "analysis": "Empty section - nothing to analyze",
            "issues": [],
            "recommendations": [],
            "references": [],
            "relevant_authorities": [],
            "related_topics": []
        }

    try:
        # 1. Prioritize and filter chunks
        def chunk_priority(chunk):
            return (
                chunk.get('similarity', 0),  # Higher similarity first
                -len(chunk['content'])     # Shorter content first
            )

        # Take top chunks by priority and similarity
        sorted_chunks = sorted(similar_chunks, key=chunk_priority, reverse=True)[:MAX_REFERENCES]
        
        # 2. Build optimized reference texts
        reference_texts = []
        seen_laws = set()
        
        for chunk in sorted_chunks:
            law_key = chunk.get('law_title') or chunk.get('law') or 'Unknown Law'
            if law_key in seen_laws:
                continue
                
            seen_laws.add(law_key)
            
            # Extract key elements
            article = chunk.get('article', extract_article_number(chunk['content']))
            authority = chunk.get('authority', 'Unknown')
            topic = chunk.get('topic', '')
            
            # Smart content truncation - keep beginning and relevant parts
            content = chunk['content']
            if len(content) > 800:
                # Keep first 300 chars and last 300 chars with separator
                content = f"{content[:300]} [...] {content[-300:]}"
            
            ref_text = (
                f"🔹 {law_key}\n"
                f"Authority: {authority}\n"
                f"Article: {article}\n"
                f"Relevance: {chunk.get('similarity', 0):.2f}\n"
                f"Excerpt:\n{content}"
            )
            reference_texts.append(ref_text)
            
            # Stop if we have enough quality references
            if len(reference_texts) >= 10:  # Reduced from MAX_REFERENCES
                break

        # 3. Prepare the section text with key context
        analyzed_text = section_text[:2500]  # Slightly reduced from 3000
        
        # 4. Structured prompt with clear sections
        if lang == "ar":
            prompt = f"""
            **التحليل القانوني المتعمق**

            **النص المراد تحليله:**
            {analyzed_text}

            **أهم المراجع القانونية ({len(reference_texts)} مراجع):**
            {reference_texts[0]} [...] [+{len(reference_texts)-1} مراجع أخرى]

            **تعليمات التحليل:**
            1. حدد حالة التوافق بدقة (متوافق تمامًا/متوافق جزئيًا/غير متوافق)
            2. قدم تحليلاً قانونياً مفصلاً مع ذكر المواد ذات الصلة
            3. اذكر المخالفات المحتملة إن وجدت
            4. قدم توصيات عملية لتحسين التوافق
            5. حدد الجهات الرقابية المختصة
            """
        else:
            prompt = f"""
            **Detailed Legal Analysis**

            **Text to Analyze:**
            {analyzed_text}

            **Key Legal References ({len(reference_texts)} references):**
            {reference_texts[0]} [...] [+{len(reference_texts)-1} more references]

            **Analysis Instructions:**
            1. Determine precise compliance status (Fully/Partially/Non-Compliant)
            2. Provide detailed legal analysis citing relevant articles
            3. List potential violations if any
            4. Offer practical recommendations
            5. Identify relevant regulatory bodies
            """

        # 5. API call with better token management
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "You're a legal expert providing detailed analysis. Be thorough but concise where possible."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=0.3,  # Slightly higher for better analysis
            max_tokens=1800,  # Balanced token count
            top_p=0.9
        )

        return parse_compliance_response(response.choices[0].message.content, lang)

    except Exception as e:
        logger.error(f"Compliance check error: {e}")
        return {
            "status": "Error",
            "analysis": f"Analysis failed: {str(e)}",
            "issues": ["Technical error occurred"],
            "recommendations": ["Please try with a smaller text section", "Consult legal expert if critical"],
            "references": [],
            "relevant_authorities": [],
            "related_topics": []
        }
        return parse_compliance_response(response.choices[0].message.content, lang)
        
    except Exception as e:
        logger.error(f"Compliance check error: {e}")
        return {
            "status": "Error",
            "analysis": f"Analysis failed: {str(e)}",
            "issues": [],
            "recommendations": ["Consult a legal professional for review"],
            "references": [],
            "relevant_authorities": [],
            "related_topics": []
        }

def format_retrieved_chunk(chunk: Dict) -> Dict:
    metadata = chunk.get('metadata', {})
    
    display_fields = {
        'law_title': metadata.get('law_title') or chunk.get('law_title') or chunk.get('name') or 'اتفاقية غير معنونة',
        'authority': metadata.get('authority') or chunk.get('authority') or 'غير محدد',
        'article': chunk.get('article') or extract_article_number(chunk['content']) or 'N/A',
        'country': metadata.get('country') or 'غير محدد',
        'doc_type': metadata.get('doc_type') or 'غير محدد',
        'effective_date': metadata.get('effective_date'),
        'signatories': metadata.get('signatories')
    }
    
    return {
        **chunk,
        'display': display_fields,
        'formatted_content': chunk['content'][:500] + ('...' if len(chunk['content']) > 500 else '')
    }

def generate_chunk_id(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]

def reset_connection(conn):
    if conn:
        try:
            if not conn.closed:
                try:
                    conn.rollback()
                except:
                    pass
                try:
                    with conn.cursor() as cur:
                        cur.execute("RESET ALL")
                except:
                    pass
            return conn
        except Exception as e:
            logger.error(f"Error resetting connection: {e}")
            return_db_connection(conn)
            return get_db_connection()
    return conn 