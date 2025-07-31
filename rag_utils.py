import os
import re
import fitz
import mammoth
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
from openai import OpenAI
import cohere
import numpy as np
import psycopg2
import psycopg2.extras
import json
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

MAX_SECTION_LENGTH = 3000
MAX_REFERENCE_LENGTH = 1000
MAX_REFERENCES = 3
MAX_TOKENS_SAFE = 7000
MAX_WORDS_PER_CHUNK = 300
MIN_WORDS_PER_CHUNK = 50
ARABIC_HEADER_THRESHOLD = 0.7
EMBEDDING_DIM = 2000

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_text_from_docx(path: str) -> str:
    with open(path, "rb") as docx_file:
        result = mammoth.convert_to_markdown(docx_file, ignore_empty_paragraphs=True)
        return result.value

def extract_text_from_pdf(path: str) -> str:
    try:
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) > 100:
                return text
        images = convert_from_path(path)
        ocr_text = "\n".join(pytesseract.image_to_string(img, lang='ara+eng') for img in images)
        return gpt_clean_text(ocr_text)
    except Exception:
        images = convert_from_path(path)
        ocr_text = "\n".join(pytesseract.image_to_string(img, lang='ara+eng') for img in images)
        return gpt_clean_text(ocr_text)

def extract_text_from_image(path: str) -> str:
    ocr_text = pytesseract.image_to_string(Image.open(path), lang='ara+eng')
    return gpt_clean_text(ocr_text)

def gpt_clean_text(raw_text: str) -> str:
    prompt = (
        "You are an expert at cleaning up OCR-extracted legal documents. "
        "Fix spelling errors, correct any encoding artifacts (especially in Arabic like اإل to الإ or لا), and make the text readable. "
        "Keep all Arabic or English words as-is, just fix broken words and characters. "
        "Do not remove any important legal information or section markers.\n\n"
        "Text to clean:\n"
        f"{raw_text[:4000]}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT Clean Error]: {e}")
        return raw_text

def detect_language(text: str) -> str:
    try:
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = max(len(re.sub(r'\s', '', text)), 1)
        if arabic_chars / total_chars > 0.3:
            return "ar"
        return detect(text) or "en"
    except:
        return "en"

def detect_country(text: str) -> str:
    country_keywords = {
        "لبنان": "LB", "المملكة العربية السعودية": "SA", 
        "دولة الإمارات": "AE", "مصر": "EG",
        "العراق": "IQ", "الأردن": "JO"
    }
    for keyword, code in country_keywords.items():
        if keyword in text[:2000]:
            return code
    prompt = """Analyze this legal document excerpt and return ONLY the 2-letter ISO country code:
LB (Lebanon), SA (Saudi Arabia), AE (UAE), EG (Egypt), IQ (Iraq), JO (Jordan), or XX if uncertain.
Document excerpt: """ + text[:2000]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        return result if result in ["LB", "SA", "AE", "EG", "IQ", "JO"] else "XX"
    except:
        return "XX"

def detect_doc_type(text: str) -> str:
    arabic_types = {
        "عقد": "Contract", "إيجار": "Lease Agreement",
        "عمل": "Employment Agreement", "شروط الخدمة": "Terms of Service",
        "سياسة الخصوصية": "Privacy Policy", "حكم": "Court Decision",
        "تشريع": "Legislation", "مذكرة قانونية": "Legal Memo"
    }
    for arabic_type, english_type in arabic_types.items():
        if arabic_type in text[:1000]:
            return english_type
    prompt = """Classify this legal document into one type:
- Contract
- Lease Agreement
- Employment Agreement
- Terms of Service
- Privacy Policy
- Court Decision
- Legislation
- Legal Memo
- Other
Provide ONLY the type name. Document excerpt: """ + text[:3000]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return "Unknown"

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

# --- Footnote-aware splitting for legal references ---

def split_legal_references_section(chunk):
    """
    Splits a 'النصوص القانونية' section into separate chunks for each article/provision,
    and attaches the corresponding footnote to each article if found.
    """
    header = chunk['header']
    content = chunk['content']

    # Find footnotes at the end (lines starting with numbers or [n])
    # Assume footnotes start at the last group of "digit. " or "[digit]"
    lines = content.strip().splitlines()
    footnote_start = None
    for i, line in enumerate(lines[::-1]):
        if re.match(r'^\s*(\d+|\[\d+\])[\.\)]?\s', line):
            footnote_start = len(lines) - 1 - i
    if footnote_start is not None and footnote_start > 0:
        content_main = "\n".join(lines[:footnote_start])
        footnotes_block = "\n".join(lines[footnote_start:])
        footnote_lines = re.findall(
            r'^\s*(\d+|\[\d+\])[\.\)]?\s*(.*?)(?=\n\d+[\.\)]|\n\[\d+\]|$)',
            footnotes_block, re.MULTILINE)
        footnote_map = {str(int(re.sub(r'\D', '', num))): text.strip()
                        for num, text in footnote_lines}
    else:
        content_main = content
        footnote_map = {}

    # Split main content into articles (المادة n)
    article_splits = re.split(r'(?=__المادة\s+\d+)', content_main)
    new_chunks = []
    for article in article_splits:
        article = article.strip()
        if not article:
            continue
        # Find article number for footnote mapping
        match = re.search(r'__المادة\s+(\d+)', article)
        article_num = match.group(1) if match else None
        # Attach corresponding footnote(s)
        footnote_text = ""
        if article_num and article_num in footnote_map:
            footnote_text = "\n\n[Footnote] " + footnote_map[article_num]
        new_chunks.append({
            "header": f"{header} - المادة {article_num}" if article_num else header,
            "level": chunk.get("level", 2),
            "content": article + footnote_text,
            "is_toc": False
        })
    # If nothing was split, fallback to returning the original
    return new_chunks if new_chunks else [chunk]

def gpt_chunk_document(text: str) -> Optional[List[Dict]]:
    prompt = (
        "قسم المستند القانوني التالي إلى أقسام منطقية.\n"
        "لكل قسم أعد:\n"
        "- header: عنوان القسم\n"
        "- content: نص القسم\n"
        "أعد النتيجة كـ JSON array مثل: [{\"header\": \"...\", \"content\": \"...\"}, ...]\n\n"
        f"المستند:\n{text[:60000]}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return json.loads(raw)
    except Exception as e:
        print("[GPT Chunking error]", e)
        return None

def chunk_legal_document_rule_based(text: str) -> List[Dict]:
    text = text.replace('\r\n', '\n').replace('\u2028', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    chunks = []
    current_chunk = []
    current_header = "Document Preamble"
    current_header_level = 0
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#'):
            if current_chunk:
                chunks.append({
                    "header": clean_section_title(current_header),
                    "level": current_header_level,
                    "content": "\n".join(current_chunk).strip()
                })
                current_chunk = []
            current_header_level = line.count('#')
            header_text = line.split(maxsplit=1)[1].strip() if ' ' in line else line[1:].strip()
            current_header = clean_section_title(header_text)
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
                    "level": current_header_level,
                    "content": "\n".join(current_chunk).strip()
                })
                current_chunk = []
            current_header = line.strip()
            current_header_level = 2 if "الجزء" in line or "الباب" in line else 3
            i += 1
            continue
        if i + 1 < len(lines) and re.match(r'^[-=]{3,}$', lines[i+1].strip()):
            if current_chunk:
                chunks.append({
                    "header": clean_section_title(current_header),
                    "level": current_header_level,
                    "content": "\n".join(current_chunk).strip()
                })
                current_chunk = []
            current_header = line.strip()
            current_header_level = 1 if '=' in lines[i+1] else 2
            i += 2
            continue
        if line:
            current_chunk.append(line)
        i += 1
    if current_chunk:
        chunks.append({
            "header": clean_section_title(current_header),
            "level": current_header_level,
            "content": "\n".join(current_chunk).strip()
        })
    processed_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        word_count = len(content.split())
        if word_count == 0:
            continue
        if word_count > MAX_WORDS_PER_CHUNK * 3:
            paragraphs = re.split(r'\n{2,}', content)
            current_para_group = []
            current_word_count = 0
            for para in paragraphs:
                para_word_count = len(para.split())
                if current_para_group and current_word_count + para_word_count > MAX_WORDS_PER_CHUNK:
                    processed_chunks.append({
                        "header": chunk["header"],
                        "level": chunk["level"],
                        "content": "\n\n".join(current_para_group),
                        "is_toc": False
                    })
                    current_para_group = []
                    current_word_count = 0
                current_para_group.append(para)
                current_word_count += para_word_count
            if current_para_group:
                processed_chunks.append({
                    "header": chunk["header"],
                    "level": chunk["level"],
                    "content": "\n\n".join(current_para_group),
                    "is_toc": False
                })
        else:
            processed_chunks.append(chunk)
    return processed_chunks

def chunk_legal_document_semantic(text: str) -> List[Dict]:
    try:
        if len(text) < 60000:
            gpt_chunks = gpt_chunk_document(text)
            if gpt_chunks and len(gpt_chunks) > 1:
                for ch in gpt_chunks:
                    if "header" not in ch: ch["header"] = ""
                    if "content" not in ch: ch["content"] = ""
                    ch["header"] = clean_section_title(ch["header"])
                    ch["level"] = 2
                    ch["is_toc"] = False
                original_chunks = gpt_chunks
            else:
                original_chunks = chunk_legal_document_rule_based(text)
        else:
            original_chunks = chunk_legal_document_rule_based(text)

        # Split and fix "النصوص القانونية" section: split articles and attach footnotes
        processed_chunks = []
        for chunk in original_chunks:
            if "النصوص القانونية" in chunk["header"]:
                split_chunks = split_legal_references_section(chunk)
                processed_chunks.extend(split_chunks)
            else:
                processed_chunks.append(chunk)
        return processed_chunks

    except Exception as e:
        print(f"Critical chunking error: {str(e)}")
        return [{
            "header": "Error",
            "level": 1,
            "content": f"Chunking failed: {str(e)}",
            "is_toc": False
        }]

# --- Embedding/Retrieval/Compliance logic below  ---

def embed_text(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    try:
        res = cohere_client.embed(
            texts=[text],
            model="embed-multilingual-v3.0",
            input_type="search_document"
        )
        emb = res.embeddings[0]
        if len(emb) < dim:
            return emb + [0.0] * (dim - len(emb))
        return emb[:dim]
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return [0.0] * dim

def retrieve_top_similar_chunks(
    embedding: List[float], 
    country_code: str, 
    doc_type: str = None,
    top_k: int = 50, 
    min_sim: float = 0.7
) -> List[Dict]:
    conn = psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT")
    )
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        embedding = embedding[:EMBEDDING_DIM] + [0.0] * (EMBEDDING_DIM - len(embedding))
        sql = """
        SELECT 
            c.id AS chunk_id,
            c.text AS chunk_text,
            d.name AS document_name,
            d.folder_path,
            'Unknown' AS document_type,
            1 - (e.vector <#> %s::vector) AS similarity
        FROM embedding e
        JOIN chunk c ON e.chunk_id = c.id
        JOIN document_part dp ON c.document_part_id = dp.id
        JOIN document d ON dp.document_id = d.id
        JOIN dataset ds ON c.dataset_id = ds.id
        WHERE ds.priority = 1
          AND ds.country_code = %s
          AND 1 - (e.vector <#> %s::vector) >= %s
        ORDER BY similarity DESC
        LIMIT %s;
        """
        cur.execute(sql, [embedding, country_code, embedding, min_sim, top_k])
        rows = cur.fetchall()
        seen = set()
        unique_results = []
        for r in rows:
            if r['chunk_id'] not in seen:
                seen.add(r['chunk_id'])
                unique_results.append(r)
        return unique_results[:top_k]
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []
    finally:
        conn.close()

def analyze_compliance_status(response_text: str) -> Dict[str, Union[str, List[str]]]:
    result = {
        "status": "Unknown",
        "issues": [],
        "recommendations": []
    }
    if not response_text:
        return result
    try:
        response_text = re.sub(r'<[^>]+>', '', response_text)
        status_patterns = {
            "Compliant": [
                r"(?:Compliance Status:|حالة التوافق:)\s*(Compliant|متوافق)",
                r"(?:is|are) (?:fully )?compliant",
                r"(?:fully|completely) (?:compliant|in compliance)",
                r"(?:no|zero) compliance issues",
                r"لا يوجد مشاكل في الامتثال",
                r"مطابق لجميع المتطلبات"
            ],
            "Partially Compliant": [
                r"(?:Compliance Status:|حالة التوافق:)\s*(Partially Compliant|متوافق جزئيًا)",
                r"partial(?:ly)? compliant",
                r"some compliance issues",
                r"بعض مشاكل الامتثال",
                r"يتوافق جزئيا"
            ],
            "Non-Compliant": [
                r"(?:Compliance Status:|حالة التوافق:)\s*(Non-Compliant|غير متوافق)",
                r"(?:is|are) not compliant",
                r"(?:does|do) not comply",
                r"fails to comply",
                r"يخالف المتطلبات",
                r"غير مطابق"
            ]
        }
        for status, patterns in status_patterns.items():
            if any(re.search(pattern, response_text, re.IGNORECASE) for pattern in patterns):
                result["status"] = status
                break
        if result["status"] == "Unknown":
            positive_terms = ["meets requirements", "compliant", "متوافق", "مطابق", "صحيح"]
            negative_terms = ["non-compliant", "violation", "مخالفة", "غير مطابق", "خطأ"]
            positive_count = sum(response_text.lower().count(term) for term in positive_terms)
            negative_count = sum(response_text.lower().count(term) for term in negative_terms)
            if positive_count > negative_count:
                result["status"] = "Compliant"
            elif negative_count > positive_count:
                result["status"] = "Non-Compliant"
        issues_pattern = r'(?:Key Issues:|المشاكل الرئيسية:)\s*([\s\S]+?)(?=Recommendations:|التوصيات:|$)'
        rec_pattern = r'(?:Recommendations:|التوصيات:)\s*([\s\S]+)$'
        if issues_match := re.search(issues_pattern, response_text, re.IGNORECASE):
            issues_text = issues_match.group(1).strip()
            issues_text = re.sub(r'^[\s•*-]+', '', issues_text)
            result["issues"] = [i.strip() for i in re.split(r'[\n•*-]+', issues_text) if i.strip()]
        if rec_match := re.search(rec_pattern, response_text, re.IGNORECASE):
            rec_text = rec_match.group(1).strip()
            rec_text = re.sub(r'^[\s•*-]+', '', rec_text)
            result["recommendations"] = [r.strip() for r in re.split(r'[\n•*-]+', rec_text) if r.strip()]
        if "النصوص القانونية" in response_text and result["status"] == "Non-Compliant":
            if "لا يوجد مخالفات" in response_text or "no violations" in response_text:
                result["status"] = "Compliant"
    except Exception as e:
        print(f"Error parsing compliance response: {str(e)}")
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
            "recommendations": []
        }
    word_count = len(section_text.split())
    if word_count > 2500:
        parts = []
        current_part = []
        current_word_count = 0
        for paragraph in re.split(r'\n{2,}', section_text):
            para_word_count = len(paragraph.split())
            if current_part and current_word_count + para_word_count > 1000:
                parts.append(" ".join(current_part))
                current_part = []
                current_word_count = 0
            current_part.append(paragraph)
            current_word_count += para_word_count
        if current_part:
            parts.append(" ".join(current_part))
        analyses = []
        for part in parts[:3]:
            analysis = perform_compliance_analysis(part, similar_chunks, lang)
            analyses.append(analysis)
        status = "Partially Compliant"
        if all(a["status"] == "Compliant" for a in analyses):
            status = "Compliant"
        elif all(a["status"] == "Non-Compliant" for a in analyses):
            status = "Non-Compliant"
        return {
            "status": status,
            "analysis": "Large section analysis (partial):\n" + "\n\n".join(a["analysis"] for a in analyses),
            "issues": list(set(issue for a in analyses for issue in a["issues"])),
            "recommendations": [
                "Full section was too large for complete analysis",
                *list(set(rec for a in analyses for rec in a["recommendations"]))
            ],
            "partial_analysis": True,
            "parts_analyzed": len(analyses),
            "total_parts": len(parts)
        }
    return perform_compliance_analysis(section_text, similar_chunks, lang)

def perform_compliance_analysis(
    text: str,
    similar_chunks: List[Dict],
    lang: str
) -> Dict:
    truncated_section = text[:MAX_SECTION_LENGTH]
    similar_blocks = []
    for c in similar_chunks[:MAX_REFERENCES]:
        truncated_chunk = c['chunk_text'][:MAX_REFERENCE_LENGTH]
        doc_type = c.get('document_type', 'N/A')
        similar_blocks.append(
            f"📄 {truncated_chunk}\n🔗 Source: {c['document_name']} ({doc_type})"
        )
    instructions = {
        "en": """Analyze this legal section for compliance with relevant laws. Provide:
1. Clear Compliance Status (Compliant/Partially Compliant/Non-Compliant)
2. Specific Legal Issues (with references to laws if possible)
3. Detailed Analysis
4. Actionable Recommendations

If the section is simply quoting laws without modifications (like "النصوص القانونية"), 
it should typically be considered Compliant unless specific issues are identified.""",
        "ar": """حلل هذا القسم القانوني للتأكد من امتثاله للقوانين ذات الصلة. قدم:
1. حالة توافق واضحة (متوافق/متوافق جزئيًا/غير متوافق)
2. مشاكل قانونية محددة (مع الإشارة إلى القوانين إذا أمكن)
3. تحليل مفصل
4. توصيات قابلة للتنفيذ

إذا كان القسم ينقل نصوصًا قانونية دون تعديل (مثل "النصوص القانونية")، 
فيجب اعتباره متوافقًا ما لم يتم تحديد مشاكل محددة."""
    }.get(lang, "en")
    prompt = "\n".join([
        instructions,
        "\n📜 Section Being Analyzed:",
        truncated_section,
        "\n🔎 Similar Legal References:",
        "\n\n---\n\n".join(similar_blocks)
    ])
    try:
        token_estimate = len(prompt.split()) * 4 // 3
        model = "gpt-4-1106-preview" if token_estimate > 3000 else "gpt-4"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        analysis = response.choices[0].message.content.strip()
        result = analyze_compliance_status(analysis)
        result["analysis"] = analysis
        if "النصوص القانونية" in truncated_section:
            if result["status"] == "Non-Compliant" and not result["issues"]:
                result["status"] = "Compliant"
                result["analysis"] += "\n\nNote: Legal texts are generally considered compliant unless specific issues are identified"
        return result
    except Exception as e:
        return {
            "status": "Error",
            "analysis": f"Error generating compliance analysis: {str(e)}",
            "issues": [],
            "recommendations": []
        }
