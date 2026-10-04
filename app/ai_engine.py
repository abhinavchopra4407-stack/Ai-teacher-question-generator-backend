import os
import json
import re
import uuid
import logging
import urllib.request
from typing import List, Dict, Any, Optional

from app.config import settings
from app.doc_processor import chunk_text_if_needed, clean_text

logger = logging.getLogger("ai_engine")

def call_groq_api(prompt: str, user_api_key: Optional[str] = None) -> Optional[str]:
    """Call Groq API or xAI Grok API using configured credentials."""
    api_key = user_api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or settings.GROQ_API_KEY
    if not api_key:
        logger.info("No Groq / Grok / xAI API key configured.")
        return None

    if api_key.startswith("xai-"):
        url = "https://api.x.ai/v1/chat/completions"
        models_to_try = ["grok-2-latest", "grok-beta", "grok-2-1212"]
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for model_name in models_to_try:
        logger.info(f"Attempting AI API call to {url} with model: {model_name}")
        data: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert educational assessment designer. Return ONLY valid JSON output."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        if not api_key.startswith("xai-"):
            data["response_format"] = {"type": "json_object"}

        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=35) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                content = res_body["choices"][0]["message"]["content"]
                if content and content.strip():
                    logger.info(f"AI API success with {model_name} (response length: {len(content)} chars)")
                    return content
        except Exception as e:
            logger.warning(f"AI API model {model_name} error: {e}")
            continue

    return None

def call_gemini_api_rest(prompt: str, user_api_key: Optional[str] = None) -> Optional[str]:
    """Call Google Gemini REST API with JSON response format."""
    api_key = user_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY
    if not api_key:
        return None

    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=35) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                candidates = res_body.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        text = parts[0]["text"]
                        if text and text.strip():
                            logger.info(f"Gemini REST API success with {model_name} (response length: {len(text)} chars)")
                            return text
        except Exception as e:
            logger.warning(f"Gemini REST API model {model_name} error: {e}")
            continue

    return None

def get_gemini_client(user_api_key: Optional[str] = None):
    """Obtain initialized Google GenAI client or key."""
    api_key = user_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY
    if not api_key:
        return None
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logger.warning(f"Could not initialize google.genai Client: {e}")
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            return genai_legacy
        except Exception as e2:
            logger.error(f"Failed to initialize generative AI SDK: {e2}")
            return None

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Clean markdown backticks and parse JSON safely."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No valid JSON found in AI response")

def extract_grounded_answer_from_text(question_text: str, topic: str, chapter_text: str, max_words: int = 80) -> str:
    """Extract exact relevant sentences from chapter_text that match the question or topic keywords, filtering out PDF metadata noise."""
    if not chapter_text or not chapter_text.strip():
        return f"Refer to the core concepts outlined in {topic}."

    cleaned = clean_text(chapter_text)
    # Remove chapter numbering like "1. ", "2. " from beginning of lines
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned, flags=re.MULTILINE)

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if len(s.strip()) > 15]
    
    # Filter out sentences that look like metadata/headers/page numbers
    filtered_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(bad in s_lower for bad in ["sample document", "testing pdf", "page 1", "page 2", "page 3", "page 4", "page 5", "page 6", "page 7", "page 8", "page 9", "page 10", "for project testing"]):
            continue
        # Skip sentences ending with a bare digit/page number like "The Clockmaker of Riverton 2." or "Garden Beyond the Stars 3."
        if re.search(r'\b\d+\s*\.?\s*$', s):
            continue
        # Skip title headers / section headings (short sentence without main verbs)
        words_in_s = s.split()
        if len(words_in_s) <= 8 and not any(v in s_lower for v in ["is", "was", "are", "were", "had", "have", "been", "came", "went", "stood", "looked", "found", "lived", "worked", "opened", "said", "stopped", "ran", "saw", "built", "turned", "smelled", "blamed", "spent", "entered", "placed", "asked", "changed", "sent", "spoke", "returned", "vanished", "leaving", "looked", "admitted", "remembered", "listed", "cleaned", "studied", "measured"]):
            continue
        if re.search(r'^\d+\.\s+[A-Z]', s): # Skip chapter headers like "1. The Shop at the End"
            continue
        filtered_sentences.append(s)

    if not filtered_sentences:
        filtered_sentences = sentences or [cleaned]

    stop_words = {"what", "is", "the", "define", "key", "term", "presented", "in", "chapter", "how", "why", "explain", "of", "and", "a", "to", "or", "regarding", "as", "described", "text", "which", "state", "outline", "list", "compare", "contrast", "page", "short", "story", "10", "page"}
    words = re.findall(r'\w+', (question_text + " " + topic).lower())
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]

    scored_sentences = []
    for s in filtered_sentences:
        s_lower = s.lower()
        score = sum(1 for kw in keywords if kw in s_lower)
        scored_sentences.append((score, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_matches = [s for score, s in scored_sentences if score > 0]
    
    if best_matches:
        answer_text = " ".join(best_matches[:2])
    else:
        answer_text = " ".join(filtered_sentences[:2])

    words_list = answer_text.split()
    if len(words_list) > max_words:
        return " ".join(words_list[:max_words]) + "..."
    return answer_text

def normalize_sections_config(sections: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize and filter enabled sections from input config."""
    active = []
    if sections:
        for idx, sec in enumerate(sections):
            if isinstance(sec, dict) and sec.get("enabled", True):
                sec_id = sec.get("id") or f"sec-{idx + 1}"
                name = sec.get("name") or f"Section {chr(65 + idx)}"
                q_type = sec.get("type") or name
                count = max(1, int(sec.get("question_count", 3)))
                marks = max(1, int(sec.get("marks_per_question", 2)))
                length = sec.get("expected_length") or ("1-10 words" if "Very Short" in q_type else ("40-60 words" if "Short" in q_type else "150-250 words"))
                diff = sec.get("difficulty") or "Medium"
                
                active.append({
                    "id": sec_id,
                    "name": name,
                    "type": q_type,
                    "enabled": True,
                    "question_count": count,
                    "marks_per_question": marks,
                    "expected_length": length,
                    "difficulty": diff
                })
    
    if not active:
        active = [
            {"id": "sec-1", "name": "Very Short Answer", "type": "Very Short Answer", "enabled": True, "question_count": 3, "marks_per_question": 2, "expected_length": "1-10 words", "difficulty": "Easy"},
            {"id": "sec-2", "name": "Short Answer", "type": "Short Answer", "enabled": True, "question_count": 3, "marks_per_question": 4, "expected_length": "40-60 words", "difficulty": "Medium"},
            {"id": "sec-3", "name": "Long Answer", "type": "Long Answer", "enabled": True, "question_count": 3, "marks_per_question": 8, "expected_length": "150-250 words", "difficulty": "Hard"}
        ]
    return active

def parse_and_validate_ai_response(
    raw_res: str,
    active_sections: List[Dict[str, Any]],
    chapter_title: str,
    chapter_text: str
) -> Optional[Dict[str, Any]]:
    """Parse raw AI JSON output and group questions into requested active sections."""
    try:
        parsed = extract_json_from_text(raw_res)
    except Exception as e:
        logger.warning(f"JSON parsing error: {e}")
        return None

    # Collect questions from JSON structure
    extracted_q_list = []
    if isinstance(parsed, dict):
        if "sections" in parsed and isinstance(parsed["sections"], list):
            for sec_obj in parsed["sections"]:
                if isinstance(sec_obj, dict) and "questions" in sec_obj and isinstance(sec_obj["questions"], list):
                    sec_name_val = sec_obj.get("section_name", "")
                    for q in sec_obj["questions"]:
                        if isinstance(q, dict):
                            if not q.get("section_name") and sec_name_val:
                                q["section_name"] = sec_name_val
                            extracted_q_list.append(q)
        else:
            for k in ["questions", "very_short_questions", "short_questions", "long_questions", "all_questions"]:
                if k in parsed and isinstance(parsed[k], list):
                    extracted_q_list.extend([q for q in parsed[k] if isinstance(q, dict)])
    elif isinstance(parsed, list):
        extracted_q_list = [q for q in parsed if isinstance(q, dict)]

    if not extracted_q_list:
        return None

    # Clean & validate individual questions
    valid_questions = []
    seen_texts = set()
    for q in extracted_q_list:
        q_text = q.get("question_text", "").strip()
        if not q_text or len(q_text) < 8 or q_text.lower() in seen_texts:
            continue
        seen_texts.add(q_text.lower())
        
        topic = q.get("related_topic", "").strip()
        if not topic or topic.lower() in ["overview", "general", "none", "chapter overview"]:
            q["related_topic"] = f"Topic: {q_text[:25]}"
            
        ans = q.get("answer", "").strip()
        if not ans or len(ans) > len(chapter_text) * 0.5:
            q["answer"] = extract_grounded_answer_from_text(q_text, topic or chapter_title, chapter_text, max_words=60)
            
        if not q.get("id"):
            q["id"] = str(uuid.uuid4())
            
        valid_questions.append(q)

    if not valid_questions:
        return None

    # Distribute valid questions among active sections
    all_final_questions = []
    vs_list = []
    sq_list = []
    lq_list = []
    global_num = 1
    unassigned_pool = list(valid_questions)

    for sec in active_sections:
        sec_name = sec["name"]
        sec_type = sec["type"]
        req_count = sec["question_count"]
        marks = sec["marks_per_question"]
        expected_len = sec.get("expected_length", "Standard")

        # Find matching questions for this section
        matched = []
        remaining_pool = []
        for q in unassigned_pool:
            q_sec = q.get("section_name", "")
            q_type = q.get("question_type", "")
            if (q_sec and q_sec.lower() == sec_name.lower()) or (q_type and q_type.lower() in sec_name.lower()) or (q_type and q_type.lower() in sec_type.lower()):
                matched.append(q)
            else:
                remaining_pool.append(q)
                
        unassigned_pool = remaining_pool

        # If matching couldn't satisfy count, take from unassigned pool
        while len(matched) < req_count and unassigned_pool:
            matched.append(unassigned_pool.pop(0))

        # Assign section attributes to matched questions
        for q in matched[:req_count]:
            q["question_number"] = global_num
            q["section_name"] = sec_name
            q["question_type"] = sec_type
            q["marks"] = marks
            q["expected_length"] = q.get("expected_length") or expected_len
            if not q.get("marking_points"):
                q["marking_points"] = [f"Correct assessment & response ({marks} marks)"]

            all_final_questions.append(q)
            if "Very Short" in sec_type:
                vs_list.append(q)
            elif "Short" in sec_type:
                sq_list.append(q)
            else:
                lq_list.append(q)

            global_num += 1

    total_expected = sum(s["question_count"] for s in active_sections)
    if len(all_final_questions) < total_expected:
        logger.warning(f"Parsed {len(all_final_questions)} questions, but {total_expected} were requested.")
        return None

    return {
        "very_short_questions": vs_list,
        "short_questions": sq_list,
        "long_questions": lq_list,
        "all_questions": all_final_questions,
        "sections": active_sections
    }

def generate_questions_from_chapter(
    chapter_title: str,
    chapter_text: str,
    subject: str,
    grade: str,
    board: str = "General",
    language: str = "English",
    difficulty: str = "Medium",
    sections: Optional[List[Dict[str, Any]]] = None,
    marks_dist: Optional[Dict[str, int]] = None,
    special_instructions: str = "",
    user_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core function to generate questions dynamically based on configured sections.
    """
    active_sections = normalize_sections_config(sections)
    chunked_text = chunk_text_if_needed(chapter_text, max_words=4000)
    word_count = len(chapter_text.split())
    total_requested_questions = sum(s["question_count"] for s in active_sections)

    logger.info(f"Generating {total_requested_questions} questions across {len(active_sections)} sections for '{chapter_title}'.")

    sec_descriptions = []
    for idx, s in enumerate(active_sections, start=1):
        sec_descriptions.append(
            f"Section {idx}: \"{s['name']}\" (Type: {s['type']}, Questions Needed: EXACTLY {s['question_count']}, {s['marks_per_question']} marks each, Length: {s.get('expected_length', 'Standard')})"
        )
    sec_prompt_str = "\n".join(sec_descriptions)

    prompt = f"""
You are an expert educational curriculum designer and question paper creator.
Analyze the provided chapter text and generate EXACTLY {total_requested_questions} questions across the following {len(active_sections)} sections based ONLY on the provided text.

CHAPTER METADATA:
- Chapter Title: {chapter_title}
- Subject: {subject}
- Grade/Class: {grade}
- Education Board: {board}
- Target Language: {language}
- Overall Difficulty Level: {difficulty}
- Special Instructions: {special_instructions or "None"}

SECTIONS REQUIRED:
{sec_prompt_str}

RULES:
- Base every single question and model answer directly on the provided chapter text below. Do NOT invent facts or characters.
- Generate EXACTLY the requested question count for each section.
- Every single question MUST have its OWN UNIQUE model answer answering that question and specific marking scheme.
- Return ONLY a valid JSON object matching the EXACT JSON structure below.

CHAPTER TEXT CONTENT:
\"\"\"
{chunked_text}
\"\"\"

REQUIRED JSON OUTPUT FORMAT:
{{
  "sections": [
    {{
      "section_name": "{active_sections[0]['name']}",
      "questions": [
        {{
          "question_number": 1,
          "question_text": "...",
          "question_type": "{active_sections[0]['type']}",
          "difficulty": "{difficulty}",
          "marks": {active_sections[0]['marks_per_question']},
          "related_topic": "Specific Topic Name",
          "answer": "Concise direct answer.",
          "marking_points": ["Correct response ({active_sections[0]['marks_per_question']} marks)"],
          "expected_length": "{active_sections[0].get('expected_length', 'Standard')}"
        }}
      ]
    }}
  ]
}}
"""

    # 1. Try Groq / xAI API Provider
    for attempt in range(2):
        res_text = call_groq_api(prompt, user_api_key)
        if res_text:
            result = parse_and_validate_ai_response(res_text, active_sections, chapter_title, chapter_text)
            if result:
                logger.info(f"Successfully generated {len(result['all_questions'])} questions via Groq/xAI API!")
                return result

    # 2. Try Gemini REST API Provider
    res_text = call_gemini_api_rest(prompt, user_api_key)
    if res_text:
        result = parse_and_validate_ai_response(res_text, active_sections, chapter_title, chapter_text)
        if result:
            logger.info(f"Successfully generated {len(result['all_questions'])} questions via Gemini REST API!")
            return result

    # 3. Try Gemini SDK Client Provider
    client = get_gemini_client(user_api_key)
    if client:
        try:
            logger.info("Calling Gemini API Provider SDK...")
            raw_response = ""
            if hasattr(client, "models"):
                response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                raw_response = response.text
            elif hasattr(client, "GenerativeModel"):
                model = client.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                raw_response = response.text
                
            result = parse_and_validate_ai_response(raw_response, active_sections, chapter_title, chapter_text)
            if result:
                logger.info(f"Successfully generated {len(result['all_questions'])} questions via Gemini SDK!")
                return result
        except Exception as e:
            logger.error(f"Gemini API SDK error: {e}")

    # 4. Smart Semantic Fallback Question Generator
    logger.info("Executing Smart Semantic Fallback Question Generator for dynamic sections...")
    return generate_fallback_questions(
        chapter_title=chapter_title,
        chapter_text=chapter_text,
        subject=subject,
        grade=grade,
        language=language,
        difficulty=difficulty,
        sections=active_sections
    )

def generate_single_replacement_question(
    chapter_title: str,
    chapter_text: str,
    subject: str,
    grade: str,
    question_type: str,
    existing_question: str,
    topic: str = "",
    language: str = "English",
    difficulty: str = "Medium",
    special_instructions: str = "",
    user_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a single new question to replace an existing one."""
    marks = 2 if "Very Short" in question_type else (4 if "Short" in question_type else 8)
    chunked_text = chunk_text_if_needed(chapter_text, max_words=3000)
    
    prompt = f"""
You are an expert question paper author.
Generate a NEW replacement question of type '{question_type}' for the chapter '{chapter_title}' in {subject} ({grade}). Include a comprehensive model answer and marking scheme.

REQUIREMENTS:
- Do NOT generate this existing question again: "{existing_question}"
- Language: {language}
- Difficulty: {difficulty}
- Marks: {marks}
- Must be based ONLY on the provided chapter text below.
- Return ONLY a JSON object:

{{
  "id": "{str(uuid.uuid4())}",
  "question_number": 1,
  "question_text": "...",
  "question_type": "{question_type}",
  "difficulty": "{difficulty}",
  "marks": {marks},
  "related_topic": "Specific Topic Name",
  "answer": "Detailed model answer specifically addressing the question.",
  "marking_points": ["Point 1 (1 mark)", "Point 2 (1 mark)"],
  "expected_length": "{'1-10 words' if 'Very Short' in question_type else ('40-60 words' if 'Short' in question_type else '150-250 words')}"
}}

CHAPTER TEXT:
{chunked_text}
"""
    # 1. Groq API
    groq_res = call_groq_api(prompt, user_api_key)
    if groq_res:
        try:
            q_data = extract_json_from_text(groq_res)
            q_data["id"] = str(uuid.uuid4())
            q_data["marks"] = marks
            q_data["question_type"] = question_type
            if not q_data.get("answer"):
                q_data["answer"] = extract_grounded_answer_from_text(q_data.get("question_text", ""), topic or chapter_title, chapter_text)
            return q_data
        except Exception as e:
            logger.warning(f"Groq single question parse error: {e}")

    # 2. Gemini API
    client = get_gemini_client(user_api_key)
    if client:
        try:
            if hasattr(client, "models"):
                raw = client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
            elif hasattr(client, "GenerativeModel"):
                raw = client.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text
            
            q_data = extract_json_from_text(raw)
            q_data["id"] = str(uuid.uuid4())
            q_data["marks"] = marks
            q_data["question_type"] = question_type
            if not q_data.get("answer"):
                q_data["answer"] = extract_grounded_answer_from_text(q_data.get("question_text", ""), topic or chapter_title, chapter_text)
            return q_data
        except Exception as e:
            logger.error(f"Gemini single question error: {e}")

    # 3. Grounded Fallback
    q_topic = topic or chapter_title
    q_text = f"Explain the core principle of {q_topic} discussed in the chapter." if "Long" in question_type else (f"Briefly describe the significance of {q_topic}." if "Short" in question_type else f"Define {q_topic} according to the text.")
    q_ans = extract_grounded_answer_from_text(q_text, q_topic, chapter_text)
    
    return {
        "id": str(uuid.uuid4()),
        "question_number": 1,
        "question_text": q_text,
        "question_type": question_type,
        "difficulty": difficulty,
        "marks": marks,
        "related_topic": q_topic,
        "answer": q_ans,
        "marking_points": [f"Correct assessment ({marks} marks)"],
        "expected_length": "1-10 words" if "Very Short" in question_type else ("40-60 words" if "Short" in question_type else "150-250 words")
    }

def generate_answer_key_for_questions(
    chapter_title: str,
    chapter_text: str,
    questions: List[Dict[str, Any]],
    language: str = "English",
    user_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate comprehensive answer keys with marking points and expected length strictly mapped by ID."""
    chunked_text = chunk_text_if_needed(chapter_text, max_words=3500)
    
    questions_summary = []
    for q in questions:
        questions_summary.append({
            "id": q.get("id"),
            "question_number": q.get("question_number"),
            "question_type": q.get("question_type"),
            "question_text": q.get("question_text"),
            "marks": q.get("marks")
        })
        
    prompt = f"""
You are a senior teacher creating an official Answer Key and Marking Scheme for an exam paper on '{chapter_title}'.

RULES FOR ANSWER GENERATION:
- Generate detailed, complete, highly accurate model answers for EACH question below based strictly on the provided chapter text below.
- Do NOT use generic placeholder text or dump whole paragraphs of chapter text.
- Every model answer MUST directly answer its specific question text.
- Very Short answers must be 1-10 words. Short answers must be 40-60 words. Long answers must be 150-250 words.
- Marking scheme points MUST specify mark allocations (e.g. ["Correct definition (1 mark)", "Explanation of mechanism (1 mark)"]) that sum exactly to the question's total marks.

Target Language: {language}

QUESTIONS LIST:
{json.dumps(questions_summary, indent=2)}

CHAPTER TEXT:
{chunked_text}

OUTPUT FORMAT: Return ONLY a JSON object with a single top-level key "answer_keys" containing a list of answer objects corresponding to each question ID:
{{
  "answer_keys": [
    {{
      "id": "<question_id>",
      "question_number": 1,
      "answer": "<direct specific model answer answering question 1>",
      "marking_points": ["Point 1 (1 mark)", "Point 2 (1 mark)"],
      "expected_length": "1-10 words"
    }}
  ]
}}
"""
    # 1. Try Groq API
    groq_res = call_groq_api(prompt, user_api_key)
    if groq_res:
        try:
            parsed = extract_json_from_text(groq_res)
            ans_list = parsed.get("answer_keys", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            if isinstance(ans_list, list) and len(ans_list) > 0:
                ans_dict = {item.get("id"): item for item in ans_list if isinstance(item, dict) and item.get("id")}
                for q in questions:
                    q_id = q.get("id")
                    if q_id in ans_dict and ans_dict[q_id].get("answer"):
                        q["answer"] = ans_dict[q_id].get("answer")
                        q["marking_points"] = ans_dict[q_id].get("marking_points", [])
                        q["expected_length"] = ans_dict[q_id].get("expected_length")
                    else:
                        q["answer"] = extract_grounded_answer_from_text(q.get("question_text", ""), q.get("related_topic", chapter_title), chapter_text, max_words=60)
                return questions
        except Exception as e:
            logger.warning(f"Groq answer key parse error: {e}")

    # 2. Try Gemini API
    client = get_gemini_client(user_api_key)
    if client:
        try:
            res_text = ""
            if hasattr(client, "models"):
                res_text = client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
            elif hasattr(client, "GenerativeModel"):
                res_text = client.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text
                
            parsed = extract_json_from_text(res_text)
            ans_list = parsed.get("answer_keys", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            if isinstance(ans_list, list) and len(ans_list) > 0:
                ans_dict = {item.get("id"): item for item in ans_list if isinstance(item, dict) and item.get("id")}
                for q in questions:
                    q_id = q.get("id")
                    if q_id in ans_dict and ans_dict[q_id].get("answer"):
                        q["answer"] = ans_dict[q_id].get("answer")
                        q["marking_points"] = ans_dict[q_id].get("marking_points", [])
                        q["expected_length"] = ans_dict[q_id].get("expected_length")
                    else:
                        q["answer"] = extract_grounded_answer_from_text(q.get("question_text", ""), q.get("related_topic", chapter_title), chapter_text, max_words=60)
                return questions
        except Exception as e:
            logger.error(f"Gemini answer key error: {e}")

    # 3. Grounded Specific Fallback Answers from Chapter Text (Mapped by Question)
    for q in questions:
        q_type = q.get("question_type", "")
        q_text = q.get("question_text", "")
        topic = q.get("related_topic", chapter_title)
        marks = q.get("marks", 2)
        
        extracted_ans = extract_grounded_answer_from_text(q_text, topic, chapter_text, max_words=60)
        q["answer"] = extracted_ans
        
        if "Very Short" in q_type:
            q["expected_length"] = "1 - 10 words"
            q["marking_points"] = [f"Correct definition/fact ({marks} marks)"]
        elif "Short" in q_type:
            p1 = marks // 2
            p2 = marks - p1
            q["expected_length"] = "40 - 60 words"
            q["marking_points"] = [f"Identification of concept ({p1} marks)", f"Explanation & context ({p2} marks)"]
        else:
            p1 = marks // 4
            p2 = marks // 2
            p3 = marks - p1 - p2
            q["expected_length"] = "150 - 250 words"
            q["marking_points"] = [
                f"Introduction & context ({p1} marks)",
                f"Detailed multi-step analysis ({p2} marks)",
                f"Conclusion & impact ({p3} marks)"
            ]
            
    return questions

def generate_fallback_questions(
    chapter_title: str,
    chapter_text: str,
    subject: str,
    grade: str,
    language: str = "English",
    difficulty: str = "Medium",
    sections: Optional[List[Dict[str, Any]]] = None,
    marks_dist: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Smart Semantic Fallback Generator:
    Divides chapter text into distinct paragraphs/sections and extracts unique questions 
    and answers dynamically based on active configured sections!
    """
    active_sections = normalize_sections_config(sections)
    cleaned_full = chapter_text.strip()

    clean_title = re.sub(r'\s*\d+\s*page\s*story.*$', '', chapter_title, flags=re.IGNORECASE)
    clean_title = re.sub(r'\.pdf$', '', clean_title, flags=re.IGNORECASE)
    clean_title = clean_title.replace('_', ' ').replace('-', ' ').strip().title() or "The Chapter"
    
    raw_paras = [p.strip() for p in re.split(r'\n\n+', cleaned_full) if len(p.strip()) > 30]
    if len(raw_paras) < 3:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_full) if len(s.strip()) > 15]
        raw_paras = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
        
    if not raw_paras:
        raw_paras = [cleaned_full]

    paragraphs = list(raw_paras)
    total_q_needed = sum(s["question_count"] for s in active_sections)
    if len(paragraphs) < total_q_needed:
        fine_paragraphs = []
        for p in paragraphs:
            s_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if len(s.strip()) > 15]
            if len(s_list) > 1:
                fine_paragraphs.extend(s_list)
            else:
                fine_paragraphs.append(p)
        if len(fine_paragraphs) >= 3:
            paragraphs = fine_paragraphs

    num_paras = len(paragraphs)

    def get_unique_section_topic(p_text: str, p_idx: int) -> str:
        p_clean = p_text.replace('\r', '').strip()
        lines = [l.strip() for l in p_clean.splitlines() if l.strip()]
        if lines:
            first_line = re.sub(r'^\d+[\.\:]\s*', '', lines[0]).strip()
            first_line = re.sub(r'\b\d+\b', '', first_line).strip()
            if 2 <= len(first_line.split()) <= 7 and not first_line.endswith('.') and not first_line.lower().startswith(("part", "chapter", "section", "page")):
                return first_line.title()
                
        prop_nouns = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', p_clean.replace('\n', ' '))
        skip = {"The Garden", "The Clockmaker", "The Lighthouse", "Very Short", "Short Answer", "Long Answer", "Moonbay"}
        valid_props = [p for p in prop_nouns if p not in skip]
        if valid_props:
            return valid_props[0].title()

        stop_words = {"chapter", "section", "story", "the", "this", "that", "where", "which", "would", "could", "should", "about", "testing", "stands", "found", "with", "from", "they", "there", "page", "when", "into", "through", "small", "carried", "inserted", "stated", "regarding", "significance", "described", "text", "narrative", "weeks", "followed", "people", "hill", "climbing", "midday", "last", "town", "listen", "choice", "stay", "light", "tomorrow", "northern", "edge", "seventy", "years", "guided", "modern", "satellite"}
        words = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{4,}\b', p_text)
        key_words = []
        for w in words:
            if w.lower() not in stop_words and w.lower() not in [k.lower() for k in key_words]:
                key_words.append(w.capitalize())
            if len(key_words) == 2:
                break
        if len(key_words) == 2:
            return f"{key_words[0]} & {key_words[1]} Overview"
        elif len(key_words) == 1:
            return f"{key_words[0]} Context"
        return f"Narrative Event {p_idx + 1}"

    all_questions = []
    vs_list = []
    sq_list = []
    lq_list = []

    global_q_num = 1
    para_cursor = 0

    for sec in active_sections:
        sec_name = sec["name"]
        sec_type = sec["type"]
        q_count = sec["question_count"]
        marks = sec["marks_per_question"]
        expected_len = sec.get("expected_length") or ("1 - 10 words" if "Very Short" in sec_type else ("40 - 60 words" if "Short" in sec_type else "150 - 250 words"))

        for i in range(q_count):
            p_idx = para_cursor % num_paras
            para_cursor += 1
            p_content = paragraphs[p_idx]
            topic = get_unique_section_topic(p_content, p_idx)

            if "Very Short" in sec_type or marks <= 2:
                q_text = f"What key detail or fact is described regarding {topic} in the text?"
                ans = extract_grounded_answer_from_text(q_text, topic, p_content, max_words=25)
                mp = [f"Correct fact/definition identification ({marks} marks)"]
            elif "Short" in sec_type or marks <= 5:
                q_text = f"Explain the significance of {topic} as presented in the chapter narrative."
                ans = extract_grounded_answer_from_text(q_text, topic, p_content, max_words=60)
                p1 = marks // 2
                p2 = marks - p1
                mp = [f"Identification of concept ({p1} marks)", f"Explanation & context ({p2} marks)"]
            else:
                q_text = f"Provide a comprehensive analysis of {topic}, explaining its background, core mechanisms, and broader outcome."
                raw_ans = extract_grounded_answer_from_text(q_text, topic, p_content, max_words=180)
                ans = f"A detailed analysis of this section shows that {raw_ans[0].lower() + raw_ans[1:] if raw_ans else p_content[:250]}"
                p1 = marks // 4
                p2 = marks // 2
                p3 = marks - p1 - p2
                mp = [f"Introduction & context ({p1} marks)", f"Detailed analysis ({p2} marks)", f"Conclusion ({p3} marks)"]

            q_obj = {
                "id": str(uuid.uuid4()),
                "question_number": global_q_num,
                "question_text": q_text,
                "question_type": sec_type,
                "difficulty": sec.get("difficulty", difficulty),
                "marks": marks,
                "related_topic": f"Part {p_idx + 1}: {topic}",
                "section_name": sec_name,
                "answer": ans,
                "marking_points": mp,
                "expected_length": expected_len
            }

            all_questions.append(q_obj)
            if "Very Short" in sec_type:
                vs_list.append(q_obj)
            elif "Short" in sec_type:
                sq_list.append(q_obj)
            else:
                lq_list.append(q_obj)

            global_q_num += 1

    return {
        "very_short_questions": vs_list,
        "short_questions": sq_list,
        "long_questions": lq_list,
        "all_questions": all_questions,
        "sections": active_sections
    }

SYSTEM_CHAT_PROMPT = (
    "You are TeachGenie.AI Assistant, an intelligent, helpful AI assistant for teachers, students, and educators.\n"
    "- Answer the user's actual question directly, accurately, concisely, and naturally according to their intent.\n"
    "- For direct factual questions (e.g., 'When did India become independent?', 'What is 25 x 16?'), provide a direct, accurate answer first (e.g., 'India became independent on 15 August 1947.', '25 × 16 = 400.').\n"
    "- Do NOT wrap every response in a generic 'Teaching Guide' or lesson plan format unless the user explicitly requests a lesson plan or teaching guide.\n"
    "- If the user asks for a specific number of questions (e.g., 'Generate 20 questions on photosynthesis'), generate EXACTLY the requested quantity of questions.\n"
    "- Use previous conversation history context when answering follow-up questions (e.g., 'What was my previous question?').\n"
    "- For concept explanations, lesson plans, or problem solving, provide clear, step-by-step responses using clean Markdown formatting (headings, lists, code blocks, tables).\n"
    "- Answer in the user's requested language (English, Hindi, Hinglish, etc.)."
)

def format_gemini_contents(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Format chat messages to satisfy Gemini API requirement of alternating user/model roles."""
    contents = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "model"
        content = m.get("content", "").strip()
        if not content:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += f"\n\n{content}"
        else:
            contents.append({"role": role, "parts": [{"text": content}]})
    return contents

def generate_chat_response(messages: List[Dict[str, str]], user_api_key: Optional[str] = None) -> str:
    """
    Generate a real LLM AI response for multi-turn chat conversations using Groq, Gemini, or OpenAI.
    """
    if not messages:
        raise ValueError("No message history provided for chat generation.")

    # 1. Try Groq / xAI API Provider
    groq_key = user_api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or settings.GROQ_API_KEY
    if groq_key:
        if groq_key.startswith("xai-"):
            url = "https://api.x.ai/v1/chat/completions"
            models = ["grok-2-latest", "grok-beta"]
        else:
            url = "https://api.groq.com/openai/v1/chat/completions"
            models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

        groq_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]
        if not any(m["role"] == "system" for m in groq_messages):
            groq_messages.insert(0, {"role": "system", "content": SYSTEM_CHAT_PROMPT})

        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        for model_name in models:
            data = {
                "model": model_name,
                "messages": groq_messages,
                "temperature": 0.3
            }
            try:
                req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=35) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    content = res_body["choices"][0]["message"]["content"]
                    if content and content.strip():
                        logger.info(f"Chat response generated successfully via Groq/xAI model: {model_name}")
                        return content.strip()
            except Exception as e:
                logger.warning(f"Chat Groq/xAI model {model_name} error: {e}")
                continue

    # 2. Try Gemini REST API Provider
    gemini_key = user_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY
    if gemini_key:
        models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        contents = format_gemini_contents(messages)
            
        for model_name in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "system_instruction": {
                    "parts": [{"text": SYSTEM_CHAT_PROMPT}]
                },
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.3
                }
            }
            try:
                req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=35) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    candidates = res_body.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            text = parts[0]["text"]
                            if text and text.strip():
                                logger.info(f"Chat response generated successfully via Gemini REST model: {model_name}")
                                return text.strip()
            except Exception as e:
                logger.warning(f"Chat Gemini REST model {model_name} error: {e}")
                continue

    # 3. Try Gemini SDK Client Provider
    client = get_gemini_client(user_api_key)
    if client:
        try:
            last_msg = messages[-1]["content"] if messages else ""
            if hasattr(client, "models"):
                res = client.models.generate_content(model='gemini-2.0-flash', contents=last_msg)
                if res and res.text and res.text.strip():
                    return res.text.strip()
            elif hasattr(client, "GenerativeModel"):
                res = client.GenerativeModel("gemini-1.5-flash").generate_content(last_msg)
                if res and res.text and res.text.strip():
                    return res.text.strip()
        except Exception as e:
            logger.error(f"Chat Gemini SDK error: {e}")

    # No fake hardcoded fallback. Raise clear error so frontend displays helpful error message.
    raise ValueError(
        "AI Assistant provider is currently unavailable. "
        "Please ensure a valid GEMINI_API_KEY or GROQ_API_KEY is configured in server environment variables or account settings."
    )
