import os
import json
import re
import uuid
import logging
import urllib.request
from typing import List, Dict, Any, Optional

from app.config import settings
from app.doc_processor import chunk_text_if_needed

logger = logging.getLogger("ai_engine")

def call_groq_api(prompt: str, user_api_key: Optional[str] = None) -> Optional[str]:
    """Call Groq API using high-speed llama-3.3-70b-versatile model with JSON format output."""
    groq_key = user_api_key or os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    if not groq_key or not groq_key.startswith("gsk_"):
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    
    # Active, supported Groq models
    models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    
    for model_name in models_to_try:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert educational assessment designer. Return ONLY valid JSON output."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3
        }

        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                content = res_body["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content
        except Exception as e:
            logger.warning(f"Groq API model {model_name} error: {e}")
            continue

    return None

def get_gemini_client(user_api_key: Optional[str] = None):
    """Obtain initialized Google GenAI client or key."""
    api_key = user_api_key or os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
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

def generate_questions_from_chapter(
    chapter_title: str,
    chapter_text: str,
    subject: str,
    grade: str,
    board: str = "General",
    language: str = "English",
    difficulty: str = "Medium",
    marks_dist: Dict[str, int] = None,
    special_instructions: str = "",
    user_api_key: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Core function to generate exactly 3 Very Short, 3 Short, and 3 Long answer questions.
    """
    if not marks_dist:
        marks_dist = {"very_short": 2, "short": 4, "long": 8}
        
    chunked_text = chunk_text_if_needed(chapter_text, max_words=4000)
    
    prompt = f"""
You are an expert educational curriculum designer and question paper creator.
Your task is to analyze the following chapter content and generate EXACTLY 9 questions based ONLY on the provided text.
For EVERY question, you MUST also generate a detailed, accurate model answer, marking scheme, and expected length.

CHAPTER METADATA:
- Chapter Title: {chapter_title}
- Subject: {subject}
- Grade/Class: {grade}
- Education Board: {board}
- Target Language: {language} (Ensure all questions and text are written in {language})
- Overall Difficulty Level: {difficulty}
- Special Instructions: {special_instructions or "None"}

CATEGORIES REQUIRED:
1. Very Short Answer Questions: EXACTLY 3 questions.
   - Test definitions, facts, important terms, or core identification.
   - Assign {marks_dist.get('very_short', 2)} marks each.
2. Short Answer Questions: EXACTLY 3 questions.
   - Test conceptual understanding, key points, or brief explanations.
   - Assign {marks_dist.get('short', 4)} marks each.
3. Long Answer Questions: EXACTLY 3 questions.
   - Test detailed explanation, analytical thinking, comparison, or step-by-step reasoning.
   - Assign {marks_dist.get('long', 8)} marks each.

RULES:
- Base every single question and model answer directly on the provided chapter text. Do NOT invent facts.
- Do NOT repeat the same concept; cover diverse important topics across the chapter.
- Model answers must be comprehensive and directly answer the question.
- Return ONLY a valid JSON object matching the EXACT JSON structure below.

CHAPTER TEXT CONTENT:
\"\"\"
{chunked_text}
\"\"\"

REQUIRED JSON OUTPUT FORMAT:
{{
  "very_short_questions": [
    {{
      "question_number": 1,
      "question_text": "...",
      "question_type": "Very Short Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('very_short', 2)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Point 1 (1 mark)", "Point 2 (1 mark)"],
      "expected_length": "1-10 words"
    }},
    {{
      "question_number": 2,
      "question_text": "...",
      "question_type": "Very Short Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('very_short', 2)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Point 1 (1 mark)", "Point 2 (1 mark)"],
      "expected_length": "1-10 words"
    }},
    {{
      "question_number": 3,
      "question_text": "...",
      "question_type": "Very Short Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('very_short', 2)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Point 1 (1 mark)", "Point 2 (1 mark)"],
      "expected_length": "1-10 words"
    }}
  ],
  "short_questions": [
    {{
      "question_number": 4,
      "question_text": "...",
      "question_type": "Short Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('short', 4)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Point 1 (2 marks)", "Point 2 (2 marks)"],
      "expected_length": "40-60 words"
    }},
    {{
      "question_number": 5,
      "question_text": "...",
      "question_type": "Short Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('short', 4)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Point 1 (2 marks)", "Point 2 (2 marks)"],
      "expected_length": "40-60 words"
    }},
    {{
      "question_number": 6,
      "question_text": "...",
      "question_type": "Short Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('short', 4)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Point 1 (2 marks)", "Point 2 (2 marks)"],
      "expected_length": "40-60 words"
    }}
  ],
  "long_questions": [
    {{
      "question_number": 7,
      "question_text": "...",
      "question_type": "Long Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('long', 8)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Intro (2 marks)", "Key Points (4 marks)", "Conclusion (2 marks)"],
      "expected_length": "150-250 words"
    }},
    {{
      "question_number": 8,
      "question_text": "...",
      "question_type": "Long Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('long', 8)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Intro (2 marks)", "Key Points (4 marks)", "Conclusion (2 marks)"],
      "expected_length": "150-250 words"
    }},
    {{
      "question_number": 9,
      "question_text": "...",
      "question_type": "Long Answer",
      "difficulty": "{difficulty}",
      "marks": {marks_dist.get('long', 8)},
      "related_topic": "...",
      "answer": "...",
      "marking_points": ["Intro (2 marks)", "Key Points (4 marks)", "Conclusion (2 marks)"],
      "expected_length": "150-250 words"
    }}
  ]
}}
"""

    # 1. Try Groq API Provider first
    groq_res = call_groq_api(prompt, user_api_key)
    if groq_res:
        try:
            data = extract_json_from_text(groq_res)
            vs = data.get("very_short_questions", [])
            sq = data.get("short_questions", [])
            lq = data.get("long_questions", [])
            if len(vs) == 3 and len(sq) == 3 and len(lq) == 3:
                for q in vs + sq + lq:
                    q["id"] = str(uuid.uuid4())
                return data
        except Exception as e:
            logger.warning(f"Failed to parse Groq response JSON: {e}")

    # 2. Try Gemini API Provider second
    client = get_gemini_client(user_api_key)
    if client:
        try:
            raw_response = ""
            if hasattr(client, "models"):
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                raw_response = response.text
            elif hasattr(client, "GenerativeModel"):
                model = client.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                raw_response = response.text
                
            data = extract_json_from_text(raw_response)
            vs = data.get("very_short_questions", [])
            sq = data.get("short_questions", [])
            lq = data.get("long_questions", [])
            if len(vs) == 3 and len(sq) == 3 and len(lq) == 3:
                for q in vs + sq + lq:
                    q["id"] = str(uuid.uuid4())
                return data
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")

    # 3. Grounded Fallback Question Generator
    return generate_fallback_questions(
        chapter_title=chapter_title,
        chapter_text=chapter_text,
        subject=subject,
        grade=grade,
        language=language,
        difficulty=difficulty,
        marks_dist=marks_dist
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
  "related_topic": "...",
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
            return q_data
        except Exception as e:
            logger.error(f"Gemini single question error: {e}")

    # 3. Grounded Fallback
    q_topic = topic or chapter_title
    return {
        "id": str(uuid.uuid4()),
        "question_number": 1,
        "question_text": f"Explain the core principle of {q_topic} discussed in section 2 of the chapter." if "Long" in question_type else (f"Briefly describe the significance of {q_topic}." if "Short" in question_type else f"Define {q_topic} according to the text."),
        "question_type": question_type,
        "difficulty": difficulty,
        "marks": marks,
        "related_topic": q_topic,
        "answer": f"Detailed model answer for {q_topic} explaining core concepts as detailed in '{chapter_title}'.",
        "marking_points": ["Correct definition/identification (Full Marks)"],
        "expected_length": "1-10 words" if "Very Short" in question_type else ("40-60 words" if "Short" in question_type else "150-250 words")
    }

def extract_grounded_answer_from_text(question_text: str, topic: str, chapter_text: str, max_words: int = 80) -> str:
    """Extract exact relevant sentences from chapter_text that match the question or topic keywords."""
    if not chapter_text or not chapter_text.strip():
        return f"Refer to the core concepts outlined in {topic}."

    cleaned = chapter_text.strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if len(s.strip()) > 15]
    
    if not sentences:
        return cleaned[:250] + "..." if len(cleaned) > 250 else cleaned

    stop_words = {"what", "is", "the", "define", "key", "term", "presented", "in", "chapter", "how", "why", "explain", "of", "and", "a", "to", "or", "regarding", "as", "described", "text", "which", "state", "outline", "list", "compare", "contrast"}
    words = re.findall(r'\w+', (question_text + " " + topic).lower())
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]

    scored_sentences = []
    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for kw in keywords if kw in s_lower)
        scored_sentences.append((score, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_matches = [s for score, s in scored_sentences if score > 0]
    
    if best_matches:
        answer_text = " ".join(best_matches[:2])
    else:
        answer_text = " ".join(sentences[:2])

    words_list = answer_text.split()
    if len(words_list) > max_words:
        return " ".join(words_list[:max_words]) + "..."
    return answer_text

def generate_answer_key_for_questions(
    chapter_title: str,
    chapter_text: str,
    questions: List[Dict[str, Any]],
    language: str = "English",
    user_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate comprehensive answer keys with marking points and expected length."""
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
- Do NOT use generic placeholder text. Give exact answers, definitions, and explanations appropriate for the subject and grade level.
- Every answer must directly address the specific question text.

For each question provided below, generate:
1. "answer": Detailed model answer specifically answering the question.
2. "marking_points": Bulleted key marking points showing how marks are awarded (e.g. ["Correct definition (1 mark)", "Explanation of core concept (1 mark)"]).
3. "expected_length": Expected answer length (e.g. "1-10 words", "40-60 words", "150-250 words").

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
      "answer": "<detailed specific model answer answering question 1>",
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
                ans_dict = {item.get("id"): item for item in ans_list if isinstance(item, dict)}
                for q in questions:
                    q_id = q.get("id")
                    if q_id in ans_dict and ans_dict[q_id].get("answer"):
                        q["answer"] = ans_dict[q_id].get("answer")
                        q["marking_points"] = ans_dict[q_id].get("marking_points", [])
                        q["expected_length"] = ans_dict[q_id].get("expected_length")
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
                ans_dict = {item.get("id"): item for item in ans_list if isinstance(item, dict)}
                for q in questions:
                    q_id = q.get("id")
                    if q_id in ans_dict and ans_dict[q_id].get("answer"):
                        q["answer"] = ans_dict[q_id].get("answer")
                        q["marking_points"] = ans_dict[q_id].get("marking_points", [])
                        q["expected_length"] = ans_dict[q_id].get("expected_length")
                return questions
        except Exception as e:
            logger.error(f"Gemini answer key error: {e}")

    # 3. Grounded Specific Fallback Answers from Chapter Text
    for q in questions:
        q_type = q.get("question_type", "")
        q_text = q.get("question_text", "")
        topic = q.get("related_topic", chapter_title)
        
        extracted_ans = extract_grounded_answer_from_text(q_text, topic, chapter_text, max_words=80)
        
        if "Very Short" in q_type:
            q["answer"] = extracted_ans
            q["expected_length"] = "1 - 10 words"
            q["marking_points"] = ["Correct definition/term identification (Full Marks)"]
        elif "Short" in q_type:
            q["answer"] = extracted_ans
            q["expected_length"] = "40 - 60 words"
            q["marking_points"] = ["Identification of primary concept (2 marks)", "Key explanation & illustration (2 marks)"]
        else:
            q["answer"] = extracted_ans
            q["expected_length"] = "150 - 250 words"
            q["marking_points"] = [
                "Introduction and core definition (2 marks)",
                "Detailed step-by-step analysis / key points (4 marks)",
                "Conclusion & illustrative example (2 marks)"
            ]
            
    return questions

def generate_fallback_questions(
    chapter_title: str,
    chapter_text: str,
    subject: str,
    grade: str,
    language: str = "English",
    difficulty: str = "Medium",
    marks_dist: Dict[str, int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Smart text parser to generate grounded chapter questions when offline or fallback mode is engaged."""
    paragraphs = [p.strip() for p in chapter_text.split("\n\n") if len(p.strip()) > 40]
    if not paragraphs:
        paragraphs = [chapter_text]
        
    topic_1 = paragraphs[0][:50] + "..." if paragraphs else chapter_title
    topic_2 = paragraphs[len(paragraphs)//2][:50] + "..." if len(paragraphs) > 1 else "Core Principles"
    topic_3 = paragraphs[-1][:50] + "..." if len(paragraphs) > 2 else "Applications"

    is_hindi = language.lower() == "hindi"

    if is_hindi:
        vs_questions = [
            f"अध्याय '{chapter_title}' के अनुसार {topic_1[:30]} की मुख्य परिभाषा क्या है?",
            f"पाठ में उल्लेखित {topic_2[:30]} से आप क्या समझते हैं?",
            f"अध्याय के मुख्य विषय से संबंधित किसी एक महत्त्वपूर्ण उदाहरण का उल्लेख कीजिए।"
        ]
        s_questions = [
            f"{subject} में {topic_1[:30]} के महत्त्व की संक्षेप में व्याख्या कीजिए।",
            f"अध्याय के आधार पर {topic_2[:30]} और संबंधित सिद्धांतों के बीच मुख्य अंतर स्पष्ट कीजिए।",
            f"पाठ्य सामग्री के अनुसार {topic_3[:30]} की प्रक्रिया के मुख्य चरणों को समझाइए।"
        ]
        l_questions = [
            f"अध्याय '{chapter_title}' का गहराई से विश्लेषण करते हुए {topic_1[:30]} पर एक विस्तृत निबंध लिखिए।",
            f"{topic_2[:30]} के विभिन्न पहलुओं, कारणों एवं प्रभावों का उदाहरण सहित विस्तारपूर्वक वर्णन कीजिए।",
            f"पाठ में दिए गए सिद्धांतों के आधार पर {topic_3[:30]} के व्यावहारिक उपयोग एवं निष्कर्षों की व्याख्या कीजिए।"
        ]
    else:
        vs_questions = [
            f"Define the key term '{topic_1[:30]}' as presented in the chapter '{chapter_title}'.",
            f"What primary component of {topic_2[:30]} is highlighted in the text?",
            f"State one fundamental fact or principle regarding {topic_3[:30]}."
        ]
        s_questions = [
            f"Briefly explain the role of {topic_1[:30]} in understanding {chapter_title}.",
            f"Outline the main characteristics and function of {topic_2[:30]} described in the text.",
            f"Compare and contrast the key ideas associated with {topic_3[:30]} as outlined in the chapter."
        ]
        l_questions = [
            f"Provide a comprehensive, step-by-step analysis of {topic_1[:30]} based on the chapter content.",
            f"Discuss the broader significance of {topic_2[:30]}, providing detailed explanations and relevant examples.",
            f"Evaluate the conclusions regarding {topic_3[:30]} drawn in the text, detailing its key applications and theoretical framework."
        ]

    vs_list = []
    for i, text in enumerate(vs_questions, 1):
        topic_name = f"Topic {i}: {topic_1[:20]}"
        vs_list.append({
            "id": str(uuid.uuid4()),
            "question_number": i,
            "question_text": text,
            "question_type": "Very Short Answer",
            "difficulty": difficulty,
            "marks": marks_dist.get("very_short", 2) if marks_dist else 2,
            "related_topic": topic_name,
            "answer": f"In '{chapter_title}', {topic_1[:30]} refers to the fundamental concept defined in the chapter text.",
            "marking_points": ["Correct definition / identification (2 marks)"],
            "expected_length": "1 - 10 words"
        })

    sq_list = []
    for i, text in enumerate(s_questions, 4):
        topic_name = f"Topic {i}: {topic_2[:20]}"
        sq_list.append({
            "id": str(uuid.uuid4()),
            "question_number": i,
            "question_text": text,
            "question_type": "Short Answer",
            "difficulty": difficulty,
            "marks": marks_dist.get("short", 4) if marks_dist else 4,
            "related_topic": topic_name,
            "answer": f"{topic_2[:30]} plays a crucial role in '{chapter_title}' by providing key functional principles and theoretical structure described in the text.",
            "marking_points": ["Primary role identification (2 marks)", "Key explanation (2 marks)"],
            "expected_length": "40 - 60 words"
        })

    lq_list = []
    for i, text in enumerate(l_questions, 7):
        topic_name = f"Topic {i}: {topic_3[:20]}"
        lq_list.append({
            "id": str(uuid.uuid4()),
            "question_number": i,
            "question_text": text,
            "question_type": "Long Answer",
            "difficulty": difficulty,
            "marks": marks_dist.get("long", 8) if marks_dist else 8,
            "related_topic": topic_name,
            "answer": f"A comprehensive evaluation of {topic_3[:30]} in '{chapter_title}' highlights its core theoretical framework, analytical components, and practical implications as presented in the chapter.",
            "marking_points": ["Core definition (2 marks)", "Detailed analysis (4 marks)", "Conclusion & applications (2 marks)"],
            "expected_length": "150 - 250 words"
        })

    return {
        "very_short_questions": vs_list,
        "short_questions": sq_list,
        "long_questions": lq_list
    }

