from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from gtts import gTTS
import uuid
from sympy import symbols, solve, sympify
import subprocess
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Fix Windows encoding
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI GATE ECE + Interview Assistant (English/Hindi)",
    description="GATE ECE prep + Interview questions - Powered by LLaMA 3",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration

OLLAMA_CMD = "ollama"
MODEL = "llama3"          # single model for all languages

AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

LANG_MAP = {
    "English": {"code": "en", "tts": "en"},
    "Hindi":   {"code": "hi", "tts": "hi"},
}

GATE_SUBJECTS = {
    "Networks": ["Network Theorems", "Two Port Networks", "Transient Analysis", "Resonance"],
    "Electronic Devices": ["PN Junction", "BJT", "MOSFET", "Optoelectronic Devices"],
    "Analog Circuits": ["Op-Amp", "Oscillators", "Amplifiers", "Filters"],
    "Digital Circuits": ["Boolean Algebra", "Combinational Circuits", "Sequential Circuits", "Memory"],
    "Signals and Systems": ["Fourier Transform", "Laplace Transform", "Z-Transform", "Sampling"],
    "Control Systems": ["Time Response", "Frequency Response", "Stability", "State Space"],
    "Communications": ["AM/FM", "Digital Modulation", "Information Theory", "Error Control"],
    "Electromagnetics": ["Maxwell Equations", "Wave Propagation", "Transmission Lines", "Antennas"],
}


# AI & Audio helpers


def generate_ai_response(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            logger.info(f"LLaMA 3 call (attempt {attempt + 1})")
            process = subprocess.Popen(
                [OLLAMA_CMD, "run", MODEL],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = process.communicate(input=prompt, timeout=300)
            if process.returncode != 0:
                logger.error(f"Ollama error: {stderr.strip()}")
                if attempt < max_retries - 1:
                    continue
                return f"Error: {stderr.strip()}"
            response = stdout.strip()
            if response:
                return response
        except subprocess.TimeoutExpired:
            logger.error("Timeout")
            if attempt < max_retries - 1:
                continue
            return "Error: Timeout – model took too long to respond."
        except Exception as e:
            logger.error(str(e))
            if attempt < max_retries - 1:
                continue
            return f"Error: {str(e)}"
    return "Error: Failed after all retries."


def generate_audio(text: str, language: str) -> Optional[str]:
    try:
        lang_code = LANG_MAP.get(language, {}).get("tts", "en")
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        audio_text = text[:1200]
        gTTS(text=audio_text, lang=lang_code, slow=False).save(filepath)
        logger.info(f"Audio saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Audio error: {e}")
        return None


# Prompt builders


def _gate_prompt_english(subject, topic, difficulty, qtype) -> str:
    qtype_instructions = {
        "MCQ": "Single correct answer MCQ with 4 options (A)(B)(C)(D). Mark exactly ONE correct option.",
        "MSQ": "Multiple Select Question with 4 options (A)(B)(C)(D). One or more options may be correct. Clearly state ALL correct options.",
        "NUMERICAL": "Numerical Answer Type (NAT). The answer is a specific number (integer or decimal). No options needed.",
        "NAT": "Numerical Answer Type (NAT). The answer is a specific number (integer or decimal). No options needed.",
    }
    instruction = qtype_instructions.get(qtype.upper(), "")

    return f"""You are a GATE ECE expert. Generate ONE high-quality GATE ECE question strictly following the format below.

Subject  : {subject}
Topic    : {topic}
Difficulty: {difficulty}
Type     : {qtype} — {instruction}

### FORMAT (use these exact section headers):

**1. QUESTION**
Write a precise, technically accurate question. For numerical types include units where applicable.

**2. OPTIONS** (only for MCQ/MSQ)
(A) ...
(B) ...
(C) ...
(D) ...

**3. CORRECT ANSWER**
State the correct option(s) or numerical value with units.

**4. STEP-BY-STEP SOLUTION**
Show all working clearly. Use formulas, substitute values, and derive the answer.

**5. KEY CONCEPT**
Explain the underlying theory/formula in 3-5 sentences.

**6. COMMON MISTAKES**
List 2-3 frequent errors students make on this topic.

**7. EXAM INFO**
Marks: [1 or 2] | Time: [X] minutes | Negative marking: [Yes/No]

**8. RELATED TOPICS**
List 2-3 closely related GATE topics.

Be 100% factually correct. Start directly with section 1."""


def _gate_prompt_hindi(subject, topic, difficulty, qtype) -> str:
    qtype_map = {
        "MCQ": "एकल सही उत्तर MCQ — 4 विकल्प (A)(B)(C)(D), केवल एक सही।",
        "MSQ": "बहु-चयन प्रश्न (MSQ) — 4 विकल्प, एक या अधिक सही हो सकते हैं। सभी सही विकल्प स्पष्ट करें।",
        "NUMERICAL": "संख्यात्मक उत्तर प्रकार (NAT) — सटीक संख्या/दशमलव उत्तर दें, कोई विकल्प नहीं।",
        "NAT": "संख्यात्मक उत्तर प्रकार (NAT) — सटीक संख्या/दशमलव उत्तर दें, कोई विकल्प नहीं।",
    }
    instruction = qtype_map.get(qtype.upper(), "")

    return f"""आप एक GATE ECE विशेषज्ञ हैं।

⚠️ सख्त निर्देश: नीचे दिए गए प्रत्येक खंड का उत्तर पूरी तरह हिंदी में लिखें।
❌ किसी भी खंड में अंग्रेजी वाक्य या अंग्रेजी व्याख्या न लिखें।
✅ केवल गणितीय सूत्र (जैसे V=IR, H(s)=...) और तकनीकी संज्ञाएँ (जैसे Fourier, BJT) अंग्रेजी में लिख सकते हैं।
✅ हर खंड की व्याख्या, विवरण, और निष्कर्ष हिंदी में ही होना चाहिए।

विषय        : {subject}
टॉपिक       : {topic}
कठिनाई      : {difficulty}
प्रश्न प्रकार : {qtype} — {instruction}

नीचे दिए गए प्रारूप का पालन करें। प्रत्येक खंड हिंदी में लिखें:

**1. प्रश्न**
(हिंदी में स्पष्ट और तकनीकी रूप से सटीक प्रश्न लिखें)

**2. विकल्प** (केवल MCQ/MSQ के लिए)
(A) ...
(B) ...
(C) ...
(D) ...

**3. सही उत्तर**
(सही विकल्प या संख्यात्मक मान इकाई सहित — हिंदी में लिखें)

**4. चरण-दर-चरण हल**
(सभी गणनाएँ और व्युत्पत्ति हिंदी में समझाएँ। सूत्र अंग्रेजी में लिख सकते हैं लेकिन उनकी व्याख्या हिंदी में दें)

**5. मुख्य अवधारणा**
(3-5 हिंदी वाक्यों में सिद्धांत समझाएँ — अंग्रेजी वाक्य नहीं)

**6. सामान्य गलतियाँ**
(2-3 गलतियाँ हिंदी में लिखें जो छात्र अक्सर करते हैं)

**7. परीक्षा जानकारी**
अंक: [1 या 2] | समय: [X] मिनट | नकारात्मक अंकन: [हाँ/नहीं]

**8. संबंधित विषय**
(2-3 संबंधित GATE टॉपिक हिंदी में लिखें)

याद रखें: हर खंड की भाषा हिंदी होनी चाहिए। सीधे खंड 1 से शुरू करें।"""


def build_gate_prompt(subject: str, topic: str, difficulty: str, qtype: str, language: str) -> str:
    if language == "Hindi":
        return _gate_prompt_hindi(subject, topic, difficulty, qtype)
    else:
        return _gate_prompt_english(subject, topic, difficulty, qtype)


def build_interview_prompt(role: str, domain: str, difficulty: str, language: str) -> str:
    if language == "Hindi":
        return f"""आप एक अनुभवी तकनीकी साक्षात्कारकर्ता हैं।

⚠️ सख्त निर्देश: नीचे दिए गए प्रत्येक खंड का उत्तर पूरी तरह हिंदी में लिखें।
❌ किसी भी खंड में अंग्रेजी वाक्य या अंग्रेजी व्याख्या न लिखें।
✅ केवल तकनीकी संज्ञाएँ (जैसे API, VLSI, MOSFET) और कोड स्निपेट अंग्रेजी में लिख सकते हैं।
✅ हर खंड की व्याख्या, विवरण, और निष्कर्ष हिंदी में ही होना चाहिए।

पद      : {role}
क्षेत्र   : {domain}
कठिनाई  : {difficulty}

नीचे दिए गए प्रारूप का पालन करें। प्रत्येक खंड हिंदी में लिखें:

**1. साक्षात्कार प्रश्न**
(हिंदी में स्पष्ट और तकनीकी प्रश्न लिखें)

**2. आदर्श उत्तर / मुख्य बिंदु**
(हिंदी में विस्तृत उत्तर दें — अंग्रेजी वाक्य नहीं)

**3. अनुवर्ती प्रश्न** (2-3)
(हिंदी में फॉलो-अप प्रश्न लिखें)

**4. मूल्यांकन मानदंड**
(हिंदी में बताएं कि उम्मीदवार का मूल्यांकन कैसे होगा)

**5. उम्मीदवारों की सामान्य गलतियाँ**
(हिंदी में 2-3 सामान्य गलतियाँ लिखें)

**6. उम्मीदवार के लिए सुझाव**
(हिंदी में व्यावहारिक सुझाव दें)

याद रखें: हर खंड की भाषा हिंदी होनी चाहिए। सीधे खंड 1 से शुरू करें।"""

    else:
        return f"""You are a senior technical interviewer. Generate one strong interview question in the format below.

Role      : {role}
Domain    : {domain}
Difficulty: {difficulty}

**1. INTERVIEW QUESTION**
**2. MODEL ANSWER / KEY POINTS**
**3. FOLLOW-UP QUESTIONS** (2-3)
**4. EVALUATION CRITERIA**
**5. COMMON CANDIDATE MISTAKES**
**6. TIPS FOR THE CANDIDATE**

Be precise, professional, and technically accurate. Start directly with section 1."""


def _calc_question_count(duration: int, test_type: str) -> dict:
    """
    Scale question count and marks to fit the given duration.
    GATE: ~1.8 min/question average (mix of 1-mark and 2-mark)
    Placement: ~2 min/question average (longer reasoning questions)
    """
    if test_type == "PLACEMENT":
        total_q = max(5, round(duration / 2))
        total_marks = total_q * 2
        aptitude_q = max(2, round(total_q * 0.25))
        technical_q = total_q - aptitude_q
    else:  # GATE
        total_q = max(5, round(duration / 1.8))
        total_q = min(total_q, 65)          # GATE max is 65
        total_marks = round(total_q * 1.5)   # mix of 1M and 2M questions
        aptitude_q = max(2, round(total_q * 0.15))
        technical_q = total_q - aptitude_q

    # difficulty split based on duration
    if duration <= 30:
        diff_split = "Easy 60% / Medium 30% / Hard 10%"
    elif duration <= 60:
        diff_split = "Easy 40% / Medium 40% / Hard 20%"
    elif duration <= 120:
        diff_split = "Easy 30% / Medium 45% / Hard 25%"
    else:
        diff_split = "Easy 25% / Medium 45% / Hard 30%"

    return {
        "total_q": total_q,
        "total_marks": total_marks,
        "aptitude_q": aptitude_q,
        "technical_q": technical_q,
        "diff_split": diff_split,
        "time_per_q": round(duration / total_q, 1),
    }


def build_full_test_prompt(language: str, duration: int, test_type: str = "GATE", include_solutions: bool = False) -> str:
    cfg = _calc_question_count(duration, test_type)
    sol_note = (
        "Include correct answer and brief explanation for every question."
        if include_solutions
        else "Do NOT include answers or solutions — this is a question paper only."
    )

    aq = cfg["aptitude_q"]
    tq = cfg["technical_q"]
    tm = cfg["total_marks"]

    if test_type == "PLACEMENT":
        domain_note = (
            "Topics: Aptitude (quantitative, logical reasoning, verbal), "
            "Core ECE/CS concepts (data structures, OS, networking, VLSI, embedded systems, "
            "digital circuits), and coding/problem-solving questions relevant to placement drives."
        )
        section_note = (
            f"Section A — Aptitude: {aq} questions\n"
            f"Section B — Technical (ECE/CS core + coding): {tq} questions"
        )
    else:
        domain_note = (
            "Topics strictly from GATE ECE syllabus: Networks, Electronic Devices, "
            "Analog Circuits, Digital Circuits, Signals & Systems, Control Systems, "
            "Communications, Electromagnetics, Engineering Mathematics, General Aptitude."
        )
        section_note = (
            f"Section A — General Aptitude: {aq} questions (10 marks)\n"
            f"Section B — Technical ECE: {tq} questions ({tm - 10} marks)"
        )

    sol_hi = "उत्तर और व्याख्या सहित प्रत्येक प्रश्न दें।" if include_solutions else "केवल प्रश्न पत्र — उत्तर या हल न दें।"
    ans_key_hi = "**उत्तर कुंजी** — प्रत्येक प्रश्न का सही उत्तर और संक्षिप्त व्याख्या हिंदी में दें।" if include_solutions else ""
    ans_key_en = "**Answer Key** — After all questions, list each Qno with the correct answer and a one-line explanation." if include_solutions else ""
    type_hi = "GATE ECE" if test_type == "GATE" else "Placement"
    type_en = "GATE ECE" if test_type == "GATE" else "Placement"

    if language == "Hindi":
        return (
            f"आप एक {type_hi} विशेषज्ञ हैं।\n"
            f"⚠️ पूरा उत्तर हिंदी में लिखें। अंग्रेजी वाक्य न लिखें।\n\n"
            f"परीक्षा विवरण:\n"
            f"- प्रकार       : {test_type}\n"
            f"- कुल समय      : {duration} मिनट\n"
            f"- कुल प्रश्न    : {cfg['total_q']} (प्रति प्रश्न औसत {cfg['time_per_q']} मिनट)\n"
            f"- कुल अंक      : {tm}\n"
            f"- कठिनाई       : {cfg['diff_split']}\n\n"
            f"{section_note}\n\n"
            f"{sol_hi}\n\n"
            f"निम्न प्रारूप में प्रश्न पत्र तैयार करें:\n\n"
            f"**खंड A** ({aq} प्रश्न)\n"
            f"प्रत्येक प्रश्न को Q1, Q2... क्रमांक दें।\n"
            f"MCQ के लिए (A)(B)(C)(D) विकल्प दें।\n\n"
            f"**खंड B** ({tq} प्रश्न)\n"
            f"प्रत्येक प्रश्न को Q{aq+1}, Q{aq+2}... क्रमांक दें।\n"
            f"MCQ, NAT, MSQ — मिश्रित प्रकार के प्रश्न दें।\n"
            f"{ans_key_hi}\n\n"
            f"सभी प्रश्न GATE ECE पाठ्यक्रम/Placement विषयों से लें।\n"
            f"तथ्यात्मक रूप से 100% सटीक रहें। सीधे खंड A Q1 से शुरू करें।"
        )
    else:
        return (
            f"You are a {type_en} expert. Generate a complete question paper — "
            f"NOT a blueprint. Write actual exam questions.\n\n"
            f"Exam Details:\n"
            f"- Type         : {test_type}\n"
            f"- Total Time   : {duration} minutes\n"
            f"- Total Qs     : {cfg['total_q']} questions (~{cfg['time_per_q']} min/question)\n"
            f"- Total Marks  : {tm}\n"
            f"- Difficulty   : {cfg['diff_split']}\n\n"
            f"{section_note}\n"
            f"{sol_note}\n\n"
            f"{domain_note}\n\n"
            f"Format:\n"
            f"**Section A** — {aq} Questions\n"
            f"Number each as Q1, Q2... Give (A)(B)(C)(D) for MCQ.\n\n"
            f"**Section B** — {tq} Questions\n"
            f"Number continuing from Q{aq+1}. Mix of MCQ, NAT, MSQ types.\n"
            f"{ans_key_en}\n\n"
            f"All questions must be factually accurate and solvable within the given time. "
            f"Start directly with Section A Q1."
        )





# Pydantic Models


class GateQuestionRequest(BaseModel):
    subject: str
    topic: str
    difficulty: str
    qtype: str
    language: str = "English"
    include_audio: bool = True

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        d = v.lower()
        if d not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        return d

    @field_validator("qtype")
    @classmethod
    def validate_qtype(cls, v: str) -> str:
        q = v.upper()
        if q not in {"MCQ", "MSQ", "NUMERICAL", "NAT"}:
            raise ValueError("qtype must be MCQ, MSQ, NUMERICAL, or NAT")
        return q

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v


class FullTestRequest(BaseModel):
    language: str = "English"
    duration_minutes: int = 180
    include_solutions: bool = False
    test_type: str = "GATE"   # GATE or PLACEMENT

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v < 10:
            raise ValueError("Minimum duration is 10 minutes")
        if v > 180:
            raise ValueError("Maximum duration is 180 minutes (3 hours)")
        return v

    @field_validator("test_type")
    @classmethod
    def validate_test_type(cls, v: str) -> str:
        t = v.upper()
        if t not in {"GATE", "PLACEMENT"}:
            raise ValueError("test_type must be GATE or PLACEMENT")
        return t


class InterviewRequest(BaseModel):
    role: str
    domain: str
    language: str = "English"
    difficulty: str = "medium"
    include_audio: bool = True

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v


    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        d = v.lower()
        if d not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        return d


class PerformanceRequest(BaseModel):
    attempts: Dict[str, float]
    user_id: Optional[str] = None

    @field_validator("attempts")
    @classmethod
    def validate_attempts(cls, v: Dict[str, float]) -> Dict[str, float]:
        for topic, score in v.items():
            if not (0 <= score <= 1):
                raise ValueError(f"Score for '{topic}' must be between 0 and 1")
        return v


class MathValidationRequest(BaseModel):
    expression: str
    variable: str = "x"



# Endpoints


@app.get("/")
def home():
    return {
        "message": "AI GATE ECE + Interview Assistant API",
        "version": "3.1.0",
        "model": MODEL,
        "status": "running",
        "languages": list(LANG_MAP.keys()),
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL, "timestamp": datetime.now().isoformat()}


@app.get("/languages")
def get_languages():
    return {"languages": list(LANG_MAP.keys())}


@app.get("/subjects")
def get_subjects():
    return {
        "subjects": GATE_SUBJECTS,
        "total_subjects": len(GATE_SUBJECTS),
        "total_topics": sum(len(t) for t in GATE_SUBJECTS.values()),
    }


@app.post("/generate-gate-question")
def generate_gate_question(req: GateQuestionRequest):
    try:
        prompt = build_gate_prompt(req.subject, req.topic, req.difficulty, req.qtype, req.language)
        ai_response = generate_ai_response(prompt)

        if ai_response.startswith("Error"):
            raise HTTPException(500, detail=ai_response)

        audio_file = generate_audio(ai_response, req.language) if req.include_audio else None

        return {
            "success": True,
            "subject": req.subject,
            "topic": req.topic,
            "difficulty": req.difficulty,
            "question_type": req.qtype,
            "language": req.language,
            "model": MODEL,
            "content": ai_response,
            "audio_file": audio_file,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(500, detail=str(e))


@app.post("/generate-full-test")
def generate_full_test(req: FullTestRequest):
    try:
        cfg = _calc_question_count(req.duration_minutes, req.test_type)
        prompt = build_full_test_prompt(
            req.language,
            req.duration_minutes,
            req.test_type,
            req.include_solutions,
        )
        ai_response = generate_ai_response(prompt)

        if ai_response.startswith("Error"):
            raise HTTPException(500, detail=ai_response)

        return {
            "success": True,
            "language": req.language,
            "model": MODEL,
            "test_type": req.test_type,
            "duration_minutes": req.duration_minutes,
            "total_questions": cfg["total_q"],
            "total_marks": cfg["total_marks"],
            "difficulty_split": cfg["diff_split"],
            "avg_time_per_question_minutes": cfg["time_per_q"],
            "include_solutions": req.include_solutions,
            "question_paper": ai_response,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/interview-agent")
def interview_agent(req: InterviewRequest):
    try:
        prompt = build_interview_prompt(req.role, req.domain, req.difficulty, req.language)
        ai_response = generate_ai_response(prompt)

        if ai_response.startswith("Error"):
            raise HTTPException(500, detail=ai_response)

        audio_file = generate_audio(ai_response, req.language) if req.include_audio else None

        return {
            "success": True,
            "role": req.role,
            "domain": req.domain,
            "difficulty": req.difficulty,
            "language": req.language,
            "model": MODEL,
            "interview_content": ai_response,
            "audio_file": audio_file,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/performance-analytics")
def performance_analytics(req: PerformanceRequest):
    try:
        if not req.attempts:
            raise HTTPException(400, detail="No attempts data provided")

        strengths, moderate, weaknesses = [], [], []
        for topic, score in req.attempts.items():
            entry = {"topic": topic, "score": score}
            if score >= 0.75:
                strengths.append(entry)
            elif score >= 0.5:
                moderate.append(entry)
            else:
                weaknesses.append(entry)

        readiness = sum(req.attempts.values()) / len(req.attempts)

        recommendations = []
        if readiness < 0.5:
            recommendations.append("Significant improvement needed — revise fundamentals across all subjects.")
        elif readiness < 0.7:
            recommendations.append("Good progress — focus on moderate and weak areas.")
        else:
            recommendations.append("Strong preparation level — focus on speed and accuracy.")

        if weaknesses:
            weak_names = ", ".join(w["topic"] for w in weaknesses[:3])
            recommendations.append(f"Prioritize these weak topics: {weak_names}")

        readiness_level = (
            "Excellent" if readiness >= 0.8 else
            "Good"      if readiness >= 0.7 else
            "Average"   if readiness >= 0.5 else
            "Needs Work"
        )

        return {
            "success": True,
            "user_id": req.user_id,
            "readiness_score": round(readiness, 3),
            "readiness_level": readiness_level,
            "strengths": strengths,
            "moderate": moderate,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/validate-math")
def validate_math(req: MathValidationRequest):
    try:
        var = symbols(req.variable)
        expr = sympify(req.expression)
        solutions = solve(expr, var)
        return {
            "success": True,
            "expression": str(expr),
            "variable": req.variable,
            "solutions": [str(s) for s in solutions],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    filepath = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg")
    raise HTTPException(404, detail="Audio file not found")



# Startup cleanup


@app.on_event("startup")
async def startup_event():
    logger.info("Server starting — cleaning stale audio files (>1 hour old)")
    try:
        if os.path.exists(AUDIO_DIR):
            for f in os.listdir(AUDIO_DIR):
                path = os.path.join(AUDIO_DIR, f)
                if os.path.isfile(path):
                    age = datetime.now().timestamp() - os.path.getmtime(path)
                    if age > 3600:
                        os.remove(path)
                        logger.info(f"Removed old audio: {f}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")



# Entry point


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  AI GATE ECE + Interview Assistant  v3.0.0")
    print(f"  Model  : {MODEL} (via Ollama)")
    print("  Languages: English | Hindi")
    print("  API    : http://127.0.0.1:8000")
    print("  Docs   : http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)