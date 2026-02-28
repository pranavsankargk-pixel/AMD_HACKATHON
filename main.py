from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from gtts import gTTS
import uuid
import httpx
import asyncio
from sympy import symbols, solve, sympify
import os
import sys
from typing import Dict, Optional
from datetime import datetime
import logging

# Windows UTF-8 fix
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI GATE ECE + Interview Assistant",
    description="GATE ECE prep + Interview questions - Powered by LLaMA 3",
    version="3.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# SPEED CHOICE — change MODEL to control speed vs quality
# "llama3.2:3b"  → 5-10  sec  (recommended — fast + good quality)
# "llama3.2:1b"  → 3-5   sec  (fastest — decent quality)
# "llama3"       → 20-40 sec  (best quality — slowest)
# After changing model run: ollama pull <model-name>
# ─────────────────────────────────────────────────────────────
MODEL      = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
AUDIO_DIR  = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

LANG_MAP = {
    "English": {"code": "en", "tts": "en"},
    "Hindi":   {"code": "hi", "tts": "hi"},
}

GATE_SUBJECTS = {
    "Networks":            ["Network Theorems", "Two Port Networks", "Transient Analysis", "Resonance"],
    "Electronic Devices":  ["PN Junction", "BJT", "MOSFET", "Optoelectronic Devices"],
    "Analog Circuits":     ["Op-Amp", "Oscillators", "Amplifiers", "Filters"],
    "Digital Circuits":    ["Boolean Algebra", "Combinational Circuits", "Sequential Circuits", "Memory"],
    "Signals and Systems": ["Fourier Transform", "Laplace Transform", "Z-Transform", "Sampling"],
    "Control Systems":     ["Time Response", "Frequency Response", "Stability", "State Space"],
    "Communications":      ["AM/FM", "Digital Modulation", "Information Theory", "Error Control"],
    "Electromagnetics":    ["Maxwell Equations", "Wave Propagation", "Transmission Lines", "Antennas"],
}

_http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup_event():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=120.0)
    logger.info(f"Client ready — model: {MODEL}")
    try:
        logger.info(f"Warming up {MODEL}...")
        await _http_client.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 1}},
        )
        logger.info("Warm-up done")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")
    try:
        for f in os.listdir(AUDIO_DIR):
            path = os.path.join(AUDIO_DIR, f)
            if os.path.isfile(path):
                if datetime.now().timestamp() - os.path.getmtime(path) > 3600:
                    os.remove(path)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    if _http_client:
        await _http_client.aclose()


async def generate_ai_response(prompt: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries):
        try:
            logger.info(f"AI call attempt {attempt + 1}")
            resp = await _http_client.post(
                OLLAMA_URL,
                json={
                    "model":  MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature":    0.7,
                        "num_predict":    800,
                        "num_ctx":        1024,
                        "repeat_penalty": 1.1,
                        "top_p":          0.9,
                        "top_k":          40,
                        "num_thread":     4,
                    }
                },
            )
            if resp.status_code != 200:
                if attempt < max_retries - 1:
                    continue
                return f"Error: Ollama returned status {resp.status_code}"
            result = resp.json().get("response", "").strip()
            if result:
                logger.info("Response received")
                return result
            if attempt < max_retries - 1:
                continue
            return "Error: Empty response from model"
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return "Error: Timeout"
        except httpx.ConnectError:
            return "Error: Cannot connect to Ollama. Run: ollama serve"
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return f"Error: {str(e)}"
    return "Error: Failed after all retries."


def _audio_sync(text: str, language: str) -> Optional[str]:
    try:
        lang_code = LANG_MAP.get(language, {}).get("tts", "en")
        filename  = f"audio_{uuid.uuid4().hex}.mp3"
        gTTS(text=text[:1000], lang=lang_code, slow=False).save(
            os.path.join(AUDIO_DIR, filename)
        )
        return filename
    except Exception as e:
        logger.error(f"Audio: {e}")
        return None

async def generate_audio(text: str, language: str) -> Optional[str]:
    return await asyncio.get_event_loop().run_in_executor(
        None, _audio_sync, text, language
    )


def _gate_prompt_english(subject, topic, difficulty, qtype) -> str:
    instructions = {
        "MCQ":       "4 options (A)(B)(C)(D). Exactly ONE correct.",
        "MSQ":       "4 options (A)(B)(C)(D). ONE OR MORE correct. State ALL.",
        "NUMERICAL": "No options. Exact number answer with units.",
        "NAT":       "No options. Exact number answer with units.",
    }
    return f"""You are a GATE ECE expert. Generate ONE question.
Subject:{subject} Topic:{topic} Difficulty:{difficulty} Type:{qtype}
Rule:{instructions.get(qtype.upper(),"")}

Use EXACTLY these headers:
**1. QUESTION**
**2. OPTIONS**
(A)...(B)...(C)...(D)...
**3. CORRECT ANSWER**
**4. STEP-BY-STEP SOLUTION**
**5. KEY CONCEPT**
**6. COMMON MISTAKES**
**7. EXAM INFO**
Marks:[1or2] | Time:[X]min | Negative:[Yes/No]
**8. RELATED TOPICS**

Be factually accurate. Start with section 1 immediately."""


def _gate_prompt_hindi(subject, topic, difficulty, qtype) -> str:
    instructions = {
        "MCQ":       "4 options (A)(B)(C)(D), only one correct.",
        "MSQ":       "4 options, one or more correct. State all.",
        "NUMERICAL": "No options. Exact number answer.",
        "NAT":       "No options. Exact number answer.",
    }
    return f"""You are a GATE ECE expert. Generate ONE question IN HINDI language only.
Subject:{subject} Topic:{topic} Difficulty:{difficulty} Type:{qtype}
Rule:{instructions.get(qtype.upper(),"")}

STRICT: Write entire response in Hindi. Only formulas like V=IR and technical terms like Fourier/BJT can be in English. All explanations must be in Hindi.

Use EXACTLY these Hindi headers:
**1. प्रश्न**
**2. विकल्प**
(A)...(B)...(C)...(D)...
**3. सही उत्तर**
**4. चरण-दर-चरण हल**
**5. मुख्य अवधारणा**
**6. सामान्य गलतियाँ**
**7. परीक्षा जानकारी**
अंक:[1या2] | समय:[X]मिनट | नकारात्मक:[हाँ/नहीं]
**8. संबंधित विषय**

Start with section 1 immediately."""


def build_gate_prompt(subject, topic, difficulty, qtype, language) -> str:
    if language == "Hindi":
        return _gate_prompt_hindi(subject, topic, difficulty, qtype)
    return _gate_prompt_english(subject, topic, difficulty, qtype)


def build_interview_prompt(role, domain, difficulty, language) -> str:
    if language == "Hindi":
        return f"""You are a senior technical interviewer. Generate ONE interview question IN HINDI only.
Role:{role} Domain:{domain} Difficulty:{difficulty}
STRICT: Write entire response in Hindi. Technical terms can be English.

**1. साक्षात्कार प्रश्न**
**2. आदर्श उत्तर**
**3. अनुवर्ती प्रश्न** (2-3)
**4. मूल्यांकन मानदंड**
**5. सामान्य गलतियाँ**
**6. सुझाव**

Start with section 1."""

    return f"""You are a senior technical interviewer. Generate ONE interview question.
Role:{role} Domain:{domain} Difficulty:{difficulty}

**1. INTERVIEW QUESTION**
**2. MODEL ANSWER**
**3. FOLLOW-UP QUESTIONS** (2-3)
**4. EVALUATION CRITERIA**
**5. COMMON MISTAKES**
**6. TIPS FOR CANDIDATE**

Start with section 1 immediately."""


def _calc_question_count(duration: int, test_type: str) -> dict:
    if test_type == "PLACEMENT":
        total_q     = min(90, max(5, round(duration / 2)))
        total_marks = total_q * 2
        aptitude_q  = max(2, round(total_q * 0.25))
    else:
        total_q     = min(65, max(5, round(duration / 1.8)))
        total_marks = round(total_q * 1.5)
        aptitude_q  = max(2, round(total_q * 0.15))

    if   duration <= 30:  diff = "Easy 60% / Medium 30% / Hard 10%"
    elif duration <= 60:  diff = "Easy 40% / Medium 40% / Hard 20%"
    elif duration <= 120: diff = "Easy 30% / Medium 45% / Hard 25%"
    else:                 diff = "Easy 25% / Medium 45% / Hard 30%"

    return {
        "total_q":     total_q,
        "total_marks": total_marks,
        "aptitude_q":  aptitude_q,
        "technical_q": total_q - aptitude_q,
        "diff_split":  diff,
        "time_per_q":  round(duration / total_q, 1),
    }


def build_full_test_prompt(language, duration, test_type="GATE", include_solutions=False) -> str:
    cfg = _calc_question_count(duration, test_type)
    aq, tq, tm = cfg["aptitude_q"], cfg["technical_q"], cfg["total_marks"]
    sol = "Show correct answer after each question." if include_solutions else "No answers — question paper only."
    topics = (
        "Aptitude, ECE/CS (data structures, OS, networking, VLSI, digital circuits), coding."
        if test_type == "PLACEMENT"
        else "GATE ECE: Networks, Devices, Analog, Digital, Signals, Control, Comms, EM, Maths, Aptitude."
    )
    lang_note = "Write entire paper in Hindi." if language == "Hindi" else ""
    # strong instruction to generate exactly cfg['total_q'] questions, numbered continuously, each on
    # its own line/block. avoid any prose or explanation, just the question list.
    return (
        f"You are a {test_type} expert. Produce exactly {cfg['total_q']} distinct exam questions "
        f"following the indicated sections, with no extra text before or after. {lang_note}\n"
        f"Time:{duration}min Questions:{cfg['total_q']} Marks:{tm} Difficulty:{cfg['diff_split']}\n"
        f"Section A(Aptitude):{aq}Q  Section B(Technical):{tq}Q\n"
        f"Topics:{topics}\n{sol}\n"
        f"IMPORTANT: Enumerate questions sequentially starting Q1 and ending Q{cfg['total_q']}."
        f" Use the format Qn. <question>. For MCQs include options labelled (A)(B)(C)(D) on separate lines."
        f" For NAT/Numerical, give only the number. Do NOT add any summaries, explanations or headings."
        f" If you produce fewer or more questions than {cfg['total_q']}, the output will be rejected.\n"
        f"Start with Q1 immediately."
    )


class GateQuestionRequest(BaseModel):
    subject:       str
    topic:         str
    difficulty:    str
    qtype:         str
    language:      str  = "English"
    include_audio: bool = True

    @field_validator("difficulty")
    @classmethod
    def val_diff(cls, v):
        if v.lower() not in {"easy","medium","hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        return v.lower()

    @field_validator("qtype")
    @classmethod
    def val_qtype(cls, v):
        if v.upper() not in {"MCQ","MSQ","NUMERICAL","NAT"}:
            raise ValueError("qtype must be MCQ, MSQ, NUMERICAL, or NAT")
        return v.upper()

    @field_validator("language")
    @classmethod
    def val_lang(cls, v):
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v


class FullTestRequest(BaseModel):
    language:          str  = "English"
    duration_minutes:  int  = 60
    include_solutions: bool = False
    test_type:         str  = "GATE"

    @field_validator("language")
    @classmethod
    def val_lang(cls, v):
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v

    @field_validator("duration_minutes")
    @classmethod
    def val_dur(cls, v):
        if v < 10:  raise ValueError("Minimum 10 minutes")
        if v > 180: raise ValueError("Maximum 180 minutes")
        return v

    @field_validator("test_type")
    @classmethod
    def val_type(cls, v):
        if v.upper() not in {"GATE","PLACEMENT"}:
            raise ValueError("test_type must be GATE or PLACEMENT")
        return v.upper()


class InterviewRequest(BaseModel):
    role:          str
    domain:        str
    language:      str  = "English"
    difficulty:    str  = "medium"
    include_audio: bool = True

    @field_validator("language")
    @classmethod
    def val_lang(cls, v):
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v

    @field_validator("difficulty")
    @classmethod
    def val_diff(cls, v):
        if v.lower() not in {"easy","medium","hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        return v.lower()


class PerformanceRequest(BaseModel):
    attempts: Dict[str, float]
    user_id:  Optional[str] = None

    @field_validator("attempts")
    @classmethod
    def val_attempts(cls, v):
        for t, s in v.items():
            if not (0 <= s <= 1):
                raise ValueError(f"Score for '{t}' must be 0.0 to 1.0")
        return v


class MathValidationRequest(BaseModel):
    expression: str
    variable:   str = "x"


@app.get("/")
def home():
    return {
        "message":   "AI GATE ECE + Interview Assistant API",
        "version":   "3.3.0",
        "model":     MODEL,
        "status":    "running",
        "languages": list(LANG_MAP.keys()),
        "docs":      "/docs",
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
        "subjects":       GATE_SUBJECTS,
        "total_subjects": len(GATE_SUBJECTS),
        "total_topics":   sum(len(t) for t in GATE_SUBJECTS.values()),
    }

@app.post("/generate-gate-question")
async def generate_gate_question(req: GateQuestionRequest):
    try:
        prompt      = build_gate_prompt(req.subject, req.topic, req.difficulty, req.qtype, req.language)
        ai_response = await generate_ai_response(prompt)
        if ai_response.startswith("Error"):
            raise HTTPException(500, detail=ai_response)
        audio_file = await generate_audio(ai_response, req.language) if req.include_audio else None
        return {
            "success":       True,
            "subject":       req.subject,
            "topic":         req.topic,
            "difficulty":    req.difficulty,
            "question_type": req.qtype,
            "language":      req.language,
            "model":         MODEL,
            "content":       ai_response,
            "audio_file":    audio_file,
            "timestamp":     datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/generate-full-test")
async def generate_full_test(req: FullTestRequest):
    try:
        cfg    = _calc_question_count(req.duration_minutes, req.test_type)
        prompt = build_full_test_prompt(req.language, req.duration_minutes, req.test_type, req.include_solutions)
        ai_response = await generate_ai_response(prompt)
        if ai_response.startswith("Error"):
            raise HTTPException(500, detail=ai_response)
        return {
            "success":                       True,
            "language":                      req.language,
            "model":                         MODEL,
            "test_type":                     req.test_type,
            "duration_minutes":              req.duration_minutes,
            "total_questions":               cfg["total_q"],
            "total_marks":                   cfg["total_marks"],
            "difficulty_split":              cfg["diff_split"],
            "avg_time_per_question_minutes": cfg["time_per_q"],
            "include_solutions":             req.include_solutions,
            "question_paper":                ai_response,
            "timestamp":                     datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/interview-agent")
async def interview_agent(req: InterviewRequest):
    try:
        prompt      = build_interview_prompt(req.role, req.domain, req.difficulty, req.language)
        ai_response = await generate_ai_response(prompt)
        if ai_response.startswith("Error"):
            raise HTTPException(500, detail=ai_response)
        audio_file = await generate_audio(ai_response, req.language) if req.include_audio else None
        return {
            "success":           True,
            "role":              req.role,
            "domain":            req.domain,
            "difficulty":        req.difficulty,
            "language":          req.language,
            "model":             MODEL,
            "interview_content": ai_response,
            "audio_file":        audio_file,
            "timestamp":         datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/performance-analytics")
def performance_analytics(req: PerformanceRequest):
    if not req.attempts:
        raise HTTPException(400, detail="No attempts data provided")
    strengths, moderate, weaknesses = [], [], []
    for topic, score in req.attempts.items():
        entry = {"topic": topic, "score": score}
        if   score >= 0.75: strengths.append(entry)
        elif score >= 0.50: moderate.append(entry)
        else:               weaknesses.append(entry)
    readiness = sum(req.attempts.values()) / len(req.attempts)
    level = (
        "Excellent"  if readiness >= 0.8 else
        "Good"       if readiness >= 0.7 else
        "Average"    if readiness >= 0.5 else
        "Needs Work"
    )
    recs = []
    if   readiness < 0.5: recs.append("Significant improvement needed.")
    elif readiness < 0.7: recs.append("Good progress — focus on weak areas.")
    else:                 recs.append("Strong level — focus on speed and accuracy.")
    if weaknesses:
        recs.append(f"Prioritize: {', '.join(w['topic'] for w in weaknesses[:3])}")
    return {
        "success":         True,
        "user_id":         req.user_id,
        "readiness_score": round(readiness, 3),
        "readiness_level": level,
        "strengths":       strengths,
        "moderate":        moderate,
        "weaknesses":      weaknesses,
        "recommendations": recs,
        "timestamp":       datetime.now().isoformat(),
    }

@app.post("/validate-math")
def validate_math(req: MathValidationRequest):
    try:
        var       = symbols(req.variable)
        expr      = sympify(req.expression)
        solutions = solve(expr, var)
        return {
            "success":    True,
            "expression": str(expr),
            "variable":   req.variable,
            "solutions":  [str(s) for s in solutions],
            "timestamp":  datetime.now().isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/audio/{filename}")
def get_audio(filename: str):
    filepath = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg")
    raise HTTPException(404, detail="Audio file not found")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AI GATE ECE + Interview Assistant  v3.3.0")
    print(f"  Model : {MODEL}  (~5-10 sec/response)")
    print("  Langs : English | Hindi")
    print("  API   : http://127.0.0.1:8000")
    print("  Docs  : http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)