from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Dict, Optional
from datetime import datetime
import uuid

app = FastAPI(
    title="AI GATE ECE + Interview Assistant (Demo)",
    description="Demo API for frontend development. Real AI responses via local server.",
    version="3.1.0-demo"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LANG_MAP = {"English": {"code": "en"}, "Hindi": {"code": "hi"}}

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

# ── Sample responses (realistic shape) ─────────────────────────────

SAMPLE_GATE_EN = """**1. QUESTION**
A discrete-time signal x[n] = cos(0.2πn) is sampled. What is the fundamental period of this signal?

**2. OPTIONS**
(A) 5
(B) 10
(C) 20
(D) The signal is aperiodic

**3. CORRECT ANSWER**
(B) 10

**4. STEP-BY-STEP SOLUTION**
For a discrete-time sinusoid x[n] = cos(ω₀n), the signal is periodic if ω₀/2π = k/N for integers k, N.
Here ω₀ = 0.2π
ω₀/2π = 0.2π/2π = 0.1 = 1/10
So N = 10 (with k = 1). The fundamental period is N = 10.

**5. KEY CONCEPT**
A discrete-time sinusoid cos(ω₀n) is periodic only if ω₀/2π is a rational number p/q in lowest terms. The fundamental period is then N = q. Unlike continuous-time signals, not all discrete sinusoids are periodic.

**6. COMMON MISTAKES**
- Confusing continuous-time period (T = 2π/ω₀) with discrete-time period.
- Forgetting to reduce p/q to lowest terms before reading N.
- Assuming all sinusoids are periodic — in discrete time, irrational ω₀/2π gives aperiodic signals.

**7. EXAM INFO**
Marks: 2 | Time: 2 minutes | Negative marking: Yes (-0.67)

**8. RELATED TOPICS**
Z-Transform, Sampling Theorem, DFT"""

SAMPLE_GATE_HI = """**1. प्रश्न**
एक discrete-time संकेत x[n] = cos(0.2πn) दिया गया है। इस संकेत की मौलिक अवधि (fundamental period) क्या है?

**2. विकल्प**
(A) 5
(B) 10
(C) 20
(D) संकेत अवधिक नहीं है

**3. सही उत्तर**
(B) 10

**4. चरण-दर-चरण हल**
Discrete-time sinusoid x[n] = cos(ω₀n) के लिए, संकेत तब आवधिक होता है जब ω₀/2π = k/N हो।
यहाँ ω₀ = 0.2π
ω₀/2π = 0.1 = 1/10
अतः N = 10 (k=1 के साथ)। मौलिक अवधि N = 10 है।

**5. मुख्य अवधारणा**
Discrete-time sinusoid cos(ω₀n) तब आवधिक होता है जब ω₀/2π एक परिमेय संख्या हो। यदि ω₀/2π = p/q (न्यूनतम रूप में) हो, तो मौलिक अवधि N = q होती है।

**6. सामान्य गलतियाँ**
- Continuous-time और discrete-time अवधि को आपस में भ्रमित करना।
- p/q को न्यूनतम रूप में न लाना।
- यह मान लेना कि सभी sinusoids आवधिक होते हैं।

**7. परीक्षा जानकारी**
अंक: 2 | समय: 2 मिनट | नकारात्मक अंकन: हाँ (-0.67)

**8. संबंधित विषय**
Z-Transform, Sampling Theorem, DFT"""

SAMPLE_TEST = """**Section A — General Aptitude** (5 Questions)

Q1. [MCQ] A train travels 60 km in 1 hour. How long will it take to travel 150 km at the same speed?
(A) 2 hours  (B) 2.5 hours  (C) 3 hours  (D) 1.5 hours

Q2. [MCQ] Choose the word most similar in meaning to ELOQUENT:
(A) Silent  (B) Fluent  (C) Angry  (D) Confused

Q3. [NAT] If 3x + 7 = 22, find x. Answer: ___

Q4. [MCQ] Complete the series: 2, 6, 12, 20, 30, ___
(A) 40  (B) 42  (C) 44  (D) 38

Q5. [MCQ] A is the father of B. B is the sister of C. How is A related to C?
(A) Uncle  (B) Father  (C) Grandfather  (D) Brother

---

**Section B — Technical ECE** (12 Questions)

Q6. [MCQ] The Fourier transform of a rectangular pulse of width τ is:
(A) sinc(fτ)  (B) τ·sinc(fτ)  (C) sinc²(fτ)  (D) δ(f)

Q7. [NAT] For a BJT in active region with β=100 and IB=20μA, find IC in mA. Answer: ___

Q8. [MCQ] Which of the following is NOT a property of an ideal op-amp?
(A) Infinite input impedance  (B) Zero output impedance
(C) Finite open-loop gain     (D) Infinite bandwidth

Q9. [MCQ] The transfer function of a system is H(s) = 1/(s+2). The system is:
(A) Unstable  (B) Stable  (C) Marginally stable  (D) Cannot determine

Q10. [MSQ] Which of the following modulation techniques are digital?
(A) ASK  (B) AM  (C) FSK  (D) PSK

Q11. [NAT] A lossless transmission line has L=0.25μH/m and C=100pF/m. Find characteristic impedance Z₀ in Ω. Answer: ___

Q12. [MCQ] The Boolean expression A·(A+B) simplifies to:
(A) A+B  (B) A  (C) B  (D) AB

Q13. [MCQ] Z-transform of u[n] (unit step) is:
(A) z/(z-1)  (B) 1/(z-1)  (C) z/(z+1)  (D) 1/z

Q14. [NAT] For a MOSFET with VGS=3V, VTH=1V, kn=2mA/V², find ID in mA (saturation). Answer: ___

Q15. [MCQ] Nyquist sampling rate for a signal with maximum frequency 4kHz is:
(A) 4 kHz  (B) 8 kHz  (C) 2 kHz  (D) 16 kHz

Q16. [MCQ] The steady-state error for a Type-1 system with ramp input is:
(A) Zero  (B) Infinite  (C) Finite non-zero  (D) Depends on gain

Q17. [MSQ] Maxwell's equations include:
(A) Gauss's law for electric field  (B) Faraday's law
(C) Newton's second law            (D) Ampere's law with Maxwell's addition"""

SAMPLE_INTERVIEW = """**1. INTERVIEW QUESTION**
Explain the difference between a latch and a flip-flop. When would you use each in a digital design?

**2. MODEL ANSWER / KEY POINTS**
A latch is a level-sensitive storage element — it is transparent and changes output while the enable/clock signal is active (high or low). A flip-flop is edge-triggered — it samples input only at the rising or falling edge of the clock, making it predictable and suitable for synchronous design.

Key points to mention:
- Latches: faster, less area, but cause timing hazards in synchronous circuits
- Flip-flops: used in registers, counters, state machines — standard in VLSI synchronous design
- D-latch vs D flip-flop comparison is the most common interview example
- Metastability is a risk when async signals cross clock domains

**3. FOLLOW-UP QUESTIONS**
- What is metastability and how do you resolve it using flip-flops?
- Draw the timing diagram of a D flip-flop with setup and hold time violations.
- How does a master-slave flip-flop work internally?

**4. EVALUATION CRITERIA**
- Can clearly articulate level-sensitive vs edge-triggered difference (30%)
- Knows practical use cases — not just definitions (30%)
- Mentions timing hazards or setup/hold time (20%)
- Can draw or describe a circuit (20%)

**5. COMMON CANDIDATE MISTAKES**
- Saying latches and flip-flops are the same thing
- Not mentioning edge-triggering as the key difference
- Unable to explain why flip-flops are preferred in synchronous design

**6. TIPS FOR THE CANDIDATE**
Start by stating the core difference (level vs edge), give a real-world use case for each, then offer to draw a timing diagram. This shows both theory and practical design sense."""

SAMPLE_ANALYTICS = {
    "readiness_score": 0.67,
    "readiness_level": "Average",
    "strengths": [
        {"topic": "Fourier Transform", "score": 0.85},
        {"topic": "Boolean Algebra",   "score": 0.80},
    ],
    "moderate": [
        {"topic": "Control Systems", "score": 0.65},
        {"topic": "Op-Amp",          "score": 0.60},
    ],
    "weaknesses": [
        {"topic": "BJT",              "score": 0.40},
        {"topic": "Maxwell Equations","score": 0.35},
    ],
    "recommendations": [
        "Good progress — focus on moderate and weak areas.",
        "Prioritize these weak topics: BJT, Maxwell Equations",
    ],
}

# ── Pydantic Models ─────────────────────────────────────────────────

class GateQuestionRequest(BaseModel):
    subject: str
    topic: str
    difficulty: str
    qtype: str
    language: str = "English"
    include_audio: bool = False

    @field_validator("difficulty")
    @classmethod
    def val_diff(cls, v):
        if v.lower() not in {"easy","medium","hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        return v.lower()

    @field_validator("qtype")
    @classmethod
    def val_qtype(cls, v):
        if v.upper() not in {"MCQ","MSQ","NAT","NUMERICAL"}:
            raise ValueError("qtype must be MCQ, MSQ, NAT, or NUMERICAL")
        return v.upper()

    @field_validator("language")
    @classmethod
    def val_lang(cls, v):
        if v not in LANG_MAP:
            raise ValueError("language must be English or Hindi")
        return v

class FullTestRequest(BaseModel):
    language: str = "English"
    duration_minutes: int = 60
    include_solutions: bool = False
    test_type: str = "GATE"

    @field_validator("duration_minutes")
    @classmethod
    def val_dur(cls, v):
        if v < 10:  raise ValueError("Minimum 10 minutes")
        if v > 180: raise ValueError("Maximum 180 minutes")
        return v

    @field_validator("language")
    @classmethod
    def val_lang(cls, v):
        if v not in LANG_MAP: raise ValueError("English or Hindi only")
        return v

    @field_validator("test_type")
    @classmethod
    def val_type(cls, v):
        if v.upper() not in {"GATE","PLACEMENT"}: raise ValueError("GATE or PLACEMENT only")
        return v.upper()

class InterviewRequest(BaseModel):
    role: str
    domain: str
    language: str = "English"
    difficulty: str = "medium"
    include_audio: bool = False

    @field_validator("language")
    @classmethod
    def val_lang(cls, v):
        if v not in LANG_MAP: raise ValueError("English or Hindi only")
        return v

    @field_validator("difficulty")
    @classmethod
    def val_diff(cls, v):
        if v.lower() not in {"easy","medium","hard"}: raise ValueError("easy/medium/hard")
        return v.lower()

class PerformanceRequest(BaseModel):
    attempts: Dict[str, float]
    user_id: Optional[str] = None

    @field_validator("attempts")
    @classmethod
    def val_scores(cls, v):
        for t, s in v.items():
            if not (0 <= s <= 1): raise ValueError(f"Score for {t} must be 0-1")
        return v

class MathValidationRequest(BaseModel):
    expression: str
    variable: str = "x"

# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "AI GATE ECE + Interview Assistant API",
        "version": "3.1.0-demo",
        "mode": "DEMO — connect to local server for real AI responses",
        "status": "running",
        "languages": list(LANG_MAP.keys()),
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "demo", "timestamp": datetime.now().isoformat()}

@app.get("/languages")
def languages():
    return {"languages": list(LANG_MAP.keys())}

@app.get("/subjects")
def subjects():
    return {
        "subjects": GATE_SUBJECTS,
        "total_subjects": len(GATE_SUBJECTS),
        "total_topics": sum(len(t) for t in GATE_SUBJECTS.values()),
    }

@app.post("/generate-gate-question")
def generate_gate_question(req: GateQuestionRequest):
    content = SAMPLE_GATE_HI if req.language == "Hindi" else SAMPLE_GATE_EN
    return {
        "success": True,
        "subject": req.subject,
        "topic": req.topic,
        "difficulty": req.difficulty,
        "question_type": req.qtype,
        "language": req.language,
        "model": "llama3",
        "mode": "demo",
        "content": content,
        "audio_file": None,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/generate-full-test")
def generate_full_test(req: FullTestRequest):
    total_q = min(65, max(5, round(req.duration_minutes / 1.8))) if req.test_type == "GATE" else max(5, round(req.duration_minutes / 2))
    return {
        "success": True,
        "language": req.language,
        "model": "llama3",
        "mode": "demo",
        "test_type": req.test_type,
        "duration_minutes": req.duration_minutes,
        "total_questions": total_q,
        "total_marks": round(total_q * 1.5),
        "difficulty_split": "Easy 40% / Medium 40% / Hard 20%",
        "avg_time_per_question_minutes": round(req.duration_minutes / total_q, 1),
        "include_solutions": req.include_solutions,
        "question_paper": SAMPLE_TEST,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/interview-agent")
def interview_agent(req: InterviewRequest):
    return {
        "success": True,
        "role": req.role,
        "domain": req.domain,
        "difficulty": req.difficulty,
        "language": req.language,
        "model": "llama3",
        "mode": "demo",
        "interview_content": SAMPLE_INTERVIEW,
        "audio_file": None,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/performance-analytics")
def performance_analytics(req: PerformanceRequest):
    if not req.attempts:
        raise HTTPException(400, "No attempts data provided")
    strengths, moderate, weaknesses = [], [], []
    for topic, score in req.attempts.items():
        entry = {"topic": topic, "score": score}
        if score >= 0.75:   strengths.append(entry)
        elif score >= 0.50: moderate.append(entry)
        else:               weaknesses.append(entry)
    readiness = sum(req.attempts.values()) / len(req.attempts)
    level = ("Excellent" if readiness >= 0.8 else "Good" if readiness >= 0.7
             else "Average" if readiness >= 0.5 else "Needs Work")
    recs = []
    if readiness < 0.5:   recs.append("Significant improvement needed.")
    elif readiness < 0.7: recs.append("Good progress — focus on weak areas.")
    else:                 recs.append("Strong level — focus on speed and accuracy.")
    if weaknesses:
        recs.append(f"Prioritize: {', '.join(w['topic'] for w in weaknesses[:3])}")
    return {
        "success": True,
        "user_id": req.user_id,
        "readiness_score": round(readiness, 3),
        "readiness_level": level,
        "strengths": strengths,
        "moderate": moderate,
        "weaknesses": weaknesses,
        "recommendations": recs,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/validate-math")
def validate_math(req: MathValidationRequest):
    # Real SymPy solving — this works without Ollama
    try:
        from sympy import symbols, solve, sympify
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
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AI GATE ECE Assistant — DEMO SERVER")
    print("  All endpoints active with sample responses")
    print("  API:  http://127.0.0.1:8000")
    print("  Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)