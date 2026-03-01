# AMD_HACKATHON<br>
AI-Powered Engineering Upskilling Platform<br>
GATE ECE preparation and placement interview training — powered by LLaMA 3 running locally on AMD hardware.<br>
Overview<br>
An on-device AI platform that generates GATE ECE questions, full exam papers, and technical interview questions on demand. Unlike static question banks, every question is generated fresh by LLaMA 3 based on the selected subject, topic, difficulty, and question type — with step-by-step solutions and bilingual support.
Built for the AMD Slingshot Hackathon | Team Intelligentia<br>
Features<br>
GATE Question Generator<br>
Generates MCQ, MSQ, NAT, and Numerical questions across 8 ECE subjects and 32 topics with step-by-step solutions, key concepts, common mistakes, and exam info. Supports Easy, Medium, and Hard difficulty.<br>
Full Test Generator<br>
Creates complete exam papers scaled to any duration from 10 to 180 minutes. Automatically adjusts question count and difficulty split. Supports GATE and Placement modes with optional answer key.<br>
Interview Prep Agent<br>
Role-based technical interview questions for VLSI, Embedded, Signal Processing, and other ECE roles. Each question includes a model answer, follow-up questions, evaluation criteria, and tips.<br>
Performance Analytics<br>
Takes topic-wise scores and classifies them into Strengths, Moderate, and Weaknesses. Returns an overall readiness score with personalised study recommendations.
Math Solver<br>
Solves algebraic and trigonometric expressions using SymPy. Useful for verifying NAT answers instantly.
Bilingual Audio<br>
All generated content available in English and Hindi with audio output via gTTS.
Tech Stack<br>
Backend: FastAPI, Python 3.11<br>
AI Model: LLaMA 3 via Ollama (local inference on AMD GPU)<br>
Frontend: Next.js<br>
Math Engine: SymPy<br>
Audio: gTTS<br>
HTTP Client: httpx (async)<br>
Setup<br>
# Install dependencies<br>
pip install fastapi uvicorn pydantic gtts sympy httpx python-multipart<br>

# Pull model<br>
ollama pull llama3.2:3b<br>

# Run<br>
ollama serve<br>
python main.py<br>
API: http://localhost:8000<br>
Docs: http://localhost:8000/docs<br>
API Endpoints<br>
Endpoint<br>
Method<br>
Description<br>
/generate-gate-question<br>
POST<br>
Generate a GATE ECE question<br>
/generate-full-test<br>
POST<br>
Generate a complete exam paper<br>
/interview-agent<br>
POST<br>
Generate an interview question<br>
/performance-analytics<br>
POST<br>
Analyse topic-wise performance<br>
/validate-math<br>
POST<br>
Solve a math expression<br>
/audio/{filename}<br>
GET<br>
Get generated audio file<br>
/subjects<br>
GET<br>
List all subjects and topics<br>
/health<br>
GET<br>
Health check<br>
AMD Hardware<br>
LLaMA 3 runs entirely on-device via AMD GPU with ROCm acceleration. No cloud API calls, no external data transfer, no per-request cost. Enables real-time generation with full student data privacy.<br>
Team<br>
Team Intelligentia — AMD Slingshot Hackathon<br>
Problem Statement: AI in Education and Skilling<br>
Team Leader: Pranavsankar Gopalakrishnan<br>
