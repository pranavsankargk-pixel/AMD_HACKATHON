# AMD_HACKATHON
AI-Powered Engineering Upskilling Platform
GATE ECE preparation and placement interview training — powered by LLaMA 3 running locally on AMD hardware.
Overview
An on-device AI platform that generates GATE ECE questions, full exam papers, and technical interview questions on demand. Unlike static question banks, every question is generated fresh by LLaMA 3 based on the selected subject, topic, difficulty, and question type — with step-by-step solutions and bilingual support.
Built for the AMD Slingshot Hackathon | Team Intelligentia
Features
GATE Question Generator
Generates MCQ, MSQ, NAT, and Numerical questions across 8 ECE subjects and 32 topics with step-by-step solutions, key concepts, common mistakes, and exam info. Supports Easy, Medium, and Hard difficulty.
Full Test Generator
Creates complete exam papers scaled to any duration from 10 to 180 minutes. Automatically adjusts question count and difficulty split. Supports GATE and Placement modes with optional answer key.
Interview Prep Agent
Role-based technical interview questions for VLSI, Embedded, Signal Processing, and other ECE roles. Each question includes a model answer, follow-up questions, evaluation criteria, and tips.
Performance Analytics
Takes topic-wise scores and classifies them into Strengths, Moderate, and Weaknesses. Returns an overall readiness score with personalised study recommendations.
Math Solver
Solves algebraic and trigonometric expressions using SymPy. Useful for verifying NAT answers instantly.
Bilingual Audio
All generated content available in English and Hindi with audio output via gTTS.
Tech Stack
Backend: FastAPI, Python 3.11
AI Model: LLaMA 3 via Ollama (local inference on AMD GPU)
Frontend: Next.js
Math Engine: SymPy
Audio: gTTS
HTTP Client: httpx (async)
Setup
# Install dependencies
pip install fastapi uvicorn pydantic gtts sympy httpx python-multipart

# Pull model
ollama pull llama3.2:3b

# Run
ollama serve
python main.py
API: http://localhost:8000
Docs: http://localhost:8000/docs
API Endpoints
Endpoint
Method
Description
/generate-gate-question
POST
Generate a GATE ECE question
/generate-full-test
POST
Generate a complete exam paper
/interview-agent
POST
Generate an interview question
/performance-analytics
POST
Analyse topic-wise performance
/validate-math
POST
Solve a math expression
/audio/{filename}
GET
Get generated audio file
/subjects
GET
List all subjects and topics
/health
GET
Health check
AMD Hardware
LLaMA 3 runs entirely on-device via AMD GPU with ROCm acceleration. No cloud API calls, no external data transfer, no per-request cost. Enables real-time generation with full student data privacy.
Team
Team Intelligentia — AMD Slingshot Hackathon
Problem Statement: AI in Education and Skilling
Team Leader: Pranavsankar Gopalakrishnan
