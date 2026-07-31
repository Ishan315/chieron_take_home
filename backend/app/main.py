from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-enabled backend service converting clinical trials queries into structured visualization specifications backed by ClinicalTrials.gov API data."
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "has_openai_key": bool(settings.OPENAI_API_KEY),
        "has_gemini_key": bool(settings.GEMINI_API_KEY)
    }
