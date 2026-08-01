from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to this file's location (backend/app/core/config.py -> project root)
# rather than a bare relative ".env", so the key loads regardless of the
# working directory the server happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    PROJECT_NAME: str = "ClinicalTrials Query-to-Visualization Agent API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # ClinicalTrials.gov API v2 Base URL
    CLINICAL_TRIALS_API_BASE: str = "https://clinicaltrials.gov/api/v2"

    # OpenAI API Key (optional; system includes deterministic NLP fallback engine)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Default search settings
    DEFAULT_MAX_TRIALS: int = 200

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
