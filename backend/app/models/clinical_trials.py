from typing import List, Optional
from pydantic import BaseModel, Field

class StudyIntervention(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None

class NormalizedStudy(BaseModel):
    nct_id: str
    brief_title: str
    official_title: Optional[str] = None
    overall_status: str = "UNKNOWN"
    start_year: Optional[int] = None
    completion_year: Optional[int] = None
    lead_sponsor: str = "Unknown Sponsor"
    lead_sponsor_class: str = "OTHER"
    collaborators: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    interventions: List[StudyIntervention] = Field(default_factory=list)
    phases: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    enrollment: Optional[int] = None
