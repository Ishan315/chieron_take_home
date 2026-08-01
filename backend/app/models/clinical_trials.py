from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class StudyIdentification(BaseModel):
    nct_id: str
    brief_title: Optional[str] = None
    official_title: Optional[str] = None

class StudyStatus(BaseModel):
    overall_status: Optional[str] = None
    start_date: Optional[str] = None
    start_date_year: Optional[int] = None
    completion_date: Optional[str] = None
    completion_date_year: Optional[int] = None

class StudySponsor(BaseModel):
    lead_sponsor_name: Optional[str] = None
    lead_sponsor_class: Optional[str] = None
    collaborators: List[str] = Field(default_factory=list)

class StudyIntervention(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None

class StudyDesign(BaseModel):
    phases: List[str] = Field(default_factory=list)
    enrollment_count: Optional[int] = None
    allocation: Optional[str] = None
    masking: Optional[str] = None
    primary_purpose: Optional[str] = None

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
    raw_protocol: Dict[str, Any] = Field(default_factory=dict)
