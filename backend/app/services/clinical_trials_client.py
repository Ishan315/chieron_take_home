import re
import logging
from typing import List, Dict, Any, Optional
import httpx
from app.core.config import settings
from app.models.clinical_trials import NormalizedStudy, StudyIntervention

logger = logging.getLogger(__name__)

class ClinicalTrialsClient:
    """
    Client for interacting with ClinicalTrials.gov API v2.
    Official Endpoint: https://clinicaltrials.gov/api/v2/studies
    """

    # No real drug/condition/sponsor name comes close to this. Query/NLP
    # extraction (regex fallback or LLM) can occasionally produce a runaway
    # capture from degenerate input; sending that straight through as a query
    # param has been observed to trigger a live 414 Request-URI Too Large from
    # the upstream API. Treat anything past this length as implausible and drop
    # it rather than let a malformed filter value take down the whole request.
    MAX_FILTER_VALUE_LENGTH = 150

    def __init__(self, base_url: str = settings.CLINICAL_TRIALS_API_BASE):
        self.base_url = base_url.rstrip("/")

    def _clean_filter_value(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        value = value.strip()
        if not value or len(value) > self.MAX_FILTER_VALUE_LENGTH:
            if value:
                logger.warning(
                    f"Dropping implausibly long filter value ({len(value)} chars): {value[:60]}..."
                )
            return None
        return value

    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        match = re.search(r'\b(19\d\d|20\d\d)\b', str(date_str))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    def _normalize_phase(self, phase_str: str) -> str:
        p = phase_str.upper().strip()
        phase_map = {
            "EARLY_PHASE1": "Early Phase 1",
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
            "NA": "Not Applicable"
        }
        return phase_map.get(p, phase_str.title())

    def _parse_study_item(self, study_raw: Dict[str, Any]) -> NormalizedStudy:
        protocol = study_raw.get("protocolSection", {})
        
        # 1. Identification
        id_mod = protocol.get("identificationModule", {})
        nct_id = id_mod.get("nctId", "UNKNOWN")
        brief_title = id_mod.get("briefTitle", "Untitled Study")
        official_title = id_mod.get("officialTitle")
        
        # 2. Status & Dates
        status_mod = protocol.get("statusModule", {})
        overall_status = status_mod.get("overallStatus", "UNKNOWN")
        
        start_date_struct = status_mod.get("startDateStruct", {})
        start_date_raw = start_date_struct.get("date")
        start_year = self._extract_year(start_date_raw)
        
        completion_date_struct = status_mod.get("completionDateStruct") or status_mod.get("primaryCompletionDateStruct") or {}
        completion_date_raw = completion_date_struct.get("date")
        completion_year = self._extract_year(completion_date_raw)
        
        # 3. Sponsor & Collaborators
        spon_mod = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor_struct = spon_mod.get("leadSponsor", {})
        lead_sponsor = lead_sponsor_struct.get("name", "Unknown Sponsor")
        lead_sponsor_class = lead_sponsor_struct.get("class", "OTHER")
        
        collaborators = [
            c.get("name") for c in spon_mod.get("collaborators", []) if c.get("name")
        ]
        
        # 4. Conditions
        cond_mod = protocol.get("conditionsModule", {})
        conditions = cond_mod.get("conditions", [])
        
        # 5. Interventions
        arms_mod = protocol.get("armsInterventionsModule", {})
        interventions_raw = arms_mod.get("interventions", [])
        interventions = []
        for item in interventions_raw:
            i_name = item.get("name")
            if i_name:
                interventions.append(
                    StudyIntervention(
                        name=i_name,
                        type=item.get("type"),
                        description=item.get("description")
                    )
                )
                
        # 6. Phases & Design
        design_mod = protocol.get("designModule", {})
        phases_raw = design_mod.get("phases", [])
        phases = [self._normalize_phase(p) for p in phases_raw] if phases_raw else ["Not Applicable"]
        
        enroll_struct = design_mod.get("enrollmentInfo", {})
        enrollment = enroll_struct.get("count")
        
        # 7. Locations & Countries
        loc_mod = protocol.get("contactsLocationsModule", {})
        locations = loc_mod.get("locations", [])
        countries_set = set()
        for loc in locations:
            c = loc.get("country")
            if c:
                countries_set.add(c)
        countries = sorted(list(countries_set))
        
        return NormalizedStudy(
            nct_id=nct_id,
            brief_title=brief_title,
            official_title=official_title,
            overall_status=overall_status,
            start_year=start_year,
            completion_year=completion_year,
            lead_sponsor=lead_sponsor,
            lead_sponsor_class=lead_sponsor_class,
            collaborators=collaborators,
            conditions=conditions,
            interventions=interventions,
            phases=phases,
            countries=countries,
            enrollment=enrollment
        )

    async def fetch_studies(
        self,
        condition: Optional[str] = None,
        term: Optional[str] = None,
        location: Optional[str] = None,
        sponsor: Optional[str] = None,
        status: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        max_results: int = 100
    ) -> List[NormalizedStudy]:
        """
        Fetch studies from ClinicalTrials.gov API v2 matching parameters.
        """
        condition = self._clean_filter_value(condition)
        term = self._clean_filter_value(term)
        location = self._clean_filter_value(location)
        sponsor = self._clean_filter_value(sponsor)

        params: Dict[str, Any] = {
            "pageSize": min(max_results, 100)
        }

        if condition:
            params["query.cond"] = condition
        if term:
            params["query.term"] = term
        if location:
            params["query.locn"] = location
        if sponsor:
            params["query.spons"] = sponsor
        if status:
            params["filter.overallStatus"] = status.upper()
        if start_year or end_year:
            lower = f"{start_year}-01-01" if start_year else "MIN"
            upper = f"{end_year}-12-31" if end_year else "MAX"
            params["filter.advanced"] = f"AREA[StartDate]RANGE[{lower},{upper}]"

        studies: List[NormalizedStudy] = []
        page_token: Optional[str] = None
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            while len(studies) < max_results:
                if page_token:
                    params["pageToken"] = page_token
                    
                try:
                    resp = await client.get(f"{self.base_url}/studies", params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"Error fetching ClinicalTrials.gov API data: {e}")
                    break
                    
                raw_studies = data.get("studies", [])
                if not raw_studies:
                    break
                    
                for raw in raw_studies:
                    parsed = self._parse_study_item(raw)
                    studies.append(parsed)
                    if len(studies) >= max_results:
                        break
                        
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                    
        return studies
